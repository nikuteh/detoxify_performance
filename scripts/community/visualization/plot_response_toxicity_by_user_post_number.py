import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"
COMMENT_ID_COLUMNS = ("comment_id", "id", "name")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample active users, take each user's first comments in "
            "chronological order, and plot the average toxicity of direct "
            "replies by user post/comment number."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Community Detoxify predictions CSV containing comment ids, "
            "parent ids, usernames, timestamps, and toxicity scores."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV containing the per-post-number averages.",
    )
    parser.add_argument(
        "--sampled-comments-output",
        type=Path,
        help=(
            "Optional output CSV containing each sampled comment and its "
            "average direct-reply toxicity."
        ),
    )
    parser.add_argument(
        "--users",
        type=int,
        default=100,
        help="Number of eligible users to randomly sample. Default: 100.",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=200,
        help="Minimum comments a user needs to be eligible. Default: 200.",
    )
    parser.add_argument(
        "--comments-per-user",
        type=int,
        default=500,
        help=(
            "Maximum chronological comments to keep per sampled user. "
            "Default: 500."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for user sampling. Default: 42.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Reply toxicity score column to average. Default: toxicity.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing Unix timestamps. Default: timestamp.",
    )
    parser.add_argument(
        "--comment-id-column",
        help=(
            "Column containing the original comment id. Default: auto-detect "
            "one of comment_id, id, or name."
        ),
    )
    parser.add_argument(
        "--parent-id-column",
        default="parent_id",
        help="Column containing Reddit parent ids. Default: parent_id.",
    )
    parser.add_argument(
        "--min-comments-per-post",
        type=int,
        default=1,
        help=(
            "Minimum sampled original comments with replies required for a "
            "post/comment number to be plotted. Default: 1."
        ),
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        help="Only plot post/comment numbers up to this value.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames.",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Default: inferred from the input CSV filename.",
    )
    return parser.parse_args()


def infer_visualization_output(input_csv, filename):
    try:
        relative_input = input_csv.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        return input_csv.with_name(filename)

    if not relative_input.parts:
        return input_csv.with_name(filename)

    subreddit_name = relative_input.parts[0]
    return VISUALIZATION_SUBREDDITS_DIR / subreddit_name / filename


def resolve_comment_id_column(columns, requested_column):
    if requested_column:
        if requested_column not in columns:
            raise SystemExit(
                f"Input CSV is missing comment id column: {requested_column}"
            )
        return requested_column

    for column in COMMENT_ID_COLUMNS:
        if column in columns:
            return column

    raise SystemExit(
        "Input CSV needs a comment id column to match replies. Expected one "
        "of: comment_id, id, name. Re-run format_reddit_comments_zst.py with "
        "the current version, or pass --comment-id-column for your CSV."
    )


def normalize_reddit_comment_id(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("t1_"):
        return text
    if text.startswith("t3_"):
        return ""
    return f"t1_{text}"


def normalize_parent_id(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text.startswith("t1_"):
        return text
    return ""


def load_comments(args, comment_id_column):
    usecols = [
        comment_id_column,
        args.parent_id_column,
        args.username_column,
        args.timestamp_column,
        args.score_column,
    ]
    df = pd.read_csv(args.input_csv, usecols=usecols)

    df[args.username_column] = (
        df[args.username_column].fillna("").astype(str).str.strip()
    )
    df = df[df[args.username_column] != ""].copy()

    if not args.include_deleted:
        deleted_names = {"[deleted]", "[removed]", "deleted", "removed"}
        df = df[~df[args.username_column].str.lower().isin(deleted_names)].copy()

    df[args.timestamp_column] = pd.to_numeric(
        df[args.timestamp_column],
        errors="coerce",
    )
    df[args.score_column] = pd.to_numeric(df[args.score_column], errors="coerce")
    df["normalized_comment_id"] = df[comment_id_column].map(
        normalize_reddit_comment_id
    )
    df["normalized_parent_id"] = df[args.parent_id_column].map(normalize_parent_id)

    df = df.dropna(subset=[args.timestamp_column])
    df = df[df["normalized_comment_id"] != ""].copy()

    if df.empty:
        raise SystemExit("No usable comment rows found after cleaning")

    return df


def sample_active_users(df, username_column, min_comments, sample_size, seed):
    comment_counts = df[username_column].value_counts()
    eligible_users = comment_counts[comment_counts >= min_comments]

    if eligible_users.empty:
        raise SystemExit(f"No users found with at least {min_comments} comments")
    if len(eligible_users) < sample_size:
        raise SystemExit(
            f"Only {len(eligible_users)} users have at least {min_comments} "
            f"comments, but --users requested {sample_size}."
        )

    sampled = eligible_users.sample(n=sample_size, random_state=seed)
    ranking = sampled.rename("comment_count").reset_index()
    ranking = ranking.rename(columns={"index": username_column})
    return ranking


def first_comments_for_users(df, sampled_users, args):
    sampled_names = set(sampled_users[args.username_column])
    selected = df[df[args.username_column].isin(sampled_names)].copy()
    selected = selected.sort_values([args.username_column, args.timestamp_column])
    selected["post_number"] = selected.groupby(args.username_column).cumcount() + 1
    selected = selected[selected["post_number"] <= args.comments_per_user].copy()

    if args.max_post_number is not None:
        selected = selected[selected["post_number"] <= args.max_post_number].copy()

    if selected.empty:
        raise SystemExit("No sampled comments left after applying filters")

    return selected


def attach_reply_toxicity(sampled_comments, all_comments, score_column):
    reply_scores = all_comments.dropna(subset=[score_column])
    reply_summary = (
        reply_scores[reply_scores["normalized_parent_id"] != ""]
        .groupby("normalized_parent_id")
        .agg(
            average_reply_toxicity=(score_column, "mean"),
            reply_count=(score_column, "size"),
        )
        .reset_index()
    )

    sampled = sampled_comments.merge(
        reply_summary,
        how="left",
        left_on="normalized_comment_id",
        right_on="normalized_parent_id",
        suffixes=("", "_reply"),
    )
    sampled["reply_count"] = sampled["reply_count"].fillna(0).astype(int)
    sampled = sampled.drop(columns=["normalized_parent_id_reply"], errors="ignore")
    return sampled


def average_reply_toxicity_by_post_number(sampled_comments, username_column):
    comments_with_replies = sampled_comments.dropna(subset=["average_reply_toxicity"])
    if comments_with_replies.empty:
        raise SystemExit("None of the sampled comments had scored direct replies")

    per_post = (
        comments_with_replies.groupby("post_number")
        .agg(
            average_reply_toxicity=("average_reply_toxicity", "mean"),
            median_reply_toxicity=("average_reply_toxicity", "median"),
            sampled_comments_with_replies=("average_reply_toxicity", "size"),
            contributing_users=(username_column, "nunique"),
            total_replies=("reply_count", "sum"),
        )
        .reset_index()
    )
    sampled_counts = (
        sampled_comments.groupby("post_number")
        .size()
        .rename("sampled_original_comments")
        .reset_index()
    )
    return per_post.merge(sampled_counts, on="post_number", how="left")


def plot_response_toxicity(per_post, output_png, title, sample_size, min_comments):
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    x = per_post["post_number"].to_numpy(dtype=int)
    y = per_post["average_reply_toxicity"].to_numpy(dtype=float)

    ax.plot(
        x,
        y,
        color="#4C78A8",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label="Average direct-reply toxicity",
    )

    if len(per_post) >= 2:
        coefficients = np.polyfit(x, y, 1)
        trend = np.poly1d(coefficients)(x)
        ax.plot(
            x,
            trend,
            color="#F58518",
            linewidth=2,
            linestyle="--",
            label="Linear trend",
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Post/comment number for each sampled user")
    ax.set_ylabel("Average toxicity of direct replies")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        f"Sampled users: {sample_size:,}; eligibility: {min_comments:,}+ comments",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.users < 1:
        raise SystemExit("--users must be at least 1")
    if args.min_comments < 1:
        raise SystemExit("--min-comments must be at least 1")
    if args.comments_per_user < 1:
        raise SystemExit("--comments-per-user must be at least 1")
    if args.min_comments_per_post < 1:
        raise SystemExit("--min-comments-per-post must be at least 1")
    if args.max_post_number is not None and args.max_post_number < 1:
        raise SystemExit("--max-post-number must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(columns, args.comment_id_column)
    required = {
        comment_id_column,
        args.parent_id_column,
        args.username_column,
        args.timestamp_column,
        args.score_column,
    }
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = load_comments(args, comment_id_column)
    sampled_users = sample_active_users(
        df,
        args.username_column,
        args.min_comments,
        args.users,
        args.seed,
    )
    sampled_comments = first_comments_for_users(df, sampled_users, args)
    sampled_comments = attach_reply_toxicity(
        sampled_comments,
        df,
        args.score_column,
    )
    per_post = average_reply_toxicity_by_post_number(
        sampled_comments,
        args.username_column,
    )
    per_post = per_post[
        per_post["sampled_comments_with_replies"] >= args.min_comments_per_post
    ].copy()

    if per_post.empty:
        raise SystemExit("No post/comment numbers met the plotting filters")

    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_response_toxicity_by_post_number.png",
    )
    title = args.title or (
        f"{args.input_csv.stem}: Response Toxicity by User Post Number"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        per_post.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    if args.sampled_comments_output:
        args.sampled_comments_output.parent.mkdir(parents=True, exist_ok=True)
        sampled_comments.to_csv(args.sampled_comments_output, index=False)
        print(f"Saved {args.sampled_comments_output}")

    plot_response_toxicity(
        per_post,
        output_png,
        title,
        len(sampled_users),
        args.min_comments,
    )

    print(f"Saved {output_png}")
    print(f"Eligible users sampled: {len(sampled_users)}")
    print(f"Sampled original comments: {len(sampled_comments)}")
    print(
        "Sampled original comments with replies: "
        f"{sampled_comments['average_reply_toxicity'].notna().sum()}"
    )
    print(f"Post/comment numbers plotted: {len(per_post)}")


if __name__ == "__main__":
    main()
