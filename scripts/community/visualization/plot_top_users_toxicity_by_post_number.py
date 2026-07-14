import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


TOXICITY_COLUMNS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For the top users in a subreddit CSV, plot average Detoxify scores "
            "by each user's post/comment number."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Subreddit CSV containing usernames, timestamps, and Detoxify scores.",
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
        "--top-n",
        type=int,
        default=100,
        help="Number of top users to include. Default: 100.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["comment_count", "average_toxicity"],
        default="comment_count",
        help="How to choose top users. Default: comment_count.",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=1,
        help="Minimum comments required for a user to be eligible. Default: 1.",
    )
    parser.add_argument(
        "--min-users-per-post",
        type=int,
        help=(
            "Minimum top users that must have a post/comment number for it to "
            "be plotted. Default: all selected top users."
        ),
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        help="Only plot post/comment numbers up to this value.",
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
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames.",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Default: inferred from the input CSV filename.",
    )
    return parser.parse_args()


def required_score_columns(columns):
    score_columns = [column for column in TOXICITY_COLUMNS if column in columns]
    if not score_columns:
        raise SystemExit(
            "Input CSV has no known Detoxify score columns. Expected one or "
            f"more of: {', '.join(TOXICITY_COLUMNS)}"
        )
    return score_columns


def load_comments(args, score_columns):
    usecols = [args.username_column, args.timestamp_column, *score_columns]
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
    for column in score_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[args.username_column, args.timestamp_column, *score_columns]
    )
    if df.empty:
        raise SystemExit("No usable comment rows found after cleaning")

    return df


def rank_top_users(df, username_column, score_columns, rank_by, min_comments, top_n):
    grouped = df.groupby(username_column)
    ranking_score_column = "toxicity" if "toxicity" in score_columns else score_columns[0]
    ranking = grouped.agg(
        comment_count=(score_columns[0], "size"),
        average_toxicity=(ranking_score_column, "mean"),
    )
    ranking = ranking[ranking["comment_count"] >= min_comments]

    if ranking.empty:
        raise SystemExit("No users met the --min-comments filter")

    if rank_by == "comment_count":
        ranking = ranking.sort_values(
            ["comment_count", "average_toxicity"],
            ascending=[False, False],
        )
    else:
        ranking = ranking.sort_values(
            ["average_toxicity", "comment_count"],
            ascending=[False, False],
        )

    ranking = ranking.head(top_n).reset_index()
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def average_scores_by_post_number(
    df,
    ranking,
    username_column,
    timestamp_column,
    score_columns,
):
    top_usernames = set(ranking[username_column])
    top_rows = df[df[username_column].isin(top_usernames)].copy()
    top_rows = top_rows.sort_values([username_column, timestamp_column])
    top_rows["post_number"] = top_rows.groupby(username_column).cumcount() + 1

    aggregations = {
        column: (column, "mean")
        for column in score_columns
    }
    per_post = (
        top_rows.groupby("post_number")
        .agg(
            contributing_posts=(score_columns[0], "size"),
            contributing_users=(username_column, "nunique"),
            **aggregations,
        )
        .reset_index()
    )
    return per_post


def plot_per_post_scores(per_post, score_columns, output_png, title, top_user_count):
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    colors = plt.get_cmap("tab10").colors
    x = per_post["post_number"].to_numpy(dtype=int)

    for index, column in enumerate(score_columns):
        ax.plot(
            x,
            per_post[column].to_numpy(dtype=float),
            color=colors[index % len(colors)],
            linewidth=2,
            marker="o",
            markersize=3,
            label=column.replace("_", " "),
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Post/comment number for each top user")
    ax.set_ylabel("Average Detoxify score")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True, ncol=2)
    ax.text(
        1,
        -0.13,
        f"Top users included: {top_user_count:,}",
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
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")
    if args.min_comments < 1:
        raise SystemExit("--min-comments must be at least 1")
    if args.max_post_number is not None and args.max_post_number < 1:
        raise SystemExit("--max-post-number must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.timestamp_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    score_columns = required_score_columns(columns)
    if args.rank_by == "average_toxicity" and "toxicity" not in score_columns:
        raise SystemExit("--rank-by average_toxicity requires a toxicity column")

    df = load_comments(args, score_columns)
    ranking = rank_top_users(
        df,
        args.username_column,
        score_columns,
        args.rank_by,
        args.min_comments,
        args.top_n,
    )
    per_post = average_scores_by_post_number(
        df,
        ranking,
        args.username_column,
        args.timestamp_column,
        score_columns,
    )

    min_users_per_post = args.min_users_per_post or len(ranking)
    if min_users_per_post < 1:
        raise SystemExit("--min-users-per-post must be at least 1")
    if min_users_per_post > len(ranking):
        raise SystemExit(
            "--min-users-per-post cannot be greater than selected top users"
        )

    per_post = per_post[per_post["contributing_users"] >= min_users_per_post]
    if args.max_post_number is not None:
        per_post = per_post[per_post["post_number"] <= args.max_post_number]
    if per_post.empty:
        raise SystemExit("No post/comment numbers met the plotting filters")

    output_png = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_top_{len(ranking)}_toxicity_by_post_number.png"
    )
    title = args.title or (
        f"{args.input_csv.stem}: Average Detoxify Scores by Post Number "
        f"for Top {len(ranking)} Users"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        per_post.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_per_post_scores(per_post, score_columns, output_png, title, len(ranking))

    print(f"Saved {output_png}")
    print(f"Users selected: {len(ranking)}")
    print(f"Post/comment numbers plotted: {len(per_post)}")
    print(
        ranking[
            ["rank", args.username_column, "comment_count", "average_toxicity"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
