import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the top users in a subreddit folder by average toxicity per "
            "post/comment number, with a scatter plot and linear regression line."
        )
    )
    parser.add_argument(
        "user_csv_folder",
        type=Path,
        help="Folder containing one CSV file per user for a subreddit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path. Default: inferred from the input folder."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of most toxic users to plot. Default: 10.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Score column to average. Default: toxicity.",
    )
    parser.add_argument(
        "--min-users-per-post",
        type=int,
        default=None,
        help=(
            "Minimum number of top users that must have a post/comment at a "
            "given post number for that point to be plotted. Default: all "
            "plotted top users."
        ),
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        help="Only plot post/comment numbers up to this value.",
    )
    return parser.parse_args()


def infer_visualization_output(user_csv_folder, filename):
    try:
        relative_folder = user_csv_folder.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        return user_csv_folder / filename

    if not relative_folder.parts:
        return user_csv_folder / filename

    subreddit_name = relative_folder.parts[0]
    return VISUALIZATION_SUBREDDITS_DIR / subreddit_name / filename


def infer_subreddit_name(user_csv_folder):
    try:
        relative_folder = user_csv_folder.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        relative_folder = None
    else:
        if relative_folder.parts:
            return relative_folder.parts[0]

    if user_csv_folder.name == "users" and user_csv_folder.parent.name:
        return user_csv_folder.parent.name

    return user_csv_folder.name


def top_users_by_average_toxicity(user_csv_folder, top_n, score_column):
    rows = []

    for csv_path in sorted(user_csv_folder.glob("*.csv")):
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except pd.errors.EmptyDataError:
            continue

        if score_column not in columns:
            continue

        usecols = [score_column]
        if "username" in columns:
            usecols.append("username")

        df = pd.read_csv(csv_path, usecols=usecols)
        scores = pd.to_numeric(df[score_column], errors="coerce").dropna()
        if scores.empty:
            continue

        if "username" in df.columns and df["username"].notna().any():
            username = str(df["username"].dropna().mode().iloc[0])
        else:
            username = csv_path.stem

        rows.append(
            {
                "username": username,
                "average_toxicity": scores.mean(),
                "comment_count": len(scores),
                "source_file": str(csv_path),
            }
        )

    if not rows:
        raise SystemExit(
            f"No CSVs with a usable {score_column!r} column found in "
            f"{user_csv_folder}"
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(["average_toxicity", "comment_count"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def load_top_user_posts(ranking, score_column):
    frames = []

    for _, user in ranking.iterrows():
        csv_path = Path(user["source_file"])

        columns = set(pd.read_csv(csv_path, nrows=0).columns)
        usecols = [score_column]
        if "timestamp" in columns:
            usecols.append("timestamp")

        df = pd.read_csv(csv_path, usecols=usecols)
        df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
        df = df.dropna(subset=[score_column]).copy()
        if df.empty:
            continue

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp", na_position="last")

        df["username"] = user["username"]
        df["rank"] = user["rank"]
        df["post_number"] = np.arange(1, len(df) + 1)
        frames.append(df[["username", "rank", "post_number", score_column]])

    if not frames:
        raise SystemExit("No usable top-user rows found to plot")

    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_args()

    if not args.user_csv_folder.is_dir():
        raise SystemExit(f"Folder does not exist: {args.user_csv_folder}")
    if args.top_n < 2:
        raise SystemExit("--top-n must be at least 2 to draw a regression line")
    if args.max_post_number is not None and args.max_post_number < 1:
        raise SystemExit("--max-post-number must be at least 1")

    output_png = args.output or infer_visualization_output(
        args.user_csv_folder,
        f"top_{args.top_n}_average_toxicity_per_post_scatter.png",
    )

    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    ranking = top_users_by_average_toxicity(
        args.user_csv_folder,
        args.top_n,
        args.score_column,
    )
    min_users_per_post = args.min_users_per_post or len(ranking)
    if min_users_per_post < 1:
        raise SystemExit("--min-users-per-post must be at least 1")
    if min_users_per_post > len(ranking):
        raise SystemExit(
            "--min-users-per-post cannot be greater than the number of "
            "ranked users"
        )

    top_posts = load_top_user_posts(ranking, args.score_column)

    if args.max_post_number is not None:
        top_posts = top_posts[top_posts["post_number"] <= args.max_post_number]

    per_post = (
        top_posts.groupby("post_number")
        .agg(
            average_toxicity=(args.score_column, "mean"),
            contributing_posts=(args.score_column, "size"),
            contributing_users=("username", "nunique"),
        )
        .reset_index()
    )
    per_post = per_post[per_post["contributing_users"] >= min_users_per_post]

    if per_post.empty:
        raise SystemExit(
            "No post/comment numbers met the --min-users-per-post filter"
        )

    x = per_post["post_number"].to_numpy(dtype=float)
    y = per_post["average_toxicity"].to_numpy(dtype=float)
    if len(per_post) < 2 or np.unique(x).size < 2:
        raise SystemExit("At least two unique post/comment numbers are needed")

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    user_counts = per_post["contributing_users"].to_numpy(dtype=float)
    if user_counts.min() == user_counts.max():
        sizes = np.full_like(user_counts, 90.0)
    else:
        sizes = np.interp(user_counts, (user_counts.min(), user_counts.max()), (45, 180))

    ax.scatter(
        x,
        y,
        s=sizes,
        alpha=0.82,
        color="#4C78A8",
        edgecolors="white",
        linewidths=1.2,
        label="Average per post/comment number",
    )
    ax.plot(
        x_line,
        y_line,
        color="#D95F02",
        linewidth=2.4,
        label=f"Regression ({slope:+.5f} toxicity per post/comment)",
    )

    subreddit_name = infer_subreddit_name(args.user_csv_folder)
    ax.set_title(
        (
            f"{subreddit_name}: Average Toxicity Per Post Number "
            f"for Top {len(ranking)} Users"
        ),
        pad=12,
    )
    ax.set_xlabel("Post/comment number for each top user")
    ax.set_ylabel(f"Average {args.score_column} per post/comment")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Post/comment numbers plotted: {len(per_post)}")
    print(
        ranking[
            ["rank", "username", "average_toxicity", "comment_count"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
