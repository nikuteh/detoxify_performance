import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"
DELETED_USERNAMES = {"[deleted]", "[removed]", "deleted", "removed"}
ACTIVITY_BINS = [0, 1, 4, 9, 24, 49, 99, 249, 499, 999, float("inf")]
ACTIVITY_LABELS = [
    "1",
    "2-4",
    "5-9",
    "10-24",
    "25-49",
    "50-99",
    "100-249",
    "250-499",
    "500-999",
    "1000+",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Group users by total community activity and compare toxicity "
            "across low-volume and established/high-volume users."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Community predictions CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV with toxicity by user activity bin.",
    )
    parser.add_argument(
        "--user-output",
        type=Path,
        help="Optional output CSV with one row per user.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Toxicity score column. Default: toxicity.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Toxicity threshold. Default: 0.5.",
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


def summarize_users(args):
    df = pd.read_csv(args.input_csv, usecols=[args.username_column, args.score_column])
    df[args.username_column] = (
        df[args.username_column].fillna("").astype(str).str.strip()
    )
    df[args.score_column] = pd.to_numeric(df[args.score_column], errors="coerce")
    df = df[(df[args.username_column] != "") & df[args.score_column].notna()].copy()

    if not args.include_deleted:
        df = df[~df[args.username_column].str.lower().isin(DELETED_USERNAMES)].copy()

    if df.empty:
        raise SystemExit("No usable user rows found after cleaning")

    df["above_threshold"] = df[args.score_column] > args.threshold
    users = (
        df.groupby(args.username_column)
        .agg(
            comment_count=(args.score_column, "size"),
            average_toxicity=(args.score_column, "mean"),
            toxic_comment_count=("above_threshold", "sum"),
            score_sum=(args.score_column, "sum"),
        )
        .reset_index()
        .rename(columns={args.username_column: "username"})
    )
    users["percent_comments_toxic"] = (
        users["toxic_comment_count"] / users["comment_count"] * 100
    )
    users["activity_bin"] = pd.cut(
        users["comment_count"],
        bins=ACTIVITY_BINS,
        labels=ACTIVITY_LABELS,
        include_lowest=True,
        right=True,
    )

    summary = (
        users.groupby("activity_bin", observed=False)
        .agg(
            user_count=("username", "size"),
            total_comments=("comment_count", "sum"),
            toxic_comments=("toxic_comment_count", "sum"),
            average_user_toxicity=("average_toxicity", "mean"),
            median_user_toxicity=("average_toxicity", "median"),
            score_sum=("score_sum", "sum"),
        )
        .reset_index()
    )
    summary = summary[summary["user_count"] > 0].copy()
    summary["comment_weighted_average_toxicity"] = (
        summary["score_sum"] / summary["total_comments"]
    )
    summary["percent_comments_toxic"] = (
        summary["toxic_comments"] / summary["total_comments"] * 100
    )
    summary["threshold"] = args.threshold
    return users, summary


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_authority(summary, output_png, title, threshold):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    x = range(len(summary))
    bars = ax.bar(
        x,
        summary["comment_weighted_average_toxicity"],
        color="#4C78A8",
        edgecolor="white",
        label="Comment-weighted average toxicity",
    )
    for bar, users in zip(bars, summary["user_count"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{int(users):,} users",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("User activity bin by total comments")
    ax.set_ylabel("Average toxicity score")
    ax.set_ylim(0, max(0.05, float(summary["comment_weighted_average_toxicity"].max()) * 1.35))
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["activity_bin"].astype(str), rotation=35, ha="right")
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)

    ax2 = ax.twinx()
    ax2.plot(
        list(x),
        summary["percent_comments_toxic"],
        color="#F58518",
        linewidth=2.2,
        marker="o",
        label=f"Percent comments above {threshold:g}",
    )
    ax2.set_ylabel(f"Percent comments above {threshold:g} toxicity")
    ax2.set_ylim(0, max(5, float(summary["percent_comments_toxic"].max()) * 1.3))

    lines = [bars, *ax2.get_lines()]
    labels = ["Comment-weighted average toxicity"] + [
        line.get_label() for line in ax2.get_lines()
    ]
    ax.legend(lines, labels, frameon=True)

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    users, summary = summarize_users(args)
    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_user_authority_toxicity.png",
    )
    title = args.title or f"{args.input_csv.stem}: Toxicity by User Activity"

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    if args.user_output:
        args.user_output.parent.mkdir(parents=True, exist_ok=True)
        users.to_csv(args.user_output, index=False)
        print(f"Saved {args.user_output}")

    plot_authority(summary, output_png, title, args.threshold)
    print(f"Saved {output_png}")
    print(f"Activity bins plotted: {len(summary):,}")


if __name__ == "__main__":
    main()
