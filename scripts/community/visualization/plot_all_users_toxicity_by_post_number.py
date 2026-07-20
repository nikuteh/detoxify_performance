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
            "For every user in a community CSV, plot toxicity by each user's "
            "chronological post/comment number."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Community CSV containing usernames, timestamps, and toxicity scores.",
    )
    parser.add_argument(
        "--average-output",
        type=Path,
        help=(
            "Output PNG for average toxicity. Default: inferred from the "
            "input CSV filename."
        ),
    )
    parser.add_argument(
        "--percent-output",
        type=Path,
        help=(
            "Output PNG for percent above threshold. Default: inferred from "
            "the input CSV filename."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV containing the per-post-number summary.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Score column to summarize. Default: toxicity.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for percent-toxic plot. Default: 0.5.",
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
        "--min-comments-per-post",
        type=int,
        default=1,
        help=(
            "Minimum comments required at a post/comment number for it to be "
            "plotted. Default: 1."
        ),
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        default=200,
        help="Only plot post/comment numbers up to this value. Default: 200.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames.",
    )
    parser.add_argument(
        "--average-title",
        help="Average-toxicity plot title. Default: inferred from input CSV.",
    )
    parser.add_argument(
        "--percent-title",
        help="Percent-toxic plot title. Default: inferred from input CSV.",
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


def load_comments(args):
    usecols = [args.username_column, args.timestamp_column, args.score_column]
    df = pd.read_csv(args.input_csv, usecols=usecols)
    df["source_order"] = range(len(df))

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
    df = df.dropna(
        subset=[args.username_column, args.timestamp_column, args.score_column]
    )

    if df.empty:
        raise SystemExit("No usable comment rows found after cleaning")

    return df


def summarize_by_post_number(df, args):
    sorted_df = df.sort_values(
        [args.username_column, args.timestamp_column, "source_order"]
    )
    sorted_df["post_number"] = sorted_df.groupby(args.username_column).cumcount() + 1
    sorted_df["above_threshold"] = sorted_df[args.score_column] > args.threshold

    if args.max_post_number is not None:
        sorted_df = sorted_df[sorted_df["post_number"] <= args.max_post_number]

    summary = (
        sorted_df.groupby("post_number")
        .agg(
            average_toxicity=(args.score_column, "mean"),
            comments_above_threshold=("above_threshold", "sum"),
            total_comments=(args.score_column, "size"),
            contributing_users=(args.username_column, "nunique"),
        )
        .reset_index()
    )
    summary["percent_above_threshold"] = (
        summary["comments_above_threshold"] / summary["total_comments"] * 100
    )
    summary["threshold"] = args.threshold

    summary = summary[summary["total_comments"] >= args.min_comments_per_post].copy()
    if summary.empty:
        raise SystemExit("No post/comment numbers met the plotting filters")

    return summary


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def add_regression_line(ax, summary, y_column, color):
    if len(summary) < 2:
        return

    x = summary["post_number"].to_numpy(dtype=float)
    y = summary[y_column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(
        x,
        slope * x + intercept,
        color=color,
        linewidth=2,
        linestyle="--",
        label=f"Linear regression (slope {slope:.4g})",
    )


def plot_average_toxicity(summary, output_png, title):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary["post_number"],
        summary["average_toxicity"],
        color="#4C78A8",
        linewidth=2.3,
        marker="o",
        markersize=3,
        label="Average toxicity",
    )
    add_regression_line(ax, summary, "average_toxicity", "#F58518")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Post/comment number for each user")
    ax.set_ylabel("Average toxicity score")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        f"Post numbers plotted: {len(summary):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_percent_toxic(summary, output_png, title, threshold):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary["post_number"],
        summary["percent_above_threshold"],
        color="#F58518",
        linewidth=2.3,
        marker="o",
        markersize=3,
        label=f"Percent above {threshold:g} toxicity",
    )
    add_regression_line(ax, summary, "percent_above_threshold", "#4C78A8")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Post/comment number for each user")
    ax.set_ylabel(f"Percent of comments above {threshold:g} toxicity")
    ax.set_ylim(0, 100)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        f"Post numbers plotted: {len(summary):,}",
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
    if args.min_comments_per_post < 1:
        raise SystemExit("--min-comments-per-post must be at least 1")
    if args.max_post_number is not None and args.max_post_number < 1:
        raise SystemExit("--max-post-number must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.timestamp_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    summary = summarize_by_post_number(load_comments(args), args)

    average_output = args.average_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_all_users_average_toxicity_by_post_number.png",
    )
    percent_output = args.percent_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_all_users_percent_toxic_by_post_number.png",
    )
    average_title = args.average_title or (
        f"{args.input_csv.stem}: Average Toxicity by User Post Number"
    )
    percent_title = args.percent_title or (
        f"{args.input_csv.stem}: Percent of Comments Above Toxicity Threshold"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_average_toxicity(summary, average_output, average_title)
    plot_percent_toxic(summary, percent_output, percent_title, args.threshold)

    print(f"Saved {average_output}")
    print(f"Saved {percent_output}")
    print(f"Post/comment numbers plotted: {len(summary):,}")
    print(f"Comments included in plotted points: {summary['total_comments'].sum():,}")
    print(
        "Users contributing to first post/comment number: "
        f"{summary['contributing_users'].iloc[0]:,}"
    )


if __name__ == "__main__":
    main()
