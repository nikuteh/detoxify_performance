import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"
SECONDS_PER_DAY = 24 * 60 * 60
TIME_UNITS = {
    "day": 1,
    "week": 7,
    "month": 30,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "For every user in a community CSV, plot toxicity over elapsed "
            "time since each user's first comment in the subreddit."
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
        help="Optional output CSV containing the per-time-bin summary.",
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
        "--time-unit",
        choices=sorted(TIME_UNITS),
        default="week",
        help="Elapsed-time unit for the x-axis. Default: week.",
    )
    parser.add_argument(
        "--max-time-number",
        type=int,
        default=100,
        help="Only plot elapsed time numbers up to this value. Default: 100.",
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
        "--min-comments-per-time",
        type=int,
        default=1,
        help=(
            "Minimum comments required at an elapsed-time number for it to be "
            "plotted. Default: 1."
        ),
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


def summarize_by_user_time(df, args):
    sorted_df = df.sort_values(
        [args.username_column, args.timestamp_column, "source_order"]
    ).copy()
    first_timestamp = sorted_df.groupby(args.username_column)[
        args.timestamp_column
    ].transform("min")
    elapsed_days = (
        (sorted_df[args.timestamp_column] - first_timestamp) / SECONDS_PER_DAY
    )
    unit_days = TIME_UNITS[args.time_unit]
    sorted_df["time_number"] = np.floor(elapsed_days / unit_days).astype(int) + 1
    sorted_df["above_threshold"] = sorted_df[args.score_column] > args.threshold

    if args.max_time_number is not None:
        sorted_df = sorted_df[sorted_df["time_number"] <= args.max_time_number]

    summary = (
        sorted_df.groupby("time_number")
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
    summary["time_unit"] = args.time_unit

    summary = summary[summary["total_comments"] >= args.min_comments_per_time].copy()
    if summary.empty:
        raise SystemExit("No elapsed-time numbers met the plotting filters")

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

    x = summary["time_number"].to_numpy(dtype=float)
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


def x_axis_label(time_unit):
    return f"{time_unit.capitalize()}s since each user's first subreddit comment"


def plot_average_toxicity(summary, output_png, title, time_unit):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary["time_number"],
        summary["average_toxicity"],
        color="#4C78A8",
        linewidth=2.3,
        marker="o",
        markersize=3,
        label="Average toxicity",
    )
    add_regression_line(ax, summary, "average_toxicity", "#F58518")
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_axis_label(time_unit))
    ax.set_ylabel("Average toxicity score")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        f"Elapsed-time numbers plotted: {len(summary):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_percent_toxic(summary, output_png, title, threshold, time_unit):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary["time_number"],
        summary["percent_above_threshold"],
        color="#F58518",
        linewidth=2.3,
        marker="o",
        markersize=3,
        label=f"Percent above {threshold:g} toxicity",
    )
    add_regression_line(ax, summary, "percent_above_threshold", "#4C78A8")
    ax.set_title(title, pad=12)
    ax.set_xlabel(x_axis_label(time_unit))
    ax.set_ylabel(f"Percent of comments above {threshold:g} toxicity")
    ax.set_ylim(0, 100)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        f"Elapsed-time numbers plotted: {len(summary):,}",
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
    if args.max_time_number is not None and args.max_time_number < 1:
        raise SystemExit("--max-time-number must be at least 1")
    if args.min_comments_per_time < 1:
        raise SystemExit("--min-comments-per-time must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.timestamp_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    summary = summarize_by_user_time(load_comments(args), args)

    average_output = args.average_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_all_users_average_toxicity_over_user_time.png",
    )
    percent_output = args.percent_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_all_users_percent_toxic_over_user_time.png",
    )
    average_title = args.average_title or (
        f"{args.input_csv.stem}: Average Toxicity Over User Time"
    )
    percent_title = args.percent_title or (
        f"{args.input_csv.stem}: Percent of Comments Above Toxicity Threshold"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_average_toxicity(summary, average_output, average_title, args.time_unit)
    plot_percent_toxic(
        summary,
        percent_output,
        percent_title,
        args.threshold,
        args.time_unit,
    )

    print(f"Saved {average_output}")
    print(f"Saved {percent_output}")
    print(f"Elapsed-time numbers plotted: {len(summary):,}")
    print(f"Comments included in plotted points: {summary['total_comments'].sum():,}")
    print(
        "Users contributing to first elapsed-time number: "
        f"{summary['contributing_users'].iloc[0]:,}"
    )


if __name__ == "__main__":
    main()
