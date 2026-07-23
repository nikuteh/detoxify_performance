import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot toxic-comment volume over time as monthly bars and a line."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Community predictions CSV containing timestamps and toxicity scores.",
    )
    parser.add_argument(
        "--bar-output",
        type=Path,
        help="Output PNG path for the bar chart.",
    )
    parser.add_argument(
        "--line-output",
        type=Path,
        help="Output PNG path for the line chart.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV with toxic-comment counts by time bin.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing Unix timestamps or parseable datetimes. Default: timestamp.",
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
        "--time-bin",
        default="MS",
        help="Pandas resample interval for counts. Default: MS (monthly).",
    )
    parser.add_argument(
        "--bar-title",
        help="Bar chart title. Default: inferred from input CSV.",
    )
    parser.add_argument(
        "--line-title",
        help="Line chart title. Default: inferred from input CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows to read at a time. Default: 100000.",
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


def parse_dates(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= values.notna().sum() * 0.8:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def summarize_toxic_volume(args):
    toxic_counts = []
    total_counts = []
    total_rows = 0

    for chunk in pd.read_csv(
        args.input_csv,
        usecols=[args.timestamp_column, args.score_column],
        chunksize=args.chunk_size,
    ):
        total_rows += len(chunk)
        dates = parse_dates(chunk[args.timestamp_column])
        scores = pd.to_numeric(chunk[args.score_column], errors="coerce")
        working = pd.DataFrame({"date": dates, "score": scores}).dropna()
        if working.empty:
            continue

        indexed = working.set_index("date").sort_index()
        total_counts.append(indexed["score"].resample(args.time_bin).size())
        toxic_counts.append(
            (indexed["score"] > args.threshold).resample(args.time_bin).sum()
        )

    if not toxic_counts:
        raise SystemExit(f"No usable timestamp and toxicity rows found in {args.input_csv}")

    toxic = pd.concat(toxic_counts).groupby(level=0).sum().sort_index()
    total = pd.concat(total_counts).groupby(level=0).sum().sort_index()
    full_index = pd.date_range(
        min(toxic.index.min(), total.index.min()),
        max(toxic.index.max(), total.index.max()),
        freq=args.time_bin,
        tz="UTC",
    )
    summary = pd.DataFrame(
        {
            "date": full_index,
            "toxic_comment_count": toxic.reindex(full_index, fill_value=0).astype(int),
            "total_comments": total.reindex(full_index, fill_value=0).astype(int),
        }
    )
    summary["percent_toxic_comments"] = 0.0
    has_comments = summary["total_comments"] > 0
    summary.loc[has_comments, "percent_toxic_comments"] = (
        summary.loc[has_comments, "toxic_comment_count"]
        / summary.loc[has_comments, "total_comments"]
        * 100
    )
    summary["threshold"] = args.threshold
    summary["time_bin"] = args.time_bin
    return summary, total_rows


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    return mdates, plt


def plot_toxic_volume(summary, output_png, title, chart_style, threshold):
    mdates, plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    if chart_style == "bar":
        ax.bar(
            summary["date"],
            summary["toxic_comment_count"],
            width=25,
            color="#E45756",
            edgecolor="white",
            linewidth=0.5,
            align="center",
        )
    else:
        ax.plot(
            summary["date"],
            summary["toxic_comment_count"],
            color="#E45756",
            linewidth=2.6,
            marker="o",
            markersize=4,
            markerfacecolor="#F58518",
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Comments above {threshold:g} toxicity")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.14,
        (
            f"Months plotted: {len(summary):,}; "
            f"toxic comments: {int(summary['toxic_comment_count'].sum()):,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.timestamp_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    summary, total_rows = summarize_toxic_volume(args)
    bar_output = args.bar_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_toxic_comment_volume_by_month_bar.png",
    )
    line_output = args.line_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_toxic_comment_volume_by_month_line.png",
    )
    bar_title = args.bar_title or (
        f"{args.input_csv.stem}: Toxic Comment Volume by Month"
    )
    line_title = args.line_title or (
        f"{args.input_csv.stem}: Toxic Comment Volume by Month"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_toxic_volume(summary, bar_output, bar_title, "bar", args.threshold)
    plot_toxic_volume(summary, line_output, line_title, "line", args.threshold)

    print(f"Saved {bar_output}")
    print(f"Saved {line_output}")
    print(f"Rows read: {total_rows:,}")
    print(f"Months plotted: {len(summary):,}")
    print(f"Toxic comments counted: {int(summary['toxic_comment_count'].sum()):,}")


if __name__ == "__main__":
    main()
