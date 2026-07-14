import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the number of subreddit posts/comments over time."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Subreddit CSV containing a timestamp column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing Unix timestamps or parseable datetimes. Default: timestamp.",
    )
    parser.add_argument(
        "--time-bin",
        default="MS",
        help="Pandas resample interval for counts. Default: MS (monthly).",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--ylabel",
        default="Posts/comments",
        help="Y-axis label. Default: Posts/comments.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows to read at a time. Default: 100000.",
    )
    return parser.parse_args()


def parse_dates(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= values.notna().sum() * 0.8:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def count_posts_over_time(input_csv, timestamp_column, time_bin, chunk_size):
    counts = []
    total_rows = 0

    for chunk in pd.read_csv(
        input_csv,
        usecols=[timestamp_column],
        chunksize=chunk_size,
    ):
        total_rows += len(chunk)
        dates = parse_dates(chunk[timestamp_column]).dropna()
        if dates.empty:
            continue
        counts.append(pd.Series(1, index=dates).resample(time_bin).sum())

    if not counts:
        raise SystemExit(f"No usable timestamps found in {input_csv}")

    combined = pd.concat(counts).groupby(level=0).sum().sort_index()
    full_index = pd.date_range(
        combined.index.min(),
        combined.index.max(),
        freq=time_bin,
        tz="UTC",
    )
    return combined.reindex(full_index, fill_value=0).astype(int), total_rows


def main():
    args = parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.csv}")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")

    columns = set(pd.read_csv(args.csv, nrows=0).columns)
    if args.timestamp_column not in columns:
        raise SystemExit(
            f"{args.csv} is missing timestamp column: {args.timestamp_column}"
        )

    output_png = args.output or args.csv.with_name(f"{args.csv.stem}_posts_over_time.png")
    title = args.title or f"{args.csv.stem} Posts Over Time"

    counts, total_rows = count_posts_over_time(
        args.csv,
        args.timestamp_column,
        args.time_bin,
        args.chunk_size,
    )

    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        counts.index,
        counts.values,
        color="#2563A6",
        linewidth=2.6,
        marker="o",
        markersize=4,
        markerfacecolor="#F28E2B",
        markeredgecolor="white",
        markeredgewidth=0.7,
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(args.ylabel)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#E2E2E2", linewidth=0.8)

    ax.text(
        1,
        -0.14,
        f"Rows read: {total_rows:,} | Time bin: {args.time_bin}",
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

    print(f"Saved {output_png}")
    print(f"Rows read: {total_rows}")
    print(f"Bins plotted: {len(counts)}")
    print(f"Date range: {counts.index.min()} to {counts.index.max()}")


if __name__ == "__main__":
    main()
