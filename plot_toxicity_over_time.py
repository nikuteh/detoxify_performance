import argparse
import os
from pathlib import Path


DEFAULT_INPUT = Path(
    "asahi_linux_reddit/subreddits25/"
    "AsahiLinux_comments_detoxify_unbiased_predictions.csv"
)
DEFAULT_OUTPUT = Path(
    "asahi_linux_reddit/subreddits25/AsahiLinux_toxicity_over_time.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Detoxify toxicity scores over calendar years."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Predictions CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        type=Path,
        help="Predictions CSV path. Kept for compatibility; positional CSV is preferred.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--time-bin",
        default="MS",
        help="Pandas resample interval for the average line. Default: MS (monthly).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = args.input_option or args.csv
    output_png = args.output
    if output_png is None:
        if input_csv == DEFAULT_INPUT:
            output_png = DEFAULT_OUTPUT
        else:
            output_png = input_csv.with_name(f"{input_csv.stem}_toxicity_over_time.png")
    title = args.title or f"{input_csv.stem} Toxicity Over Time"

    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    required_columns = {"timestamp", "toxicity"}
    columns = set(pd.read_csv(input_csv, nrows=0).columns)
    missing = sorted(required_columns - columns)
    if missing:
        raise SystemExit(
            f"{input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = pd.read_csv(input_csv, usecols=["timestamp", "toxicity"])
    df = df.dropna(subset=["timestamp", "toxicity"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("date")

    average = df.set_index("date")["toxicity"].resample(args.time_bin).mean().dropna()

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.scatter(
        df["date"],
        df["toxicity"],
        s=7,
        alpha=0.12,
        color="#4C78A8",
        linewidths=0,
        label="Individual comments",
    )
    ax.plot(
        average.index,
        average.values,
        color="#D95F02",
        linewidth=2.4,
        label="Average toxicity",
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Toxicity score")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Comments plotted: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
