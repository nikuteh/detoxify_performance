import argparse
import os
from pathlib import Path


DEFAULT_INPUT = Path(
    "data/subreddits/Asahi_Linux/"
    "AsahiLinux_comments_detoxify_unbiased_predictions.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"


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
    parser.add_argument(
        "--scatter-only",
        action="store_true",
        help="Plot only individual comment dots, with no average or regression line.",
    )
    parser.add_argument(
        "--dot-size",
        type=float,
        default=18,
        help="Scatter dot size. Default: 18.",
    )
    parser.add_argument(
        "--dot-alpha",
        type=float,
        default=0.65,
        help="Scatter dot opacity from 0 to 1. Default: 0.65.",
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


def main():
    args = parse_args()
    input_csv = args.input_option or args.csv
    output_png = args.output
    if output_png is None:
        output_png = infer_visualization_output(
            input_csv,
            f"{input_csv.stem}_toxicity_over_time.png",
        )
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

    average = None
    if not args.scatter_only:
        average = (
            df.set_index("date")["toxicity"].resample(args.time_bin).mean().dropna()
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    scatter = ax.scatter(
        df["date"],
        df["toxicity"],
        s=args.dot_size,
        alpha=args.dot_alpha,
        c=df["toxicity"],
        cmap="plasma",
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidths=0.25,
        label="Individual comments",
        rasterized=True,
    )
    if average is not None:
        ax.plot(
            average.index,
            average.values,
            color="#159A9C",
            linewidth=2.4,
            label="Average toxicity",
        )
        ax.legend(frameon=True)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("Toxicity score")

    ax.set_title(title, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Toxicity score")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#E2E2E2", linewidth=0.8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Comments plotted: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
