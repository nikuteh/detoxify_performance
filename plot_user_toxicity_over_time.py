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
            "Plot one user's Detoxify scores over time, with one regression "
            "line per toxicity measurement."
        )
    )
    parser.add_argument(
        "user_or_csv",
        help="Username, or path to that user's CSV file.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Folder containing per-user CSV files when user_or_csv is a username.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: next to the user CSV.",
    )
    return parser.parse_args()


def resolve_user_csv(user_or_csv, folder):
    possible_path = Path(user_or_csv)
    if possible_path.suffix.lower() == ".csv" or possible_path.exists():
        return possible_path

    if folder is None:
        raise SystemExit(
            "When passing a username, also pass the folder containing user CSVs.\n"
            "Example: python plot_user_toxicity_over_time.py angelbirth "
            "subreddits/asahi_linux"
        )

    return folder / f"{user_or_csv}.csv"


def decimal_year(dates):
    start = pd.to_datetime(
        {"year": dates.dt.year, "month": 1, "day": 1}, utc=True
    )
    next_start = pd.to_datetime(
        {"year": dates.dt.year + 1, "month": 1, "day": 1}, utc=True
    )
    elapsed = (dates - start).dt.total_seconds()
    year_length = (next_start - start).dt.total_seconds()
    return dates.dt.year + elapsed / year_length


def main():
    args = parse_args()
    input_csv = resolve_user_csv(args.user_or_csv, args.folder)
    if not input_csv.exists():
        raise SystemExit(f"User CSV does not exist: {input_csv}")

    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    required_columns = {"username", "timestamp", *TOXICITY_COLUMNS}
    columns = set(pd.read_csv(input_csv, nrows=0).columns)
    missing = sorted(required_columns - columns)
    if missing:
        raise SystemExit(
            f"{input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = pd.read_csv(input_csv, usecols=["username", "timestamp", *TOXICITY_COLUMNS])
    df = df.dropna(subset=["timestamp", *TOXICITY_COLUMNS])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["year"] = decimal_year(df["date"])
    df = df.sort_values("year")

    if df.empty:
        raise SystemExit(f"No usable rows found in {input_csv}")

    username = str(df["username"].mode().iloc[0])
    subreddit_name = input_csv.parent.name
    output_png = args.output or input_csv.with_name(f"{input_csv.stem}_toxicity_trends.png")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    colors = plt.get_cmap("tab10").colors
    x = df["year"].to_numpy(dtype=float)
    x_line = np.linspace(x.min(), x.max(), 200)

    for index, column in enumerate(TOXICITY_COLUMNS):
        y = df[column].to_numpy(dtype=float)
        color = colors[index % len(colors)]

        ax.scatter(
            x,
            y,
            s=12,
            alpha=0.18,
            color=color,
            linewidths=0,
        )

        if len(df) >= 2 and np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            y_line = slope * x_line + intercept
            label = f"{column} regression ({slope:+.4f}/year)"
            ax.plot(x_line, y_line, color=color, linewidth=2.2, label=label)
        else:
            ax.scatter([], [], color=color, label=column)

    ax.set_title(
        f"{subreddit_name}: {username} Detoxify Scores Over Time",
        pad=12,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Toxicity score")
    ax.set_ylim(0, 1)
    ax.set_xlim(np.floor(x.min()), np.ceil(x.max()))
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(title="Regression lines", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"User: {username}")
    print(f"Subreddit folder: {subreddit_name}")
    print(f"Comments plotted: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
