import argparse
import os
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot each Detoxify toxicity type as a percent of one user's "
            "comments that crossed at least one toxicity threshold."
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
        help="Output PNG path. Default: inferred from the user CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV for the percentage summary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for counting a comment as toxic. Default: 0.5.",
    )
    return parser.parse_args()


def resolve_user_csv(user_or_csv, folder):
    possible_path = Path(user_or_csv)
    if possible_path.suffix.lower() == ".csv" or possible_path.exists():
        return possible_path

    if folder is None:
        raise SystemExit(
            "When passing a username, also pass the folder containing user CSVs.\n"
            "Example: python scripts/users/visualization/"
            "plot_user_toxicity_over_time.py angelbirth "
            "data/subreddits/Asahi_Linux/users"
        )

    direct_path = folder / f"{user_or_csv}.csv"
    if direct_path.exists():
        return direct_path

    users_path = folder / "users" / f"{user_or_csv}.csv"
    if users_path.exists():
        return users_path

    return direct_path


def infer_visualization_output(input_csv):
    filename = f"{input_csv.stem}_toxicity_percentages.png"
    try:
        relative_input = input_csv.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        return input_csv.with_name(filename)

    if len(relative_input.parts) < 2:
        return input_csv.with_name(filename)

    subreddit_name = relative_input.parts[0]
    parent_parts = relative_input.parts[1:-1]
    return VISUALIZATION_SUBREDDITS_DIR.joinpath(
        subreddit_name,
        *parent_parts,
        filename,
    )


def compute_user_toxicity_percentages(df, threshold):
    total_comments = len(df)
    any_toxic = pd.Series(False, index=df.index)
    column_counts = []

    for column in TOXICITY_COLUMNS:
        scores = pd.to_numeric(df[column], errors="coerce")
        above_threshold = scores > threshold
        any_toxic = any_toxic | above_threshold.fillna(False)
        column_counts.append(
            (column, int(above_threshold.sum()), int(scores.notna().sum()))
        )

    toxic_comments = int(any_toxic.sum())
    toxic_percent = toxic_comments / total_comments * 100 if total_comments else 0.0

    rows = []
    for column, count, valid_scores in column_counts:
        rows.append(
            {
                "score_column": column,
                "threshold": threshold,
                "total_user_comments": total_comments,
                "comments_with_any_toxicity_type": toxic_comments,
                "valid_scores": valid_scores,
                "comments_above_threshold": count,
                "percent_of_toxic_comments": (
                    count / toxic_comments * 100 if toxic_comments else 0.0
                ),
                "percent_of_user_comments": (
                    count / total_comments * 100 if total_comments else 0.0
                ),
                "percent_of_scored_comments": (
                    count / valid_scores * 100 if valid_scores else 0.0
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary["percent_with_any_toxicity_type"] = toxic_percent
    return summary


def plot_user_toxicity_percentages(summary, output_png, username, subreddit_name):
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_data = summary.sort_values("percent_of_toxic_comments", ascending=True)
    labels = plot_data["score_column"].str.replace("_", " ").tolist()
    values = plot_data["percent_of_toxic_comments"].to_numpy(dtype=float)
    counts = plot_data["comments_above_threshold"].to_numpy(dtype=int)
    threshold = float(plot_data["threshold"].iloc[0])
    total_comments = int(plot_data["total_user_comments"].iloc[0])
    any_toxic = int(plot_data["comments_with_any_toxicity_type"].iloc[0])
    any_toxic_percent = float(plot_data["percent_with_any_toxicity_type"].iloc[0])

    fig_height = max(5, 0.58 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(11, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(labels, values, color="#4C78A8", edgecolor="white")

    max_value = values.max() if len(values) else 0
    label_offset = max(max_value * 0.015, 0.03)
    for bar, value, count in zip(bars, values, counts):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}% ({count:,})",
            va="center",
            fontsize=9,
        )

    ax.set_title(
        f"{subreddit_name}: {username} Toxic Comments by Type",
        pad=12,
    )
    ax.set_xlabel("Percent of toxic comments above threshold")
    ax.set_ylabel("Toxicity type")
    ax.set_xlim(0, max_value + label_offset * 9 if max_value else 1)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.15,
        (
            f"Threshold: {threshold:g} | Total comments: {total_comments:,} | "
            f"Toxic comments: {any_toxic:,} ({any_toxic_percent:.2f}% of all)"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = resolve_user_csv(args.user_or_csv, args.folder)
    if not input_csv.exists():
        raise SystemExit(f"User CSV does not exist: {input_csv}")

    required_columns = {"username", *TOXICITY_COLUMNS}
    columns = set(pd.read_csv(input_csv, nrows=0).columns)
    missing = sorted(required_columns - columns)
    if missing:
        raise SystemExit(
            f"{input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = pd.read_csv(input_csv, usecols=["username", *TOXICITY_COLUMNS])
    df = df.dropna(how="all", subset=TOXICITY_COLUMNS)

    if df.empty:
        raise SystemExit(f"No usable rows found in {input_csv}")

    username = str(df["username"].mode().iloc[0])
    subreddit_name = (
        input_csv.parent.parent.name
        if input_csv.parent.name == "users"
        else input_csv.parent.name
    )
    output_png = args.output or infer_visualization_output(input_csv)
    summary = compute_user_toxicity_percentages(df, args.threshold)
    plot_user_toxicity_percentages(summary, output_png, username, subreddit_name)

    display = summary.copy()
    display["percent_of_toxic_comments"] = display[
        "percent_of_toxic_comments"
    ].round(4)
    display["percent_of_user_comments"] = display[
        "percent_of_user_comments"
    ].round(4)
    display["percent_of_scored_comments"] = display[
        "percent_of_scored_comments"
    ].round(4)
    display["percent_with_any_toxicity_type"] = display[
        "percent_with_any_toxicity_type"
    ].round(4)
    print(display.to_string(index=False))

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    print(f"Saved {output_png}")
    print(f"User: {username}")
    print(f"Subreddit folder: {subreddit_name}")
    print(f"Comments analyzed: {len(df)}")


if __name__ == "__main__":
    main()
