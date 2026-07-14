import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_TOXICITY_COLUMNS = [
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
            "Compute what percent of comments are above a toxicity threshold "
            "for each Detoxify score column."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV containing Detoxify prediction columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV for the summary table.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional output PNG for a bar-chart visualization.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for counting a comment as toxic. Default: 0.5.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "Specific score columns to summarize. Default: all known Detoxify "
            "toxicity columns found in the input CSV."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows to read at a time. Default: 100000.",
    )
    return parser.parse_args()


def resolve_score_columns(input_csv, requested_columns):
    columns = set(pd.read_csv(input_csv, nrows=0).columns)
    score_columns = requested_columns or [
        column for column in DEFAULT_TOXICITY_COLUMNS if column in columns
    ]
    missing = [column for column in score_columns if column not in columns]
    if missing:
        raise SystemExit(
            f"{input_csv} is missing score column(s): {', '.join(missing)}"
        )
    if not score_columns:
        raise SystemExit(
            f"No Detoxify score columns found in {input_csv}. Expected one "
            f"or more of: {', '.join(DEFAULT_TOXICITY_COLUMNS)}"
        )
    return score_columns


def compute_percentages(input_csv, score_columns, threshold, chunk_size):
    total_comments = 0
    valid_scores = {column: 0 for column in score_columns}
    above_threshold = {column: 0 for column in score_columns}

    for chunk in pd.read_csv(input_csv, usecols=score_columns, chunksize=chunk_size):
        total_comments += len(chunk)

        for column in score_columns:
            scores = pd.to_numeric(chunk[column], errors="coerce")
            valid_scores[column] += scores.notna().sum()
            above_threshold[column] += (scores > threshold).sum()

    rows = []
    for column in score_columns:
        percent_of_all = (
            above_threshold[column] / total_comments * 100
            if total_comments
            else 0.0
        )
        percent_of_scored = (
            above_threshold[column] / valid_scores[column] * 100
            if valid_scores[column]
            else 0.0
        )
        rows.append(
            {
                "score_column": column,
                "threshold": threshold,
                "total_comments": total_comments,
                "valid_scores": valid_scores[column],
                "comments_above_threshold": above_threshold[column],
                "percent_of_all_comments": percent_of_all,
                "percent_of_scored_comments": percent_of_scored,
            }
        )

    return pd.DataFrame(rows)


def plot_percentages(summary, output_png, title=None):
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_data = summary.sort_values(
        "percent_of_all_comments",
        ascending=True,
    )
    labels = plot_data["score_column"].str.replace("_", " ").tolist()
    values = plot_data["percent_of_all_comments"].to_numpy(dtype=float)
    counts = plot_data["comments_above_threshold"].to_numpy(dtype=int)
    threshold = float(plot_data["threshold"].iloc[0])
    total_comments = int(plot_data["total_comments"].iloc[0])

    fig_height = max(5, 0.55 * len(plot_data) + 2)
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

    chart_title = title or (
        f"Comments Above {threshold:g} by Detoxify Toxicity Type"
    )
    ax.set_title(chart_title, pad=12)
    ax.set_xlabel("Percent of all comments")
    ax.set_ylabel("Toxicity type")
    ax.set_xlim(0, max_value + label_offset * 9 if max_value else 1)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.14,
        f"Total comments: {total_comments:,}",
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

    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")

    score_columns = resolve_score_columns(args.input_csv, args.columns)
    summary = compute_percentages(
        args.input_csv,
        score_columns,
        args.threshold,
        args.chunk_size,
    )

    display = summary.copy()
    display["percent_of_all_comments"] = display["percent_of_all_comments"].round(4)
    display["percent_of_scored_comments"] = display[
        "percent_of_scored_comments"
    ].round(4)

    print(display.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Saved {args.output}")

    if args.plot_output:
        plot_percentages(summary, args.plot_output)
        print(f"Saved {args.plot_output}")


if __name__ == "__main__":
    main()
