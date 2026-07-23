import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"
COMMENT_ID_COLUMNS = ("comment_id", "id", "name")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot how often parent comments receive direct replies across "
            "the 0-to-1 toxicity score range."
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
        help="Optional output CSV summarizing engagement by parent toxicity bin.",
    )
    parser.add_argument(
        "--comment-id-column",
        help="Column containing comment ids. Default: auto-detect comment_id, id, or name.",
    )
    parser.add_argument(
        "--parent-id-column",
        default="parent_id",
        help="Column containing Reddit parent ids. Default: parent_id.",
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
        help="Toxicity threshold retained in the summary metadata. Default: 0.5.",
    )
    parser.add_argument(
        "--toxicity-bins",
        type=int,
        default=20,
        help="Number of fixed-width toxicity bins between 0 and 1. Default: 20.",
    )
    parser.add_argument(
        "--min-comments-per-bin",
        type=int,
        default=1,
        help=(
            "Minimum parent comments required in a toxicity bin to plot it. "
            "Default: 1."
        ),
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


def resolve_comment_id_column(columns, requested_column):
    if requested_column:
        if requested_column not in columns:
            raise SystemExit(f"Input CSV is missing comment id column: {requested_column}")
        return requested_column

    for column in COMMENT_ID_COLUMNS:
        if column in columns:
            return column

    raise SystemExit(
        "Input CSV needs a comment id column to compare engagement. "
        "Expected one of: comment_id, id, name."
    )


def normalize_comment_id(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.startswith("t3_"):
        return ""
    if text.startswith("t1_"):
        return text
    return f"t1_{text}"


def normalize_parent_id(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text.startswith("t1_"):
        return text
    return ""


def summarize_engagement(
    input_csv,
    comment_id_column,
    parent_id_column,
    score_column,
    threshold,
    toxicity_bins,
    min_comments_per_bin,
):
    df = pd.read_csv(
        input_csv,
        usecols=[comment_id_column, parent_id_column, score_column],
    )
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df["normalized_comment_id"] = df[comment_id_column].map(normalize_comment_id)
    df["normalized_parent_id"] = df[parent_id_column].map(normalize_parent_id)
    df = df.dropna(subset=[score_column])
    df = df[df["normalized_comment_id"] != ""].copy()

    if df.empty:
        raise SystemExit("No usable comments found after cleaning")

    parents = df[["normalized_comment_id", score_column]].drop_duplicates(
        "normalized_comment_id",
        keep="first",
    )
    parents = parents.rename(
        columns={
            "normalized_comment_id": "normalized_parent_id",
            score_column: "parent_toxicity",
        }
    )

    replies = df[df["normalized_parent_id"] != ""].copy()
    reply_counts = (
        replies.groupby("normalized_parent_id")
        .agg(
            direct_reply_count=(score_column, "size"),
            average_direct_reply_toxicity=(score_column, "mean"),
        )
        .reset_index()
    )

    engagement = parents.merge(reply_counts, on="normalized_parent_id", how="left")
    engagement["direct_reply_count"] = (
        engagement["direct_reply_count"].fillna(0).astype(int)
    )
    engagement["has_direct_reply"] = engagement["direct_reply_count"] > 0
    clipped_toxicity = engagement["parent_toxicity"].clip(lower=0, upper=1)
    engagement["toxicity_bin"] = (clipped_toxicity * toxicity_bins).astype(int)
    engagement.loc[engagement["toxicity_bin"] == toxicity_bins, "toxicity_bin"] = (
        toxicity_bins - 1
    )

    summary = (
        engagement.groupby("toxicity_bin")
        .agg(
            parent_comments=("parent_toxicity", "size"),
            average_parent_toxicity=("parent_toxicity", "mean"),
            average_direct_replies=("direct_reply_count", "mean"),
            median_direct_replies=("direct_reply_count", "median"),
            parent_comments_with_replies=("has_direct_reply", "sum"),
        )
        .reset_index()
        .sort_values("toxicity_bin")
    )
    bin_width = 1 / toxicity_bins
    summary["toxicity_bin_min"] = summary["toxicity_bin"] * bin_width
    summary["toxicity_bin_max"] = summary["toxicity_bin_min"] + bin_width
    summary["toxicity_midpoint"] = (
        summary["toxicity_bin_min"] + summary["toxicity_bin_max"]
    ) / 2
    summary["percent_with_any_direct_reply"] = (
        summary["parent_comments_with_replies"] / summary["parent_comments"] * 100
    )
    summary["threshold"] = threshold
    summary = summary[summary["parent_comments"] >= min_comments_per_bin].copy()
    if summary.empty:
        raise SystemExit("No toxicity bins met the plotting filters")

    return summary


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_engagement(summary, output_png, title):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary["toxicity_midpoint"],
        summary["percent_with_any_direct_reply"],
        color="#4C78A8",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Received at least one direct reply",
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Parent comment toxicity score")
    ax.set_ylabel("Percent receiving a direct reply")
    ax.set_xlim(0, 1)
    ax.set_ylim(
        0,
        min(
            100,
            max(5, float(summary["percent_with_any_direct_reply"].max()) * 1.2),
        ),
    )
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        f"Toxicity bins plotted: {len(summary):,}",
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
    if args.toxicity_bins < 1:
        raise SystemExit("--toxicity-bins must be at least 1")
    if args.min_comments_per_bin < 1:
        raise SystemExit("--min-comments-per-bin must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(columns, args.comment_id_column)
    required = {args.parent_id_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    summary = summarize_engagement(
        args.input_csv,
        comment_id_column,
        args.parent_id_column,
        args.score_column,
        args.threshold,
        args.toxicity_bins,
        args.min_comments_per_bin,
    )
    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_toxicity_engagement.png",
    )
    title = args.title or f"{args.input_csv.stem}: Toxicity and Reply Engagement"

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_engagement(summary, output_png, title)
    print(f"Saved {output_png}")
    print(f"Toxicity bins plotted: {len(summary):,}")


if __name__ == "__main__":
    main()
