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
            "Reconstruct observed Reddit reply-chain depth from comment_id and "
            "parent_id values, then plot toxicity by local reply depth."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Community Detoxify predictions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV containing toxicity by reply depth.",
    )
    parser.add_argument(
        "--comment-id-column",
        help=(
            "Column containing comment ids. Default: auto-detect one of "
            "comment_id, id, or name."
        ),
    )
    parser.add_argument(
        "--parent-id-column",
        default="parent_id",
        help="Column containing Reddit parent ids. Default: parent_id.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Toxicity score column to summarize. Default: toxicity.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for percent-toxic line. Default: 0.5.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Only plot observed reply depths up to this value. Default: 20.",
    )
    parser.add_argument(
        "--min-comments-per-depth",
        type=int,
        default=1,
        help="Minimum comments required at a depth to plot it. Default: 1.",
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
        "Input CSV needs a comment id column to reconstruct reply chains. "
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


def load_comments(input_csv, comment_id_column, parent_id_column, score_column):
    usecols = [comment_id_column, parent_id_column, score_column]
    df = pd.read_csv(input_csv, usecols=usecols)
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df["normalized_comment_id"] = df[comment_id_column].map(normalize_comment_id)
    df["normalized_parent_id"] = df[parent_id_column].map(normalize_parent_id)
    df = df.dropna(subset=[score_column])
    df = df[df["normalized_comment_id"] != ""].copy()

    if df.empty:
        raise SystemExit("No usable comments found after cleaning")

    df = df.drop_duplicates("normalized_comment_id", keep="first")
    return df


def compute_reply_depths(comment_ids, parent_by_id):
    memo = {}

    def depth_for(comment_id):
        current = comment_id
        path = []
        seen = set()

        while True:
            if current in memo:
                base_depth = memo[current]
                break
            if current in seen:
                base_depth = 1
                break

            seen.add(current)
            path.append(current)
            parent_id = parent_by_id.get(current, "")
            if not parent_id or parent_id not in parent_by_id:
                base_depth = 1
                break
            current = parent_id

        for path_comment_id in reversed(path):
            memo[path_comment_id] = base_depth
            base_depth += 1

        return memo[comment_id]

    return {comment_id: depth_for(comment_id) for comment_id in comment_ids}


def summarize_by_depth(df, score_column, threshold, max_depth, min_comments_per_depth):
    parent_by_id = dict(
        zip(df["normalized_comment_id"], df["normalized_parent_id"], strict=False)
    )
    depths = compute_reply_depths(df["normalized_comment_id"].tolist(), parent_by_id)

    working = df.copy()
    working["reply_depth"] = working["normalized_comment_id"].map(depths).astype(int)
    working["above_threshold"] = working[score_column] > threshold

    if max_depth is not None:
        working = working[working["reply_depth"] <= max_depth].copy()

    summary = (
        working.groupby("reply_depth")
        .agg(
            average_toxicity=(score_column, "mean"),
            median_toxicity=(score_column, "median"),
            comments_above_threshold=("above_threshold", "sum"),
            total_comments=(score_column, "size"),
        )
        .reset_index()
        .sort_values("reply_depth")
    )
    summary["percent_above_threshold"] = (
        summary["comments_above_threshold"] / summary["total_comments"] * 100
    )
    summary["threshold"] = threshold
    summary = summary[summary["total_comments"] >= min_comments_per_depth].copy()

    if summary.empty:
        raise SystemExit("No reply depths met the plotting filters")

    return summary


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_thread_escalation(summary, output_png, title, threshold):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary["reply_depth"],
        summary["average_toxicity"],
        color="#4C78A8",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="Average toxicity",
    )
    ax.set_xlabel("Observed reply-chain depth")
    ax.set_ylabel("Average toxicity score")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)

    ax2 = ax.twinx()
    ax2.plot(
        summary["reply_depth"],
        summary["percent_above_threshold"],
        color="#F58518",
        linewidth=2.0,
        marker="s",
        markersize=3,
        label=f"Percent above {threshold:g}",
    )
    ax2.set_ylabel(f"Percent above {threshold:g} toxicity")
    ax2.set_ylim(0, 100)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=True)
    ax.set_title(title, pad=12)
    ax.text(
        1,
        -0.13,
        f"Depths plotted: {len(summary):,}",
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
    if args.max_depth is not None and args.max_depth < 1:
        raise SystemExit("--max-depth must be at least 1")
    if args.min_comments_per_depth < 1:
        raise SystemExit("--min-comments-per-depth must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(columns, args.comment_id_column)
    required = {args.parent_id_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = load_comments(
        args.input_csv,
        comment_id_column,
        args.parent_id_column,
        args.score_column,
    )
    summary = summarize_by_depth(
        df,
        args.score_column,
        args.threshold,
        args.max_depth,
        args.min_comments_per_depth,
    )

    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_thread_escalation.png",
    )
    title = args.title or f"{args.input_csv.stem}: Toxicity by Reply-Chain Depth"

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_thread_escalation(summary, output_png, title, args.threshold)

    print(f"Saved {output_png}")
    print(f"Comments considered: {int(summary['total_comments'].sum()):,}")
    print(f"Reply depths plotted: {len(summary):,}")


if __name__ == "__main__":
    main()
