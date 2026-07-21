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
            "For comments up to a parent post_number cap, find direct replies "
            "in the same CSV and plot the average toxicity of those replies."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Community Detoxify predictions CSV containing comment ids, "
            "parent ids, post_number, and toxicity scores."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV containing average response toxicity by post_number.",
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
        "--post-number-column",
        default="post_number",
        help="Column containing each comment's post number. Default: post_number.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Response toxicity score column to average. Default: toxicity.",
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        default=500,
        help="Only include parent post numbers up to this value. Default: 500.",
    )
    parser.add_argument(
        "--min-responses-per-post",
        type=int,
        default=1,
        help=(
            "Minimum direct responses required for a post number to be plotted. "
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
        "Input CSV needs a comment id column to match replies. Expected one "
        "of: comment_id, id, name."
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


def load_comments(args, comment_id_column):
    usecols = [
        comment_id_column,
        args.parent_id_column,
        args.post_number_column,
        args.score_column,
    ]
    df = pd.read_csv(args.input_csv, usecols=usecols)

    df[args.post_number_column] = pd.to_numeric(
        df[args.post_number_column],
        errors="coerce",
    )
    df[args.score_column] = pd.to_numeric(df[args.score_column], errors="coerce")
    df["normalized_comment_id"] = df[comment_id_column].map(normalize_comment_id)
    df["normalized_parent_id"] = df[args.parent_id_column].map(normalize_parent_id)

    df = df[df["normalized_comment_id"] != ""].copy()
    if df.empty:
        raise SystemExit("No usable comment ids found in input CSV")

    return df


def summarize_response_toxicity(df, args):
    parents = df[
        ["normalized_comment_id", args.post_number_column]
    ].dropna(subset=[args.post_number_column]).copy()
    parents[args.post_number_column] = parents[args.post_number_column].astype(int)
    parents = parents[parents[args.post_number_column] <= args.max_post_number]
    parents = parents.drop_duplicates("normalized_comment_id", keep="first")

    if parents.empty:
        raise SystemExit(
            f"No parent comments found with {args.post_number_column} <= "
            f"{args.max_post_number}"
        )

    responses = df[
        ["normalized_parent_id", args.score_column]
    ].dropna(subset=[args.score_column]).copy()
    responses = responses[responses["normalized_parent_id"] != ""]

    matched = responses.merge(
        parents,
        how="inner",
        left_on="normalized_parent_id",
        right_on="normalized_comment_id",
    )
    if matched.empty:
        raise SystemExit("No direct responses matched parent comments in the CSV")

    summary = (
        matched.groupby(args.post_number_column)
        .agg(
            average_response_toxicity=(args.score_column, "mean"),
            response_count=(args.score_column, "size"),
            parent_comments_with_responses=("normalized_comment_id", "nunique"),
        )
        .reset_index()
        .sort_values(args.post_number_column)
    )
    summary = summary[
        summary["response_count"] >= args.min_responses_per_post
    ].copy()
    if summary.empty:
        raise SystemExit("No post numbers met the --min-responses-per-post filter")

    return summary, {
        "parents_considered": len(parents),
        "responses_matched": len(matched),
        "post_numbers_plotted": len(summary),
    }


def plot_average_response_toxicity(
    summary,
    post_number_column,
    output_png,
    title,
    counts,
):
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        summary[post_number_column],
        summary["average_response_toxicity"],
        color="#4C78A8",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Parent comment post number")
    ax.set_ylabel("Average toxicity of direct responses")
    ax.set_xlim(1, max(summary[post_number_column].max(), 1))
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.13,
        (
            f"Parent comments considered: {counts['parents_considered']:,}; "
            f"matched responses: {counts['responses_matched']:,}"
        ),
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
    if args.max_post_number < 1:
        raise SystemExit("--max-post-number must be at least 1")
    if args.min_responses_per_post < 1:
        raise SystemExit("--min-responses-per-post must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(columns, args.comment_id_column)
    required = {
        comment_id_column,
        args.parent_id_column,
        args.post_number_column,
        args.score_column,
    }
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = load_comments(args, comment_id_column)
    summary, counts = summarize_response_toxicity(df, args)

    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_average_response_toxicity_by_parent_post_number.png",
    )
    title = args.title or (
        f"{args.input_csv.stem}: Average Response Toxicity by Parent Post Number"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_average_response_toxicity(
        summary,
        args.post_number_column,
        output_png,
        title,
        counts,
    )

    print(f"Saved {output_png}")
    print(f"Parent comments considered: {counts['parents_considered']:,}")
    print(f"Direct responses matched: {counts['responses_matched']:,}")
    print(f"Parent post numbers plotted: {counts['post_numbers_plotted']:,}")


if __name__ == "__main__":
    main()
