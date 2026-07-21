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
            "Count toxic comments by the post_number of the parent comment "
            "they were directed at."
        )
    )
    parser.add_argument(
        "toxic_csv",
        type=Path,
        help="CSV containing toxic comments and parent_id values.",
    )
    parser.add_argument(
        "all_comments_csv",
        type=Path,
        help=(
            "CSV containing all comments, a comment id column, and each "
            "comment's post_number."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: inferred from the toxic CSV filename.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV containing counts by parent post_number.",
    )
    parser.add_argument(
        "--parent-id-column",
        default="parent_id",
        help="Column containing parent ids in the toxic CSV. Default: parent_id.",
    )
    parser.add_argument(
        "--comment-id-column",
        help=(
            "Column containing comment ids in the all-comments CSV. Default: "
            "auto-detect one of comment_id, id, or name."
        ),
    )
    parser.add_argument(
        "--post-number-column",
        default="post_number",
        help="Column containing parent post numbers. Default: post_number.",
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        help="Only include parent post numbers up to this value.",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Default: inferred from the toxic CSV filename.",
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
            raise SystemExit(
                f"All-comments CSV is missing comment id column: {requested_column}"
            )
        return requested_column

    for column in COMMENT_ID_COLUMNS:
        if column in columns:
            return column

    raise SystemExit(
        "All-comments CSV needs a comment id column to match toxic parent ids. "
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


def load_toxic_parent_ids(toxic_csv, parent_id_column):
    toxic = pd.read_csv(toxic_csv, usecols=[parent_id_column])
    toxic["normalized_parent_id"] = toxic[parent_id_column].map(normalize_parent_id)
    toxic = toxic[toxic["normalized_parent_id"] != ""].copy()

    if toxic.empty:
        raise SystemExit("No toxic comments with t1_ parent ids found")

    return toxic[["normalized_parent_id"]]


def load_parent_lookup(all_comments_csv, comment_id_column, post_number_column):
    parents = pd.read_csv(
        all_comments_csv,
        usecols=[comment_id_column, post_number_column],
    )
    parents["normalized_comment_id"] = parents[comment_id_column].map(
        normalize_comment_id
    )
    parents[post_number_column] = pd.to_numeric(
        parents[post_number_column],
        errors="coerce",
    )
    parents = parents.dropna(subset=[post_number_column])
    parents = parents[parents["normalized_comment_id"] != ""].copy()
    parents[post_number_column] = parents[post_number_column].astype(int)

    if parents.empty:
        raise SystemExit("No comments with usable comment ids and post_number found")

    parents = parents.drop_duplicates("normalized_comment_id", keep="first")
    return parents[["normalized_comment_id", post_number_column]]


def summarize_toxic_comments_by_parent_post_number(
    toxic,
    parent_lookup,
    post_number_column,
    max_post_number,
):
    matched = toxic.merge(
        parent_lookup,
        how="left",
        left_on="normalized_parent_id",
        right_on="normalized_comment_id",
    )
    matched_parents = matched.dropna(subset=[post_number_column]).copy()
    matched_parents[post_number_column] = matched_parents[
        post_number_column
    ].astype(int)

    if max_post_number is not None:
        matched_parents = matched_parents[
            matched_parents[post_number_column] <= max_post_number
        ].copy()

    if matched_parents.empty:
        raise SystemExit("No toxic comments matched parent comments with post_number")

    summary = (
        matched_parents.groupby(post_number_column)
        .size()
        .rename("toxic_comment_count")
        .reset_index()
        .sort_values(post_number_column)
    )

    first_post_number = int(summary[post_number_column].min())
    last_post_number = int(summary[post_number_column].max())
    full_post_numbers = pd.DataFrame(
        {post_number_column: range(first_post_number, last_post_number + 1)}
    )
    summary = full_post_numbers.merge(summary, on=post_number_column, how="left")
    summary["toxic_comment_count"] = (
        summary["toxic_comment_count"].fillna(0).astype(int)
    )

    total_toxic_comments = len(toxic)
    matched_toxic_comments = int(matched[post_number_column].notna().sum())
    plotted_toxic_comments = int(summary["toxic_comment_count"].sum())
    summary["percent_of_matched_toxic_comments"] = (
        summary["toxic_comment_count"] / matched_toxic_comments * 100
    )

    return summary, {
        "total_toxic_comments_with_t1_parent": total_toxic_comments,
        "matched_toxic_comments": matched_toxic_comments,
        "plotted_toxic_comments": plotted_toxic_comments,
    }


def plot_histogram(summary, post_number_column, output_png, title, counts):
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.bar(
        summary[post_number_column],
        summary["toxic_comment_count"],
        width=0.9,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Parent comment post number")
    ax.set_ylabel("Toxic comments directed at parent comments")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.13,
        (
            "Toxic comments with t1_ parents: "
            f"{counts['total_toxic_comments_with_t1_parent']:,}; "
            f"matched parents: {counts['matched_toxic_comments']:,}"
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

    if not args.toxic_csv.is_file():
        raise SystemExit(f"Toxic comments CSV does not exist: {args.toxic_csv}")
    if not args.all_comments_csv.is_file():
        raise SystemExit(
            f"All-comments CSV does not exist: {args.all_comments_csv}"
        )
    if args.max_post_number is not None and args.max_post_number < 1:
        raise SystemExit("--max-post-number must be at least 1")

    toxic_columns = set(pd.read_csv(args.toxic_csv, nrows=0).columns)
    if args.parent_id_column not in toxic_columns:
        raise SystemExit(
            f"Toxic comments CSV is missing parent id column: {args.parent_id_column}"
        )

    all_columns = set(pd.read_csv(args.all_comments_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(
        all_columns,
        args.comment_id_column,
    )
    if args.post_number_column not in all_columns:
        raise SystemExit(
            f"All-comments CSV is missing post number column: {args.post_number_column}"
        )

    toxic = load_toxic_parent_ids(args.toxic_csv, args.parent_id_column)
    parent_lookup = load_parent_lookup(
        args.all_comments_csv,
        comment_id_column,
        args.post_number_column,
    )
    summary, counts = summarize_toxic_comments_by_parent_post_number(
        toxic,
        parent_lookup,
        args.post_number_column,
        args.max_post_number,
    )

    output_png = args.output or infer_visualization_output(
        args.toxic_csv,
        f"{args.toxic_csv.stem}_cultural_violence_parent_post_numbers.png",
    )
    title = args.title or (
        f"{args.toxic_csv.stem}: Toxic Comments by Parent Post Number"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_histogram(summary, args.post_number_column, output_png, title, counts)

    print(f"Saved {output_png}")
    print(
        "Toxic comments with t1_ parent ids: "
        f"{counts['total_toxic_comments_with_t1_parent']:,}"
    )
    print(f"Toxic comments with matched parents: {counts['matched_toxic_comments']:,}")
    print(f"Toxic comments plotted: {counts['plotted_toxic_comments']:,}")
    print(f"Parent post numbers plotted: {len(summary):,}")


if __name__ == "__main__":
    main()
