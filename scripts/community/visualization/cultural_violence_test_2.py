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
        "--likelihood-output",
        type=Path,
        help=(
            "Output PNG for percent of comments at each post_number that "
            "divides total comments by toxic responses. Default: inferred "
            "from the toxic CSV filename."
        ),
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
        default=2000,
        help="Only include parent post numbers up to this value. Default: 2000.",
    )
    parser.add_argument(
        "--max-plot-bars",
        type=int,
        default=200,
        help=(
            "Maximum bars to draw before aggregating adjacent post numbers "
            "into visible ranges. Default: 200."
        ),
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
    matched_parents = toxic.merge(
        parent_lookup,
        how="inner",
        left_on="normalized_parent_id",
        right_on="normalized_comment_id",
    )
    matched_parents = matched_parents.dropna(subset=[post_number_column]).copy()
    matched_parents[post_number_column] = matched_parents[
        post_number_column
    ].astype(int)

    linked_toxic_comments = len(matched_parents)
    if max_post_number is not None:
        matched_parents = matched_parents[
            matched_parents[post_number_column] <= max_post_number
        ].copy()

    if matched_parents.empty:
        raise SystemExit("No toxic comments matched parent comments with post_number")

    toxic_reply_counts = (
        matched_parents.groupby(post_number_column)
        .size()
        .rename("toxic_comment_count")
        .reset_index()
        .sort_values(post_number_column)
    )
    parents_with_toxic_replies = (
        matched_parents.drop_duplicates("normalized_parent_id")
        .groupby(post_number_column)
        .size()
        .rename("parent_comments_with_toxic_response")
        .reset_index()
    )
    parent_totals = (
        parent_lookup.groupby(post_number_column)
        .size()
        .rename("total_parent_comments")
        .reset_index()
    )
    if max_post_number is not None:
        parent_totals = parent_totals[
            parent_totals[post_number_column] <= max_post_number
        ].copy()

    summary = parent_totals.merge(toxic_reply_counts, on=post_number_column, how="left")
    summary = summary.merge(
        parents_with_toxic_replies,
        on=post_number_column,
        how="left",
    )
    summary["toxic_comment_count"] = (
        summary["toxic_comment_count"].fillna(0).astype(int)
    )
    summary["parent_comments_with_toxic_response"] = (
        summary["parent_comments_with_toxic_response"].fillna(0).astype(int)
    )
    summary["percent_comments_per_toxic_response"] = pd.NA
    has_toxic_responses = summary["toxic_comment_count"] > 0
    summary.loc[has_toxic_responses, "percent_comments_per_toxic_response"] = (
        summary.loc[has_toxic_responses, "total_parent_comments"]
        / summary.loc[has_toxic_responses, "toxic_comment_count"]
        * 100
    )

    toxic_comments_with_t1_parent = len(toxic)
    plotted_toxic_comments = int(summary["toxic_comment_count"].sum())
    summary["percent_of_matched_toxic_comments"] = (
        summary["toxic_comment_count"] / linked_toxic_comments * 100
    )

    return summary, {
        "toxic_comments_with_t1_parent": toxic_comments_with_t1_parent,
        "dropped_unlinked_toxic_comments": (
            toxic_comments_with_t1_parent - linked_toxic_comments
        ),
        "linked_toxic_comments": linked_toxic_comments,
        "plotted_toxic_comments": plotted_toxic_comments,
        "parent_comments_with_toxic_response": int(
            summary["parent_comments_with_toxic_response"].sum()
        ),
        "post_numbers_in_summary": len(summary),
        "post_numbers_with_toxic_response": int(
            (summary["parent_comments_with_toxic_response"] > 0).sum()
        ),
        "max_post_number": max_post_number,
    }


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def set_post_number_ticks(ax, plot_data, post_number_column):
    tick_count = min(12, len(plot_data))
    tick_step = max(len(plot_data) // tick_count, 1)
    tick_positions = list(range(0, len(plot_data), tick_step))
    if tick_positions[-1] != len(plot_data) - 1:
        tick_positions.append(len(plot_data) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        plot_data.loc[tick_positions, post_number_column].astype(str),
        rotation=45,
        ha="right",
    )


def add_post_number_labels(plot_data, post_number_column):
    plot_data = plot_data.copy()
    if "post_number_label" in plot_data.columns:
        return plot_data

    plot_data["post_number_label"] = plot_data[post_number_column].astype(str)
    return plot_data


def bin_for_plot(summary, post_number_column, max_plot_bars):
    if max_plot_bars < 1:
        raise SystemExit("--max-plot-bars must be at least 1")

    if len(summary) <= max_plot_bars:
        return add_post_number_labels(summary, post_number_column), False

    sorted_summary = summary.sort_values(post_number_column).reset_index(drop=True)
    bin_size = (len(sorted_summary) + max_plot_bars - 1) // max_plot_bars
    sorted_summary["plot_bin"] = sorted_summary.index // bin_size

    plot_data = (
        sorted_summary.groupby("plot_bin")
        .agg(
            post_number_min=(post_number_column, "min"),
            post_number_max=(post_number_column, "max"),
            toxic_comment_count=("toxic_comment_count", "sum"),
            parent_comments_with_toxic_response=(
                "parent_comments_with_toxic_response",
                "sum",
            ),
            total_parent_comments=("total_parent_comments", "sum"),
        )
        .reset_index(drop=True)
    )
    plot_data["percent_comments_per_toxic_response"] = (
        plot_data["total_parent_comments"]
        / plot_data["toxic_comment_count"]
        * 100
    )
    plot_data["post_number_label"] = plot_data["post_number_min"].astype(str)
    ranged_bins = plot_data["post_number_min"] != plot_data["post_number_max"]
    plot_data.loc[ranged_bins, "post_number_label"] = (
        plot_data.loc[ranged_bins, "post_number_min"].astype(str)
        + "-"
        + plot_data.loc[ranged_bins, "post_number_max"].astype(str)
    )
    return plot_data, True


def set_plot_ticks(ax, plot_data):
    tick_count = min(12, len(plot_data))
    tick_step = max(len(plot_data) // tick_count, 1)
    tick_positions = list(range(0, len(plot_data), tick_step))
    if tick_positions[-1] != len(plot_data) - 1:
        tick_positions.append(len(plot_data) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        plot_data.loc[tick_positions, "post_number_label"],
        rotation=45,
        ha="right",
    )


def plot_count_histogram(
    summary,
    post_number_column,
    output_png,
    title,
    counts,
    max_plot_bars,
):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    plot_data = summary[summary["toxic_comment_count"] > 0].reset_index(drop=True)
    if plot_data.empty:
        raise SystemExit("No parent post numbers have toxic comment counts to plot")
    plot_data, was_binned = bin_for_plot(plot_data, post_number_column, max_plot_bars)

    x_positions = range(len(plot_data))
    ax.bar(
        x_positions,
        plot_data["toxic_comment_count"],
        width=0.85,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    set_plot_ticks(ax, plot_data)
    ax.set_title(title, pad=12)
    xlabel = "Parent comment post number with at least one toxic reply"
    if was_binned:
        xlabel = "Parent comment post number range with at least one toxic reply"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Toxic comments directed at parent comments")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.13,
        (
            "Toxic comments with t1_ parents: "
            f"{counts['toxic_comments_with_t1_parent']:,}; "
            f"linked to parents: {counts['linked_toxic_comments']:,}; "
            f"post numbers shown: {len(plot_data):,}"
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


def plot_percent_histogram(
    summary,
    post_number_column,
    output_png,
    title,
    counts,
    max_plot_bars,
):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    plot_data = summary[
        summary["toxic_comment_count"] > 0
    ].reset_index(drop=True)
    if plot_data.empty:
        raise SystemExit("No parent post numbers have toxic-response percent to plot")
    plot_data, was_binned = bin_for_plot(plot_data, post_number_column, max_plot_bars)

    x_positions = range(len(plot_data))
    ax.bar(
        x_positions,
        plot_data["percent_comments_per_toxic_response"],
        width=0.85,
        color="#F58518",
        edgecolor="white",
        linewidth=0.4,
    )
    set_plot_ticks(ax, plot_data)
    ax.set_title(title, pad=12)
    xlabel = "Parent comment post number with at least one toxic reply"
    if was_binned:
        xlabel = "Parent comment post number range with at least one toxic reply"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Comments divided by toxic responses (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)
    ax.text(
        1,
        -0.13,
        (
            "Max parent post number: "
            f"{counts['max_post_number']:,}; "
            f"post numbers shown: {len(plot_data):,}"
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
    if args.max_plot_bars < 1:
        raise SystemExit("--max-plot-bars must be at least 1")

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
    likelihood_output_png = args.likelihood_output or infer_visualization_output(
        args.toxic_csv,
        (
            f"{args.toxic_csv.stem}_cultural_violence_parent_post_number_"
            "comments_per_toxic_response_percent.png"
        ),
    )
    title = args.title or (
        f"{args.toxic_csv.stem}: Toxic Comments by Parent Post Number"
    )
    likelihood_title = (
        f"{args.toxic_csv.stem}: Comments per Toxic Response by Parent Post Number"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_count_histogram(
        summary,
        args.post_number_column,
        output_png,
        title,
        counts,
        args.max_plot_bars,
    )
    plot_percent_histogram(
        summary,
        args.post_number_column,
        likelihood_output_png,
        likelihood_title,
        counts,
        args.max_plot_bars,
    )

    print(f"Saved {output_png}")
    print(f"Saved {likelihood_output_png}")
    print(
        "Toxic comments with t1_ parent ids: "
        f"{counts['toxic_comments_with_t1_parent']:,}"
    )
    print(
        "Dropped toxic comments without linked parents: "
        f"{counts['dropped_unlinked_toxic_comments']:,}"
    )
    print(f"Toxic comments linked to parents: {counts['linked_toxic_comments']:,}")
    print(f"Toxic comments plotted: {counts['plotted_toxic_comments']:,}")
    print(
        "Parent comments with at least one toxic reply: "
        f"{counts['parent_comments_with_toxic_response']:,}"
    )
    print(f"Parent post numbers in summary: {counts['post_numbers_in_summary']:,}")
    print(
        "Parent post numbers plotted: "
        f"{counts['post_numbers_with_toxic_response']:,}"
    )


if __name__ == "__main__":
    main()
