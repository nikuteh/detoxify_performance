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
            "Match comments to their direct parent comments, then visualize "
            "toxic-reply contagion and parent-child toxicity deltas."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Community predictions CSV.")
    parser.add_argument(
        "--contagion-output",
        type=Path,
        help="Output PNG for toxic-parent vs non-toxic-parent reply toxicity.",
    )
    parser.add_argument(
        "--delta-output",
        type=Path,
        help="Output PNG for child-minus-parent toxicity deltas.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV summarizing parent toxicity groups.",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        help="Optional output CSV with every matched parent-child pair.",
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
        help="Toxicity score column to compare. Default: toxicity.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Toxicity threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--title-prefix",
        help="Prefix used for plot titles. Default: inferred from input CSV.",
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
        "Input CSV needs a comment id column to match direct replies. "
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


def load_pairs(input_csv, comment_id_column, parent_id_column, score_column, threshold):
    usecols = [comment_id_column, parent_id_column, score_column]
    df = pd.read_csv(input_csv, usecols=usecols)
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df["normalized_comment_id"] = df[comment_id_column].map(normalize_comment_id)
    df["normalized_parent_id"] = df[parent_id_column].map(normalize_parent_id)
    df = df.dropna(subset=[score_column])
    df = df[df["normalized_comment_id"] != ""].copy()

    parents = (
        df[["normalized_comment_id", score_column]]
        .drop_duplicates("normalized_comment_id", keep="first")
        .rename(
            columns={
                "normalized_comment_id": "normalized_parent_id",
                score_column: "parent_toxicity",
            }
        )
    )
    children = df[
        ["normalized_comment_id", "normalized_parent_id", score_column]
    ].rename(
        columns={
            "normalized_comment_id": "child_comment_id",
            score_column: "child_toxicity",
        }
    )
    children = children[children["normalized_parent_id"] != ""].copy()

    pairs = children.merge(parents, on="normalized_parent_id", how="inner")
    if pairs.empty:
        raise SystemExit("No direct parent-child comment pairs could be matched")

    pairs["parent_above_threshold"] = pairs["parent_toxicity"] > threshold
    pairs["child_above_threshold"] = pairs["child_toxicity"] > threshold
    pairs["toxicity_delta"] = pairs["child_toxicity"] - pairs["parent_toxicity"]
    return pairs


def summarize_contagion(pairs, threshold):
    pairs = pairs.copy()
    pairs["parent_group"] = pairs["parent_above_threshold"].map(
        {
            False: f"Parent <= {threshold:g}",
            True: f"Parent > {threshold:g}",
        }
    )
    summary = (
        pairs.groupby("parent_group", sort=False)
        .agg(
            matched_replies=("child_toxicity", "size"),
            average_reply_toxicity=("child_toxicity", "mean"),
            median_reply_toxicity=("child_toxicity", "median"),
            toxic_replies=("child_above_threshold", "sum"),
            average_parent_toxicity=("parent_toxicity", "mean"),
            average_toxicity_delta=("toxicity_delta", "mean"),
            median_toxicity_delta=("toxicity_delta", "median"),
        )
        .reset_index()
    )
    summary["percent_replies_toxic"] = (
        summary["toxic_replies"] / summary["matched_replies"] * 100
    )
    summary["threshold"] = threshold
    return summary


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_contagion(summary, output_png, title, threshold):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        summary["parent_group"],
        summary["percent_replies_toxic"],
        color=["#4C78A8", "#F58518"][: len(summary)],
        edgecolor="white",
    )
    for bar, value, count in zip(
        bars,
        summary["percent_replies_toxic"],
        summary["matched_replies"],
        strict=False,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%\n({int(count):,} replies)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Parent comment toxicity group")
    ax.set_ylabel(f"Percent of direct replies above {threshold:g} toxicity")
    ax.set_ylim(0, max(5, float(summary["percent_replies_toxic"].max()) * 1.25))
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_delta_distribution(pairs, output_png, title):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.hist(
        pairs["toxicity_delta"],
        bins=60,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0, color="#F58518", linewidth=2.2, label="No change")
    ax.axvline(
        pairs["toxicity_delta"].mean(),
        color="#54A24B",
        linewidth=2,
        linestyle="--",
        label=f"Mean delta {pairs['toxicity_delta'].mean():.4f}",
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Reply toxicity minus parent toxicity")
    ax.set_ylabel("Matched parent-child pairs")
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.14,
        f"Pairs matched: {len(pairs):,}",
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

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(columns, args.comment_id_column)
    required = {args.parent_id_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    pairs = load_pairs(
        args.input_csv,
        comment_id_column,
        args.parent_id_column,
        args.score_column,
        args.threshold,
    )
    summary = summarize_contagion(pairs, args.threshold)

    contagion_output = args.contagion_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_toxic_reply_contagion.png",
    )
    delta_output = args.delta_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_parent_child_toxicity_delta.png",
    )
    title_prefix = args.title_prefix or args.input_csv.stem

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    if args.pairs_output:
        args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.pairs_output, index=False)
        print(f"Saved {args.pairs_output}")

    plot_contagion(
        summary,
        contagion_output,
        f"{title_prefix}: Toxic Reply Contagion",
        args.threshold,
    )
    plot_delta_distribution(
        pairs,
        delta_output,
        f"{title_prefix}: Parent-Child Toxicity Delta",
    )

    print(f"Saved {contagion_output}")
    print(f"Saved {delta_output}")
    print(f"Matched parent-child pairs: {len(pairs):,}")


if __name__ == "__main__":
    main()
