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
            "Find parent comments that received multiple direct toxic replies "
            "and plot the strongest dogpiling candidates."
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
        help="Optional output CSV with parent comments ranked by toxic replies.",
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
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--text-column",
        default="comment_text",
        help="Column containing comment text. Default: comment_text.",
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
        help="Reply toxicity threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--min-toxic-replies",
        type=int,
        default=2,
        help="Minimum direct toxic replies required for dogpiling. Default: 2.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of parent comments to plot. Default: 20.",
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
        "Input CSV needs a comment id column to identify dogpiling candidates. "
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


def load_comments(args, comment_id_column):
    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    usecols = [comment_id_column, args.parent_id_column, args.score_column]
    optional = [args.username_column, args.text_column]
    usecols.extend(column for column in optional if column in columns)

    df = pd.read_csv(args.input_csv, usecols=usecols)
    df[args.score_column] = pd.to_numeric(df[args.score_column], errors="coerce")
    df["normalized_comment_id"] = df[comment_id_column].map(normalize_comment_id)
    df["normalized_parent_id"] = df[args.parent_id_column].map(normalize_parent_id)
    df = df.dropna(subset=[args.score_column])
    df = df[df["normalized_comment_id"] != ""].copy()

    if args.username_column not in df.columns:
        df[args.username_column] = ""
    if args.text_column not in df.columns:
        df[args.text_column] = ""

    if df.empty:
        raise SystemExit("No usable comments found after cleaning")

    return df


def summarize_dogpiling(df, args):
    comments = df.drop_duplicates("normalized_comment_id", keep="first").copy()
    replies = df[df["normalized_parent_id"] != ""].copy()
    replies["reply_above_threshold"] = replies[args.score_column] > args.threshold

    reply_counts = (
        replies.groupby("normalized_parent_id")
        .agg(
            total_direct_replies=(args.score_column, "size"),
            toxic_direct_replies=("reply_above_threshold", "sum"),
            average_reply_toxicity=(args.score_column, "mean"),
        )
        .reset_index()
    )
    toxic_reply_users = (
        replies[replies["reply_above_threshold"]]
        .groupby("normalized_parent_id")[args.username_column]
        .nunique()
        .rename("unique_toxic_reply_users")
        .reset_index()
    )
    reply_counts = reply_counts.merge(
        toxic_reply_users,
        on="normalized_parent_id",
        how="left",
    )
    reply_counts["unique_toxic_reply_users"] = (
        reply_counts["unique_toxic_reply_users"].fillna(0).astype(int)
    )

    parent_info = comments[
        [
            "normalized_comment_id",
            args.username_column,
            args.text_column,
            args.score_column,
        ]
    ].rename(
        columns={
            "normalized_comment_id": "normalized_parent_id",
            args.username_column: "parent_username",
            args.text_column: "parent_comment_text",
            args.score_column: "parent_toxicity",
        }
    )

    summary = reply_counts.merge(parent_info, on="normalized_parent_id", how="inner")
    summary = summary[
        summary["toxic_direct_replies"] >= args.min_toxic_replies
    ].copy()
    summary["percent_direct_replies_toxic"] = (
        summary["toxic_direct_replies"] / summary["total_direct_replies"] * 100
    )
    summary = summary.sort_values(
        [
            "toxic_direct_replies",
            "unique_toxic_reply_users",
            "total_direct_replies",
            "average_reply_toxicity",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    summary["threshold"] = args.threshold
    return summary


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_empty(output_png, title):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.axis("off")
    ax.set_title(title, pad=12)
    ax.text(
        0.5,
        0.5,
        "No parent comments met the dogpiling threshold.",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_dogpiling(summary, output_png, title, top_n):
    if summary.empty:
        plot_empty(output_png, title)
        return

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plot_data = summary.head(top_n).iloc[::-1].copy()
    plot_data["label"] = (
        plot_data["rank"].astype(str)
        + ". "
        + plot_data["parent_username"].fillna("").astype(str).str.slice(0, 18)
        + " "
        + plot_data["normalized_parent_id"].astype(str).str.replace("t1_", "", regex=False)
    )

    fig_height = max(6, 0.42 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        plot_data["label"],
        plot_data["toxic_direct_replies"],
        color="#E45756",
        edgecolor="white",
    )
    for bar, toxic_count, total_count in zip(
        bars,
        plot_data["toxic_direct_replies"],
        plot_data["total_direct_replies"],
    ):
        ax.text(
            toxic_count + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{int(toxic_count):,} toxic / {int(total_count):,} replies",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Direct toxic replies received")
    ax.set_ylabel("Parent comment")
    ax.set_xlim(0, max(1, float(plot_data["toxic_direct_replies"].max()) * 1.25))
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.min_toxic_replies < 1:
        raise SystemExit("--min-toxic-replies must be at least 1")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    comment_id_column = resolve_comment_id_column(columns, args.comment_id_column)
    required = {args.parent_id_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    df = load_comments(args, comment_id_column)
    summary = summarize_dogpiling(df, args)

    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_dogpiling_candidates.png",
    )
    title = args.title or f"{args.input_csv.stem}: Dogpiling Candidates"

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_dogpiling(summary, output_png, title, args.top_n)

    print(f"Saved {output_png}")
    print(f"Dogpiling candidates: {len(summary):,}")


if __name__ == "__main__":
    main()
