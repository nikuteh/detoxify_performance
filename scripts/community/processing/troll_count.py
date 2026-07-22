import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
DEFAULT_COMMUNITY_OUTPUT = DATA_SUBREDDITS_DIR / "trolls_community.csv"
DEFAULT_PLOT_OUTPUT = VISUALIZATIONS_DIR / "trolls_community_percent_trolls.png"
SUMMARY_COLUMNS = [
    "subreddit",
    "score_column",
    "threshold",
    "total_users",
    "users_above_threshold",
    "percent_users_above_threshold",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count users whose average toxicity across all scored comments is "
            "above a threshold, update the shared community trolls CSV, and "
            "plot all processed subreddits by percent of trolls."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Subreddit CSV containing username and Detoxify toxicity columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_COMMUNITY_OUTPUT,
        help=(
            "Shared community summary CSV to update. Default: "
            "data/subreddits/trolls_community.csv."
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=(
            "Output PNG for the all-subreddits troll percentage bar chart. "
            "Default: visualizations/trolls_community_percent_trolls.png."
        ),
    )
    parser.add_argument(
        "--subreddit",
        help=(
            "Subreddit name to store in the shared CSV. Default: inferred "
            "from the input CSV path."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Average toxicity cutoff. Default: 0.25.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Score column to average. Default: toxicity.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows to read at a time. Default: 100000.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames in the user count.",
    )
    return parser.parse_args()


def infer_subreddit(input_csv):
    try:
        relative_input = input_csv.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        return input_csv.parent.name or input_csv.stem

    if relative_input.parts:
        return relative_input.parts[0]
    return input_csv.parent.name or input_csv.stem


def validate_columns(input_csv, username_column, score_column):
    columns = set(pd.read_csv(input_csv, nrows=0).columns)
    missing = [
        column
        for column in (username_column, score_column)
        if column not in columns
    ]
    if missing:
        raise SystemExit(
            f"{input_csv} is missing required column(s): {', '.join(missing)}"
        )


def count_users_above_average_toxicity(
    input_csv,
    username_column,
    score_column,
    threshold,
    chunk_size,
    include_deleted,
):
    sums = pd.Series(dtype="float64")
    counts = pd.Series(dtype="int64")
    deleted_names = {"[deleted]", "[removed]"}

    for chunk in pd.read_csv(
        input_csv,
        usecols=[username_column, score_column],
        chunksize=chunk_size,
    ):
        usernames = chunk[username_column].fillna("").astype(str).str.strip()
        scores = pd.to_numeric(chunk[score_column], errors="coerce")
        valid_rows = usernames.ne("") & scores.notna()

        if not include_deleted:
            valid_rows &= ~usernames.str.lower().isin(deleted_names)

        chunk = pd.DataFrame(
            {
                "username": usernames[valid_rows],
                "score": scores[valid_rows],
            }
        )
        if chunk.empty:
            continue

        chunk_sums = chunk.groupby("username")["score"].sum()
        chunk_counts = chunk.groupby("username")["score"].count()
        sums = sums.add(chunk_sums, fill_value=0)
        counts = counts.add(chunk_counts, fill_value=0)

    if counts.empty:
        raise SystemExit(
            f"No users with valid {score_column!r} scores found in {input_csv}"
        )

    average_scores = sums / counts
    users_above_threshold = int((average_scores > threshold).sum())
    total_users = int(counts.size)
    percent_above_threshold = users_above_threshold / total_users * 100

    return {
        "subreddit": None,
        "score_column": score_column,
        "threshold": threshold,
        "total_users": total_users,
        "users_above_threshold": users_above_threshold,
        "percent_users_above_threshold": percent_above_threshold,
    }


def update_community_csv(output_csv, summary):
    summary_df = pd.DataFrame([summary], columns=SUMMARY_COLUMNS)

    if output_csv.is_file() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv)
    else:
        existing = pd.DataFrame(columns=SUMMARY_COLUMNS)

    for column in SUMMARY_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA

    existing = existing[SUMMARY_COLUMNS].copy()
    same_summary = (
        existing["subreddit"].astype(str).eq(str(summary["subreddit"]))
        & existing["score_column"].astype(str).eq(str(summary["score_column"]))
        & pd.to_numeric(existing["threshold"], errors="coerce").eq(
            float(summary["threshold"])
        )
    )
    existing = existing[~same_summary]

    if existing.empty:
        combined = summary_df.copy()
    else:
        combined = pd.concat([existing, summary_df], ignore_index=True)
    combined["threshold"] = pd.to_numeric(combined["threshold"], errors="coerce")
    combined["total_users"] = pd.to_numeric(
        combined["total_users"], errors="coerce"
    ).astype("Int64")
    combined["users_above_threshold"] = pd.to_numeric(
        combined["users_above_threshold"], errors="coerce"
    ).astype("Int64")
    combined["percent_users_above_threshold"] = pd.to_numeric(
        combined["percent_users_above_threshold"], errors="coerce"
    )
    combined = combined.sort_values(
        ["percent_users_above_threshold", "subreddit"],
        ascending=[False, True],
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined


def plot_troll_percentages(community_summary, output_png):
    plot_data = community_summary.dropna(
        subset=["subreddit", "percent_users_above_threshold"]
    ).copy()
    if plot_data.empty:
        raise SystemExit("No community troll rows available to plot")

    plot_data["percent_users_above_threshold"] = pd.to_numeric(
        plot_data["percent_users_above_threshold"], errors="coerce"
    )
    plot_data["users_above_threshold"] = pd.to_numeric(
        plot_data["users_above_threshold"], errors="coerce"
    )
    plot_data["total_users"] = pd.to_numeric(
        plot_data["total_users"], errors="coerce"
    )
    plot_data = plot_data.dropna(subset=["percent_users_above_threshold"])
    plot_data = plot_data.sort_values("percent_users_above_threshold", ascending=True)

    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    labels = plot_data["subreddit"].astype(str).tolist()
    values = plot_data["percent_users_above_threshold"].to_numpy(dtype=float)
    troll_counts = plot_data["users_above_threshold"].fillna(0).astype(int).tolist()
    total_users = plot_data["total_users"].fillna(0).astype(int).tolist()

    fig_height = max(5, 0.45 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(labels, values, color="#4C78A8", edgecolor="white")

    max_value = values.max() if len(values) else 0
    label_offset = max(max_value * 0.015, 0.03)
    for bar, value, troll_count, user_count in zip(
        bars,
        values,
        troll_counts,
        total_users,
    ):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}% ({troll_count:,}/{user_count:,})",
            va="center",
            fontsize=9,
        )

    threshold_values = plot_data["threshold"].dropna().unique()
    threshold_text = (
        f" | Threshold: {threshold_values[0]:g}"
        if len(threshold_values) == 1
        else ""
    )
    ax.set_title(f"Percent of Troll Users by Subreddit{threshold_text}", pad=12)
    ax.set_xlabel("Percent of users above average toxicity threshold")
    ax.set_ylabel("Subreddit")
    ax.set_xlim(0, max_value + label_offset * 12 if max_value else 1)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)

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

    validate_columns(args.input_csv, args.username_column, args.score_column)

    summary = count_users_above_average_toxicity(
        args.input_csv,
        args.username_column,
        args.score_column,
        args.threshold,
        args.chunk_size,
        args.include_deleted,
    )
    summary["subreddit"] = args.subreddit or infer_subreddit(args.input_csv)

    display = pd.DataFrame([summary])
    display["percent_users_above_threshold"] = display[
        "percent_users_above_threshold"
    ].round(4)
    print(display.to_string(index=False))

    community_summary = update_community_csv(args.output, summary)
    print(f"Saved {args.output}")

    if args.plot_output:
        plot_troll_percentages(community_summary, args.plot_output)
        print(f"Saved {args.plot_output}")


if __name__ == "__main__":
    main()
