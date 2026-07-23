import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"
DELETED_USERNAMES = {"[deleted]", "[removed]", "deleted", "removed"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the average number of toxic comments per user for the "
            "top active users against all users."
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
        help="Optional output CSV with the two cohort averages.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
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
        help="Toxicity threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--top-active-users",
        type=int,
        default=100,
        help="Number of most active users to compare. Default: 100.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames.",
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


def load_user_toxic_counts(args):
    df = pd.read_csv(args.input_csv, usecols=[args.username_column, args.score_column])
    df[args.username_column] = (
        df[args.username_column].fillna("").astype(str).str.strip()
    )
    df[args.score_column] = pd.to_numeric(df[args.score_column], errors="coerce")
    df = df[(df[args.username_column] != "") & df[args.score_column].notna()].copy()

    if not args.include_deleted:
        df = df[~df[args.username_column].str.lower().isin(DELETED_USERNAMES)].copy()

    if df.empty:
        raise SystemExit("No usable user rows found after cleaning")

    df["above_threshold"] = df[args.score_column] > args.threshold
    users = (
        df.groupby(args.username_column)
        .agg(
            comment_count=(args.score_column, "size"),
            toxic_comment_count=("above_threshold", "sum"),
        )
        .reset_index()
        .rename(columns={args.username_column: "username"})
    )
    return users.sort_values(
        ["comment_count", "toxic_comment_count", "username"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def summarize_cohorts(users, args):
    top_active_users = users.head(args.top_active_users).copy()
    if top_active_users.empty:
        raise SystemExit("No users available for the top-active-users cohort")

    cohorts = [
        ("All users", users),
        (f"Top {len(top_active_users)} active users", top_active_users),
    ]
    rows = []
    for cohort, cohort_users in cohorts:
        rows.append(
            {
                "cohort": cohort,
                "user_count": len(cohort_users),
                "total_comments": int(cohort_users["comment_count"].sum()),
                "total_toxic_comments": int(
                    cohort_users["toxic_comment_count"].sum()
                ),
                "average_toxic_comments_per_user": float(
                    cohort_users["toxic_comment_count"].mean()
                ),
                "median_toxic_comments_per_user": float(
                    cohort_users["toxic_comment_count"].median()
                ),
                "threshold": args.threshold,
            }
        )

    return pd.DataFrame(rows)


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_average_toxic_comments(summary, output_png, title):
    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        summary["cohort"],
        summary["average_toxic_comments_per_user"],
        color=["#4C78A8", "#E45756"][: len(summary)],
        edgecolor="white",
    )
    for bar, average, user_count in zip(
        bars,
        summary["average_toxic_comments_per_user"],
        summary["user_count"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            average + 0.03,
            f"{average:.2f}\n({int(user_count):,} users)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("User cohort")
    ax.set_ylabel("Average toxic comments per user")
    ax.set_ylim(
        0,
        max(1, float(summary["average_toxic_comments_per_user"].max()) * 1.25),
    )
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.top_active_users < 1:
        raise SystemExit("--top-active-users must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    summary = summarize_cohorts(load_user_toxic_counts(args), args)
    output_png = args.output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_top_active_toxic_comment_average.png",
    )
    title = args.title or (
        f"{args.input_csv.stem}: Toxic Comments for Top Active Users vs All Users"
    )

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_average_toxic_comments(summary, output_png, title)
    print(f"Saved {output_png}")
    print(f"Cohorts plotted: {len(summary):,}")


if __name__ == "__main__":
    main()
