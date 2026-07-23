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
            "Plot how concentrated toxic comments are among the most toxic "
            "or most frequently toxic users."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Community predictions CSV.")
    parser.add_argument(
        "--curve-output",
        type=Path,
        help="Output PNG for cumulative concentration curve.",
    )
    parser.add_argument(
        "--top-users-output",
        type=Path,
        help="Output PNG for top toxic-comment contributors.",
    )
    parser.add_argument(
        "--top-percent-users-output",
        type=Path,
        help="Output PNG for top users ranked by percent toxic comments.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV with ranked users and cumulative shares.",
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
        "--top-n",
        type=int,
        default=20,
        help="Number of users for top-users bar chart. Default: 20.",
    )
    parser.add_argument(
        "--min-comments-for-percent",
        type=int,
        default=1,
        help=(
            "Minimum total comments required for users to appear in the "
            "percent-toxic chart. Default: 1."
        ),
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames.",
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


def summarize_concentration(args):
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
            average_toxicity=(args.score_column, "mean"),
            max_toxicity=(args.score_column, "max"),
        )
        .reset_index()
        .rename(columns={args.username_column: "username"})
    )
    users["percent_toxic_comments"] = (
        users["toxic_comment_count"] / users["comment_count"] * 100
    )
    users = users.sort_values(
        ["toxic_comment_count", "average_toxicity", "comment_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    users.insert(0, "rank", users.index + 1)

    total_users = len(users)
    total_comments = int(users["comment_count"].sum())
    total_toxic_comments = int(users["toxic_comment_count"].sum())
    users["cumulative_users"] = users["rank"]
    users["cumulative_percent_users"] = users["cumulative_users"] / total_users * 100
    users["cumulative_comments"] = users["comment_count"].cumsum()
    users["cumulative_percent_comments"] = (
        users["cumulative_comments"] / total_comments * 100
    )

    if total_toxic_comments:
        users["cumulative_toxic_comments"] = users["toxic_comment_count"].cumsum()
        users["cumulative_percent_toxic_comments"] = (
            users["cumulative_toxic_comments"] / total_toxic_comments * 100
        )
    else:
        users["cumulative_toxic_comments"] = 0
        users["cumulative_percent_toxic_comments"] = 0.0

    users["threshold"] = args.threshold
    return users, {
        "total_users": total_users,
        "total_comments": total_comments,
        "total_toxic_comments": total_toxic_comments,
    }


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
        "No comments exceeded the toxicity threshold.",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_curve(users, counts, output_png, title):
    if counts["total_toxic_comments"] == 0:
        plot_empty(output_png, title)
        return

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    ax.plot(
        users["cumulative_percent_users"],
        users["cumulative_percent_toxic_comments"],
        color="#4C78A8",
        linewidth=2.6,
        label="Observed concentration",
    )
    ax.plot([0, 100], [0, 100], color="#999999", linestyle="--", linewidth=1.4, label="Even distribution")

    for percent in (1, 5, 10):
        row = users[users["cumulative_percent_users"] >= percent].head(1)
        if row.empty:
            continue
        x = float(row["cumulative_percent_users"].iloc[0])
        y = float(row["cumulative_percent_toxic_comments"].iloc[0])
        ax.scatter([x], [y], color="#F58518", s=45, zorder=5)
        ax.text(x + 1, y, f"top {percent}%: {y:.1f}%", va="center", fontsize=9)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Cumulative percent of users")
    ax.set_ylabel("Cumulative percent of toxic comments")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.13,
        (
            f"Users: {counts['total_users']:,}; "
            f"toxic comments: {counts['total_toxic_comments']:,}"
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


def plot_top_users(users, counts, output_png, title, top_n):
    if counts["total_toxic_comments"] == 0:
        plot_empty(output_png, title)
        return

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plot_data = users[users["toxic_comment_count"] > 0].head(top_n).iloc[::-1].copy()
    plot_data["label"] = (
        plot_data["rank"].astype(str)
        + ". "
        + plot_data["username"].astype(str).str.slice(0, 22)
    )

    fig_height = max(6, 0.42 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        plot_data["label"],
        plot_data["toxic_comment_count"],
        color="#E45756",
        edgecolor="white",
    )
    for bar, toxic_count, comment_count in zip(
        bars,
        plot_data["toxic_comment_count"],
        plot_data["comment_count"],
    ):
        ax.text(
            toxic_count + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{int(toxic_count):,} toxic / {int(comment_count):,} comments",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Comments above toxicity threshold")
    ax.set_ylabel("User")
    ax.set_xlim(0, max(1, float(plot_data["toxic_comment_count"].max()) * 1.25))
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def plot_top_percent_users(users, counts, output_png, title, top_n, min_comments):
    if counts["total_toxic_comments"] == 0:
        plot_empty(output_png, title)
        return

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    eligible = users[
        (users["toxic_comment_count"] > 0) & (users["comment_count"] >= min_comments)
    ].copy()
    plot_data = eligible.head(top_n).iloc[::-1].copy()
    if plot_data.empty:
        plot_empty(output_png, title)
        return

    plot_data["label"] = (
        plot_data["rank"].astype(str)
        + ". "
        + plot_data["username"].astype(str).str.slice(0, 22)
    )

    fig_height = max(6, 0.42 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(
        plot_data["label"],
        plot_data["percent_toxic_comments"],
        color="#B279A2",
        edgecolor="white",
    )
    for bar, percent, toxic_count, comment_count in zip(
        bars,
        plot_data["percent_toxic_comments"],
        plot_data["toxic_comment_count"],
        plot_data["comment_count"],
    ):
        ax.text(
            percent + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.1f}% ({int(toxic_count):,}/{int(comment_count):,})",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Percent of user's comments above toxicity threshold")
    ax.set_ylabel("User ranked by toxic-comment count")
    ax.set_xlim(
        0,
        min(100, max(1, float(plot_data["percent_toxic_comments"].max()) * 1.2)),
    )
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")
    if args.min_comments_for_percent < 1:
        raise SystemExit("--min-comments-for-percent must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.score_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )

    users, counts = summarize_concentration(args)
    curve_output = args.curve_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_toxicity_concentration_curve.png",
    )
    top_users_output = args.top_users_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_top_toxicity_contributors.png",
    )
    top_percent_users_output = args.top_percent_users_output or infer_visualization_output(
        args.input_csv,
        f"{args.input_csv.stem}_top_toxicity_contributors_by_percent.png",
    )
    title_prefix = args.title_prefix or args.input_csv.stem

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        users.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    plot_curve(users, counts, curve_output, f"{title_prefix}: Toxicity Concentration")
    plot_top_users(
        users,
        counts,
        top_users_output,
        f"{title_prefix}: Top Toxicity Contributors",
        args.top_n,
    )
    plot_top_percent_users(
        users,
        counts,
        top_percent_users_output,
        f"{title_prefix}: Top Toxicity Contributors Measured by Percent Toxic",
        args.top_n,
        args.min_comments_for_percent,
    )

    print(f"Saved {curve_output}")
    print(f"Saved {top_users_output}")
    print(f"Saved {top_percent_users_output}")
    print(f"Users ranked: {counts['total_users']:,}")
    print(f"Toxic comments counted: {counts['total_toxic_comments']:,}")


if __name__ == "__main__":
    main()
