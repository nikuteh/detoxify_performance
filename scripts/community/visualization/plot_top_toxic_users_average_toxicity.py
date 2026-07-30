import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATION_SUBREDDITS_DIR = PROJECT_ROOT / "visualizations" / "subreddits"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rank top users in a subreddit folder by average toxicity, then "
            "plot toxic-comment volume from those users over time as a "
            "scatter plot with a linear regression line."
        )
    )
    parser.add_argument(
        "user_csv_folder",
        type=Path,
        help="Folder containing one CSV file per user for a subreddit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path. Default: inferred from the input folder."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of most toxic users to analyze. Default: 30.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Score column to average. Default: toxicity.",
    )
    parser.add_argument(
        "--min-users-per-post",
        type=int,
        default=None,
        help=(
            "Deprecated; retained for compatibility with older runner calls."
        ),
    )
    parser.add_argument(
        "--max-post-number",
        type=int,
        help="Deprecated; retained for compatibility with older runner calls.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Only count comments above this toxicity threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--time-bin",
        default="MS",
        help="Pandas resample interval for toxic-comment counts. Default: MS.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional output CSV with toxic-comment counts by time bin.",
    )
    return parser.parse_args()


def infer_visualization_output(user_csv_folder, filename):
    try:
        relative_folder = user_csv_folder.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        return user_csv_folder / filename

    if not relative_folder.parts:
        return user_csv_folder / filename

    subreddit_name = relative_folder.parts[0]
    return VISUALIZATION_SUBREDDITS_DIR / subreddit_name / filename


def infer_subreddit_name(user_csv_folder):
    try:
        relative_folder = user_csv_folder.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        relative_folder = None
    else:
        if relative_folder.parts:
            return relative_folder.parts[0]

    if user_csv_folder.name == "users" and user_csv_folder.parent.name:
        return user_csv_folder.parent.name

    return user_csv_folder.name


def top_users_by_average_toxicity(user_csv_folder, top_n, score_column):
    rows = []

    for csv_path in sorted(user_csv_folder.glob("*.csv")):
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except pd.errors.EmptyDataError:
            continue

        if score_column not in columns:
            continue

        usecols = [score_column]
        if "username" in columns:
            usecols.append("username")

        df = pd.read_csv(csv_path, usecols=usecols)
        scores = pd.to_numeric(df[score_column], errors="coerce").dropna()
        if scores.empty:
            continue

        if "username" in df.columns and df["username"].notna().any():
            username = str(df["username"].dropna().mode().iloc[0])
        else:
            username = csv_path.stem

        rows.append(
            {
                "username": username,
                "average_toxicity": scores.mean(),
                "comment_count": len(scores),
                "source_file": str(csv_path),
            }
        )

    if not rows:
        raise SystemExit(
            f"No CSVs with a usable {score_column!r} column found in "
            f"{user_csv_folder}"
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(["average_toxicity", "comment_count"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def load_top_user_posts(ranking, score_column):
    frames = []

    for _, user in ranking.iterrows():
        csv_path = Path(user["source_file"])

        columns = set(pd.read_csv(csv_path, nrows=0).columns)
        if "timestamp" not in columns:
            continue

        usecols = [score_column, "timestamp"]

        df = pd.read_csv(csv_path, usecols=usecols)
        df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=[score_column, "timestamp"]).copy()
        if df.empty:
            continue

        df["username"] = user["username"]
        df["rank"] = user["rank"]
        frames.append(df[["username", "rank", "timestamp", score_column]])

    if not frames:
        raise SystemExit("No usable timestamped top-user rows found to plot")

    return pd.concat(frames, ignore_index=True)


def summarize_toxic_volume_over_time(top_posts, score_column, threshold, time_bin):
    toxic_posts = top_posts[top_posts[score_column] > threshold].copy()
    if toxic_posts.empty:
        raise SystemExit(
            f"No comments from the selected top users were above {threshold:g}"
        )

    toxic_posts["date"] = pd.to_datetime(
        toxic_posts["timestamp"],
        unit="s",
        utc=True,
        errors="coerce",
    )
    toxic_posts = toxic_posts.dropna(subset=["date"]).copy()
    if toxic_posts.empty:
        raise SystemExit("No toxic comments had usable timestamps")

    indexed = toxic_posts.set_index("date").sort_index()
    counts = indexed[score_column].resample(time_bin).size()
    contributing_users = indexed["username"].resample(time_bin).nunique()
    average_toxicity = indexed[score_column].resample(time_bin).mean()

    full_index = pd.date_range(
        counts.index.min(),
        counts.index.max(),
        freq=time_bin,
        tz="UTC",
    )
    summary = pd.DataFrame(
        {
            "date": full_index,
            "toxic_comment_count": counts.reindex(full_index, fill_value=0).astype(int),
            "contributing_users": contributing_users.reindex(
                full_index,
                fill_value=0,
            ).astype(int),
            "average_toxicity": average_toxicity.reindex(full_index),
        }
    )
    summary["threshold"] = threshold
    summary["time_bin"] = time_bin
    return summary


def main():
    args = parse_args()

    if not args.user_csv_folder.is_dir():
        raise SystemExit(f"Folder does not exist: {args.user_csv_folder}")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")
    if args.threshold < 0 or args.threshold > 1:
        raise SystemExit("--threshold must be between 0 and 1")

    output_png = args.output or infer_visualization_output(
        args.user_csv_folder,
        f"top_{args.top_n}_toxic_comment_volume_over_time.png",
    )

    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    ranking = top_users_by_average_toxicity(
        args.user_csv_folder,
        args.top_n,
        args.score_column,
    )
    top_posts = load_top_user_posts(ranking, args.score_column)
    summary = summarize_toxic_volume_over_time(
        top_posts,
        args.score_column,
        args.threshold,
        args.time_bin,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
        print(f"Saved {args.summary_output}")

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    plot_dates = pd.to_datetime(summary["date"]).dt.tz_convert(None)
    x_positions = np.arange(len(summary), dtype=float)
    y = summary["toxic_comment_count"].to_numpy(dtype=float)
    contributing_users = summary["contributing_users"].to_numpy(dtype=float)
    if contributing_users.min() == contributing_users.max():
        sizes = np.full_like(contributing_users, 80.0)
    else:
        sizes = np.interp(
            contributing_users,
            (contributing_users.min(), contributing_users.max()),
            (45, 180),
        )

    ax.scatter(
        plot_dates,
        y,
        s=sizes,
        alpha=0.82,
        color="#E45756",
        edgecolors="white",
        linewidths=1.2,
        label=f"Comments above {args.threshold:g} toxicity",
    )

    if len(summary) >= 2 and np.unique(x_positions).size >= 2:
        slope, intercept = np.polyfit(x_positions, y, 1)
        x_line = np.linspace(x_positions.min(), x_positions.max(), 200)
        y_line = slope * x_line + intercept
        date_numbers = mdates.date2num(plot_dates)
        line_dates = pd.to_datetime(
            mdates.num2date(np.interp(x_line, x_positions, date_numbers))
        ).tz_convert(None)
        ax.plot(
            line_dates,
            y_line,
            color="#4C78A8",
            linewidth=2.4,
            label=f"Regression ({slope:+.2f} toxic comments per {args.time_bin})",
        )

    subreddit_name = infer_subreddit_name(args.user_csv_folder)
    ax.set_title(
        (
            f"{subreddit_name}: Toxic Comment Volume Over Time "
            f"for Top {len(ranking)} Users"
        ),
        pad=12,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Comments above {args.threshold:g} toxicity")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True)
    ax.text(
        1,
        -0.14,
        (
            f"Months plotted: {len(summary):,}; "
            f"toxic comments: {int(summary['toxic_comment_count'].sum()):,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Months plotted: {len(summary)}")
    print(f"Toxic comments counted: {int(summary['toxic_comment_count'].sum()):,}")
    print(
        ranking[
            ["rank", "username", "average_toxicity", "comment_count"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
