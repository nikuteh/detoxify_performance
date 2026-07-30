import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
COMMENT_ID_COLUMNS = ("comment_id", "id", "name")
DATE_COLUMNS = ("date", "timestamp", "created_utc", "created")
DEFAULT_SCORE_COLUMNS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
]
DELETED_USERNAMES = {"[deleted]", "[removed]", "deleted", "removed"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize every processed subreddit into one comparison CSV and "
            "plot all-community toxicity, troll, concentration, reply, and "
            "time-volatility metrics."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional subreddit names, subreddit directories, or prediction CSVs. "
            "Default: discover one prediction CSV per data/subreddits/* folder."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_SUBREDDITS_DIR,
        help="Directory containing subreddit data folders. Default: data/subreddits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_SUBREDDITS_DIR / "all_community_metrics.csv",
        help="Output CSV path for the all-community summary.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=VISUALIZATIONS_DIR / "all_communities",
        help="Directory for all-community PNG plots.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Toxicity threshold for toxic-comment percentages. Default: 0.5.",
    )
    parser.add_argument(
        "--troll-threshold",
        type=float,
        default=0.25,
        help="Average user toxicity cutoff for troll users. Default: 0.25.",
    )
    parser.add_argument(
        "--top-active-users",
        type=int,
        default=100,
        help="Number of most active users for top-active comparisons. Default: 100.",
    )
    parser.add_argument(
        "--dogpile-min-toxic-replies",
        type=int,
        default=2,
        help="Minimum toxic direct replies for dogpiling. Default: 2.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--parent-id-column",
        default="parent_id",
        help="Column containing Reddit parent ids. Default: parent_id.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Main toxicity score column. Default: toxicity.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames in user metrics.",
    )
    return parser.parse_args()


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", ".matplotlib_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def discover_prediction_csvs(data_dir):
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist: {data_dir}")

    prediction_paths = []
    for subreddit_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        cleaned = sorted(
            subreddit_dir.glob("*_cleaned_detoxify_unbiased_predictions.csv")
        )
        uncleaned = sorted(subreddit_dir.glob("*_detoxify_unbiased_predictions.csv"))
        candidates = cleaned or uncleaned
        if candidates:
            prediction_paths.append(candidates[0])

    return prediction_paths


def prediction_csvs_from_inputs(inputs, data_dir):
    if not inputs:
        return discover_prediction_csvs(data_dir)

    paths = []
    for input_value in inputs:
        candidate = input_value
        if not candidate.exists():
            candidate = data_dir / str(input_value)

        if candidate.is_dir():
            cleaned = sorted(
                candidate.glob("*_cleaned_detoxify_unbiased_predictions.csv")
            )
            uncleaned = sorted(candidate.glob("*_detoxify_unbiased_predictions.csv"))
            if not cleaned and not uncleaned:
                raise SystemExit(f"No prediction CSV found in {candidate}")
            paths.append((cleaned or uncleaned)[0])
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise SystemExit(f"Input does not exist: {input_value}")

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def infer_subreddit(input_csv, data_dir):
    try:
        relative_input = input_csv.resolve().relative_to(data_dir.resolve())
    except ValueError:
        return input_csv.parent.name or input_csv.stem

    if relative_input.parts:
        return relative_input.parts[0]
    return input_csv.parent.name or input_csv.stem


def first_existing(columns, candidates):
    for column in candidates:
        if column in columns:
            return column
    return None


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


def parse_datetime(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.8)):
        return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def clean_usernames(series, include_deleted):
    usernames = series.fillna("").astype(str).str.strip()
    valid = usernames.ne("")
    if not include_deleted:
        valid &= ~usernames.str.lower().isin(DELETED_USERNAMES)
    return usernames, valid


def safe_percent(numerator, denominator):
    if denominator is None or denominator == 0:
        return np.nan
    return numerator / denominator * 100


def safe_ratio(numerator, denominator):
    if denominator is None or denominator == 0:
        return np.nan
    return numerator / denominator


def summarize_users(df, username_column, score_column, threshold, troll_threshold, top_n):
    users = (
        df.groupby(username_column)
        .agg(
            comment_count=(score_column, "size"),
            toxic_comment_count=("above_threshold", "sum"),
            average_toxicity=(score_column, "mean"),
        )
        .reset_index()
    )
    total_users = len(users)
    troll_users = int((users["average_toxicity"] > troll_threshold).sum())
    percent_troll_users = safe_percent(troll_users, total_users)

    total_toxic_comments = int(users["toxic_comment_count"].sum())
    ranked_toxic = users.sort_values(
        ["toxic_comment_count", "average_toxicity", "comment_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    concentration = {}
    for percent in (1, 5, 10):
        count = int(math.ceil(total_users * percent / 100)) if total_users else 0
        count = max(1, count) if total_users else 0
        toxic_count = int(ranked_toxic.head(count)["toxic_comment_count"].sum())
        concentration[f"top_{percent}_percent_users_share_of_toxic_comments"] = (
            safe_percent(toxic_count, total_toxic_comments)
        )

    ranked_active = users.sort_values(
        ["comment_count", "toxic_comment_count", "average_toxicity"],
        ascending=[False, False, False],
    ).head(top_n)
    top_active_comments = int(ranked_active["comment_count"].sum())
    top_active_toxic_comments = int(ranked_active["toxic_comment_count"].sum())
    percent_toxic_top_active_users = safe_percent(
        top_active_toxic_comments,
        top_active_comments,
    )

    return {
        "total_users": total_users,
        "troll_users": troll_users,
        "percent_troll_users": percent_troll_users,
        "top_active_users": min(top_n, total_users),
        "top_active_comments": top_active_comments,
        "top_active_toxic_comments": top_active_toxic_comments,
        "percent_toxic_comments_top_active_users": percent_toxic_top_active_users,
        **concentration,
    }


def summarize_temporal(df, date_column, score_column):
    if not date_column:
        return {
            "months_observed": np.nan,
            "monthly_toxicity_volatility": np.nan,
            "max_monthly_percent_toxic_comments": np.nan,
        }

    dates = parse_datetime(df[date_column]).dt.tz_localize(None)
    temporal = pd.DataFrame(
        {
            "month": dates.dt.to_period("M").astype(str),
            "above_threshold": df["above_threshold"],
            score_column: df[score_column],
        }
    ).dropna(subset=["month"])
    temporal = temporal[temporal["month"] != "NaT"].copy()
    if temporal.empty:
        return {
            "months_observed": np.nan,
            "monthly_toxicity_volatility": np.nan,
            "max_monthly_percent_toxic_comments": np.nan,
        }

    monthly = (
        temporal.groupby("month")
        .agg(
            comments=(score_column, "size"),
            toxic_comments=("above_threshold", "sum"),
        )
        .reset_index()
    )
    monthly["percent_toxic_comments"] = (
        monthly["toxic_comments"] / monthly["comments"] * 100
    )
    return {
        "months_observed": len(monthly),
        "monthly_toxicity_volatility": float(
            monthly["percent_toxic_comments"].std(ddof=0)
        ),
        "max_monthly_percent_toxic_comments": float(
            monthly["percent_toxic_comments"].max()
        ),
    }


def summarize_reply_dynamics(
    df,
    comment_id_column,
    parent_id_column,
    score_column,
    threshold,
    dogpile_min_toxic_replies,
):
    empty = {
        "matched_direct_replies": np.nan,
        "percent_replies_toxic_to_nontoxic_parent": np.nan,
        "percent_replies_toxic_to_toxic_parent": np.nan,
        "toxic_reply_contagion_ratio": np.nan,
        "average_direct_replies_nontoxic_parent": np.nan,
        "average_direct_replies_toxic_parent": np.nan,
        "toxic_engagement_ratio": np.nan,
        "dogpiling_parent_count": np.nan,
        "dogpiling_percent_of_all_parent_comments": np.nan,
        "dogpiling_percent_of_replied_parent_comments": np.nan,
        "percent_dogpiling_parent_comments_toxic": np.nan,
    }
    if not comment_id_column or not parent_id_column:
        return empty

    relations = df[[comment_id_column, parent_id_column, score_column]].copy()
    relations["normalized_comment_id"] = relations[comment_id_column].map(
        normalize_comment_id
    )
    relations["normalized_parent_id"] = relations[parent_id_column].map(
        normalize_parent_id
    )
    relations = relations[relations["normalized_comment_id"] != ""].copy()
    if relations.empty:
        return empty

    parents = (
        relations[["normalized_comment_id", score_column]]
        .drop_duplicates("normalized_comment_id", keep="first")
        .rename(
            columns={
                "normalized_comment_id": "normalized_parent_id",
                score_column: "parent_toxicity",
            }
        )
    )
    children = relations[
        ["normalized_comment_id", "normalized_parent_id", score_column]
    ].rename(columns={score_column: "child_toxicity"})
    children = children[children["normalized_parent_id"] != ""].copy()
    pairs = children.merge(parents, on="normalized_parent_id", how="inner")
    if pairs.empty:
        return empty

    pairs["parent_above_threshold"] = pairs["parent_toxicity"] > threshold
    pairs["child_above_threshold"] = pairs["child_toxicity"] > threshold

    reply_summary = (
        pairs.groupby("parent_above_threshold")
        .agg(
            matched_replies=("child_toxicity", "size"),
            toxic_replies=("child_above_threshold", "sum"),
        )
        .reset_index()
    )
    rates = {False: np.nan, True: np.nan}
    for row in reply_summary.itertuples(index=False):
        rates[bool(row.parent_above_threshold)] = safe_percent(
            row.toxic_replies,
            row.matched_replies,
        )

    reply_counts = (
        children.groupby("normalized_parent_id")
        .agg(total_direct_replies=("normalized_comment_id", "size"))
        .reset_index()
    )
    parent_engagement = parents.merge(reply_counts, on="normalized_parent_id", how="left")
    parent_engagement["total_direct_replies"] = (
        parent_engagement["total_direct_replies"].fillna(0).astype(int)
    )
    parent_engagement["parent_above_threshold"] = (
        parent_engagement["parent_toxicity"] > threshold
    )
    engagement = parent_engagement.groupby("parent_above_threshold")[
        "total_direct_replies"
    ].mean()
    nontoxic_engagement = float(engagement.get(False, np.nan))
    toxic_engagement = float(engagement.get(True, np.nan))

    toxic_reply_counts = (
        pairs.groupby("normalized_parent_id")
        .agg(
            toxic_direct_replies=("child_above_threshold", "sum"),
            total_direct_replies=("child_toxicity", "size"),
            parent_above_threshold=("parent_above_threshold", "first"),
        )
        .reset_index()
    )
    dogpiling_parents = toxic_reply_counts[
        toxic_reply_counts["toxic_direct_replies"] >= dogpile_min_toxic_replies
    ].copy()
    dogpiling_parent_count = len(dogpiling_parents)
    toxic_dogpiling_parent_count = int(
        dogpiling_parents["parent_above_threshold"].sum()
    )
    replied_parent_comments = len(toxic_reply_counts)
    all_parent_comments = len(parents)

    return {
        "matched_direct_replies": len(pairs),
        "percent_replies_toxic_to_nontoxic_parent": rates[False],
        "percent_replies_toxic_to_toxic_parent": rates[True],
        "toxic_reply_contagion_ratio": safe_ratio(rates[True], rates[False]),
        "average_direct_replies_nontoxic_parent": nontoxic_engagement,
        "average_direct_replies_toxic_parent": toxic_engagement,
        "toxic_engagement_ratio": safe_ratio(toxic_engagement, nontoxic_engagement),
        "dogpiling_parent_count": dogpiling_parent_count,
        "dogpiling_percent_of_all_parent_comments": safe_percent(
            dogpiling_parent_count,
            all_parent_comments,
        ),
        "dogpiling_percent_of_replied_parent_comments": safe_percent(
            dogpiling_parent_count,
            replied_parent_comments,
        ),
        "percent_dogpiling_parent_comments_toxic": safe_percent(
            toxic_dogpiling_parent_count,
            dogpiling_parent_count,
        ),
    }


def summarize_community(input_csv, args):
    columns = set(pd.read_csv(input_csv, nrows=0).columns)
    if args.score_column not in columns:
        raise SystemExit(f"{input_csv} is missing score column: {args.score_column}")
    if args.username_column not in columns:
        raise SystemExit(f"{input_csv} is missing username column: {args.username_column}")

    score_columns = [column for column in DEFAULT_SCORE_COLUMNS if column in columns]
    comment_id_column = first_existing(columns, COMMENT_ID_COLUMNS)
    date_column = first_existing(columns, DATE_COLUMNS)
    optional_columns = [
        args.username_column,
        args.parent_id_column,
        comment_id_column,
        date_column,
        *score_columns,
    ]
    usecols = sorted({column for column in optional_columns if column in columns})

    df = pd.read_csv(input_csv, usecols=usecols)
    for column in score_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[args.score_column]).copy()

    usernames, valid_users = clean_usernames(
        df[args.username_column],
        args.include_deleted,
    )
    df[args.username_column] = usernames
    user_df = df[valid_users].copy()
    if user_df.empty:
        raise SystemExit(f"No valid users found in {input_csv}")

    df["above_threshold"] = df[args.score_column] > args.threshold
    user_df["above_threshold"] = user_df[args.score_column] > args.threshold

    total_comments = len(df)
    toxic_comments = int(df["above_threshold"].sum())
    summary = {
        "subreddit": infer_subreddit(input_csv, args.data_dir),
        "input_csv": str(input_csv),
        "threshold": args.threshold,
        "troll_threshold": args.troll_threshold,
        "total_comments": total_comments,
        "toxic_comments": toxic_comments,
        "percent_toxic_comments": safe_percent(toxic_comments, total_comments),
        "mean_toxicity": float(df[args.score_column].mean()),
        "median_toxicity": float(df[args.score_column].median()),
    }

    for column in score_columns:
        above = int((df[column] > args.threshold).sum())
        summary[f"percent_{column}_above_threshold"] = safe_percent(
            above,
            df[column].notna().sum(),
        )

    summary.update(
        summarize_users(
            user_df,
            args.username_column,
            args.score_column,
            args.threshold,
            args.troll_threshold,
            args.top_active_users,
        )
    )
    summary.update(summarize_temporal(df, date_column, args.score_column))
    summary.update(
        summarize_reply_dynamics(
            df,
            comment_id_column,
            args.parent_id_column if args.parent_id_column in columns else None,
            args.score_column,
            args.threshold,
            args.dogpile_min_toxic_replies,
        )
    )
    return summary


def sorted_plot_data(summary, metric):
    plot_data = summary.dropna(subset=[metric]).copy()
    return plot_data.sort_values(metric, ascending=True)


def plot_horizontal_bars(summary, metric, output_png, title, xlabel, color="#4C78A8"):
    plot_data = sorted_plot_data(summary, metric)
    if plot_data.empty:
        return False

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    labels = plot_data["subreddit"].astype(str).tolist()
    values = plot_data[metric].to_numpy(dtype=float)
    fig_height = max(5, 0.48 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(labels, values, color=color, edgecolor="white")
    max_value = values.max() if len(values) else 0
    label_offset = max(max_value * 0.015, 0.03)
    for bar, value in zip(bars, values):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Subreddit")
    ax.set_xlim(0, max_value + label_offset * 9 if max_value else 1)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return True


def plot_grouped_bars(
    summary,
    metrics,
    output_png,
    title,
    xlabel,
    ylabel="Subreddit",
    legend_labels=None,
):
    plot_metrics = [metric for metric in metrics if metric in summary.columns]
    plot_data = summary.dropna(subset=plot_metrics, how="all").copy()
    if plot_data.empty or not plot_metrics:
        return False

    sort_metric = plot_metrics[0]
    plot_data = plot_data.sort_values(sort_metric, ascending=True)

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    subreddit_labels = plot_data["subreddit"].astype(str).tolist()
    y = np.arange(len(plot_data))
    height = min(0.22, 0.8 / max(1, len(plot_metrics)))
    offsets = (np.arange(len(plot_metrics)) - (len(plot_metrics) - 1) / 2) * height
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]

    fig_height = max(5.5, 0.52 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(13, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    for index, metric in enumerate(plot_metrics):
        values = plot_data[metric].to_numpy(dtype=float)
        ax.barh(
            y + offsets[index],
            values,
            height=height,
            color=colors[index % len(colors)],
            edgecolor="white",
            label=(
                legend_labels.get(metric)
                if legend_labels and metric in legend_labels
                else metric.replace("percent_", "").replace("_", " ")
            ),
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(y)
    ax.set_yticklabels(subreddit_labels)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return True


def plot_category_heatmap(summary, metrics, output_png, title):
    plot_metrics = [metric for metric in metrics if metric in summary.columns]
    plot_data = summary.dropna(subset=plot_metrics, how="all").copy()
    if plot_data.empty or not plot_metrics:
        return False

    plot_data = plot_data.sort_values("percent_toxic_comments", ascending=False)
    matrix = plot_data[plot_metrics].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig_height = max(5, 0.45 * len(plot_data) + 2)
    fig_width = max(9, 1.3 * len(plot_metrics) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    fig.patch.set_facecolor("white")
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto")

    ax.set_title(title, pad=12)
    ax.set_xticks(np.arange(len(plot_metrics)))
    ax.set_xticklabels(
        [
            metric.replace("percent_", "").replace("_above_threshold", "").replace("_", " ")
            for metric in plot_metrics
        ],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels(plot_data["subreddit"].astype(str).tolist())

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value < np.nanmax(matrix) * 0.55 else "white",
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent of comments above threshold")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return True


def plot_category_grouped_bars(summary, metrics, output_png, title):
    plot_metrics = [metric for metric in metrics if metric in summary.columns]
    plot_data = summary.dropna(subset=plot_metrics, how="all").copy()
    if plot_data.empty or not plot_metrics:
        return False

    plot_data = plot_data.sort_values("percent_toxic_comments", ascending=True)

    plt = configure_matplotlib()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    labels = plot_data["subreddit"].astype(str).tolist()
    y = np.arange(len(plot_data))
    height = min(0.12, 0.82 / max(1, len(plot_metrics)))
    offsets = (np.arange(len(plot_metrics)) - (len(plot_metrics) - 1) / 2) * height
    colors = [
        "#E45756",
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#B279A2",
        "#72B7B2",
        "#8C6D31",
    ]

    fig_height = max(6.5, 0.75 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(15, fig_height), dpi=160)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    max_value = 0
    for index, metric in enumerate(plot_metrics):
        values = plot_data[metric].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            max_value = max(max_value, float(finite_values.max()))
        label = (
            metric.replace("percent_", "")
            .replace("_above_threshold", "")
            .replace("_", " ")
        )
        ax.barh(
            y + offsets[index],
            values,
            height=height,
            color=colors[index % len(colors)],
            edgecolor="white",
            label=label,
        )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Percent of comments above threshold")
    ax.set_ylabel("Subreddit")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max_value * 1.15 if max_value else 1)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.8)
    ax.legend(frameon=True, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return True


def make_plots(summary, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    plot_specs = [
        (
            "percent_toxic_comments",
            "all_communities_percent_toxic_comments.png",
            "Percent Toxic Comments by Community",
            "Percent of comments above threshold",
            "#E45756",
        ),
        (
            "percent_troll_users",
            "all_communities_percent_troll_users.png",
            "Percent Troll Users by Community",
            "Percent of users above average toxicity cutoff",
            "#4C78A8",
        ),
        (
            "mean_toxicity",
            "all_communities_mean_toxicity.png",
            "Mean Toxicity Score by Community",
            "Mean toxicity score",
            "#54A24B",
        ),
        (
            "monthly_toxicity_volatility",
            "all_communities_monthly_toxicity_volatility.png",
            "Monthly Toxicity Volatility by Community",
            "Standard deviation of monthly percent toxic comments",
            "#B279A2",
        ),
        (
            "toxic_reply_contagion_ratio",
            "all_communities_toxic_reply_contagion_ratio.png",
            "Toxic Reply Contagion Ratio by Community",
            "Toxic-parent reply rate / non-toxic-parent reply rate",
            "#F58518",
        ),
        (
            "toxic_engagement_ratio",
            "all_communities_toxic_engagement_ratio.png",
            "Toxic Comment Engagement Ratio by Community",
            "Average replies to toxic parents / non-toxic parents",
            "#72B7B2",
        ),
        (
            "dogpiling_percent_of_replied_parent_comments",
            "all_communities_dogpiling_rate.png",
            "Dogpiling Rate by Community",
            "Percent of replied parent comments with multiple toxic replies",
            "#E45756",
        ),
    ]
    for metric, filename, title, xlabel, color in plot_specs:
        output = plot_dir / filename
        if plot_horizontal_bars(summary, metric, output, title, xlabel, color):
            outputs.append(output)

    dogpiling_metrics = [
        "dogpiling_percent_of_replied_parent_comments",
        "percent_dogpiling_parent_comments_toxic",
    ]
    output = plot_dir / "all_communities_dogpiling_rate_and_toxic_parents.png"
    if plot_grouped_bars(
        summary,
        dogpiling_metrics,
        output,
        "Dogpiling Rate and Toxic Dogpiled Parents by Community",
        "Percent",
        legend_labels={
            "dogpiling_percent_of_replied_parent_comments": "Dogpiling rate",
            "percent_dogpiling_parent_comments_toxic": "Dogpiled parents toxic",
        },
    ):
        outputs.append(output)

    concentration_metrics = [
        "top_1_percent_users_share_of_toxic_comments",
        "top_5_percent_users_share_of_toxic_comments",
        "top_10_percent_users_share_of_toxic_comments",
    ]
    output = plot_dir / "all_communities_toxicity_concentration.png"
    if plot_grouped_bars(
        summary,
        concentration_metrics,
        output,
        "Toxicity Concentration by Community",
        "Share of toxic comments produced by top-ranked toxic users",
        legend_labels={
            "top_1_percent_users_share_of_toxic_comments": "Top 1% users",
            "top_5_percent_users_share_of_toxic_comments": "Top 5% users",
            "top_10_percent_users_share_of_toxic_comments": "Top 10% users",
        },
    ):
        outputs.append(output)

    contagion_metrics = [
        "percent_replies_toxic_to_nontoxic_parent",
        "percent_replies_toxic_to_toxic_parent",
    ]
    output = plot_dir / "all_communities_toxic_reply_contagion_by_parent_type.png"
    if plot_grouped_bars(
        summary,
        contagion_metrics,
        output,
        "Toxic Reply Contagion by Parent Comment Type",
        "Percent of direct replies above toxicity threshold",
        legend_labels={
            "percent_replies_toxic_to_nontoxic_parent": "Negative/non-toxic parent",
            "percent_replies_toxic_to_toxic_parent": "Positive/toxic parent",
        },
    ):
        outputs.append(output)

    category_metrics = [
        f"percent_{column}_above_threshold"
        for column in DEFAULT_SCORE_COLUMNS
        if f"percent_{column}_above_threshold" in summary.columns
    ]
    output = plot_dir / "all_communities_toxicity_category_rates.png"
    if plot_category_heatmap(
        summary,
        category_metrics,
        output,
        "Detoxify Category Rates by Community",
    ):
        outputs.append(output)

    output = plot_dir / "all_communities_toxicity_category_rates_bars.png"
    if plot_category_grouped_bars(
        summary,
        category_metrics,
        output,
        "Detoxify Category Rates by Community",
    ):
        outputs.append(output)

    top_active_metrics = [
        "percent_toxic_comments",
        "percent_toxic_comments_top_active_users",
    ]
    output = plot_dir / "all_communities_top_active_vs_all_percent_toxic.png"
    if plot_grouped_bars(
        summary,
        top_active_metrics,
        output,
        "Top Active Users vs All Comments",
        "Percent toxic comments",
        legend_labels={
            "percent_toxic_comments": "All comments",
            "percent_toxic_comments_top_active_users": "Top active users",
        },
    ):
        outputs.append(output)

    return outputs


def main():
    args = parse_args()
    if args.threshold < 0 or args.threshold > 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if args.troll_threshold < 0 or args.troll_threshold > 1:
        raise SystemExit("--troll-threshold must be between 0 and 1")
    if args.top_active_users < 1:
        raise SystemExit("--top-active-users must be at least 1")
    if args.dogpile_min_toxic_replies < 1:
        raise SystemExit("--dogpile-min-toxic-replies must be at least 1")

    prediction_csvs = prediction_csvs_from_inputs(args.inputs, args.data_dir)
    if not prediction_csvs:
        raise SystemExit(f"No prediction CSVs found under {args.data_dir}")

    rows = []
    for input_csv in prediction_csvs:
        print(f"Summarizing {input_csv}")
        rows.append(summarize_community(input_csv, args))

    summary = pd.DataFrame(rows).sort_values("subreddit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Saved {args.output}")

    outputs = make_plots(summary, args.plot_dir)
    for output in outputs:
        print(f"Saved {output}")

    print(f"Communities summarized: {len(summary):,}")


if __name__ == "__main__":
    main()
