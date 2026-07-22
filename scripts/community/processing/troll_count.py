import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count users whose average toxicity across all scored comments is "
            "above a threshold."
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
        help="Optional output CSV for the summary row.",
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
        "score_column": score_column,
        "threshold": threshold,
        "total_users": total_users,
        "users_above_threshold": users_above_threshold,
        "percent_users_above_threshold": percent_above_threshold,
    }


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

    display = pd.DataFrame([summary])
    display["percent_users_above_threshold"] = display[
        "percent_users_above_threshold"
    ].round(4)
    print(display.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.output, index=False)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
