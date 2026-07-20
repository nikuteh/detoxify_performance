import argparse
import re
from pathlib import Path

import pandas as pd


URL_PATTERN = re.compile(r"https?://[^\s<>)]+|(?<!\w)www\.[^\s<>)]+", re.IGNORECASE)
DELETED_USERNAMES = {"[deleted]", "[removed]", "deleted", "removed"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean a subreddit CSV by dropping deleted users, removing URLs "
            "from comment text, and adding each user's chronological "
            "post_number."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV containing username and timestamp columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path. Default: input filename with "
            "_cleaned appended."
        ),
    )
    parser.add_argument(
        "--post-number-column",
        default="post_number",
        help="Name of the column to add. Default: post_number.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--text-column",
        default="comment_text",
        help="Column containing comment text to clean. Default: comment_text.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing Unix timestamps. Default: timestamp.",
    )
    parser.add_argument(
        "--overwrite-column",
        action="store_true",
        help="Replace an existing post-number column if it already exists.",
    )
    return parser.parse_args()


def resolve_output_path(input_csv, output):
    if output is not None:
        return output

    return input_csv.with_name(f"{input_csv.stem}_cleaned.csv")


def remove_urls(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = URL_PATTERN.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_comments(
    df,
    username_column,
    timestamp_column,
    text_column,
    post_number_column,
):
    source_order_column = "__source_order"
    clean_username_column = "__clean_username"
    timestamp_numeric_column = "__timestamp_numeric"
    original_rows = len(df)

    df = df.copy()
    df[source_order_column] = range(len(df))
    df[clean_username_column] = (
        df[username_column].fillna("").astype(str).str.strip()
    )
    df[timestamp_numeric_column] = pd.to_numeric(
        df[timestamp_column],
        errors="coerce",
    )

    keep_rows = (
        (df[clean_username_column] != "")
        & df[timestamp_numeric_column].notna()
        & ~df[clean_username_column].str.lower().isin(DELETED_USERNAMES)
    )

    dropped_rows = original_rows - int(keep_rows.sum())
    df = df.loc[keep_rows].copy()
    df[text_column] = df[text_column].map(remove_urls)

    df[post_number_column] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ordered_index = df.sort_values(
        [clean_username_column, timestamp_numeric_column, source_order_column]
    ).index

    df.loc[ordered_index, post_number_column] = (
        df.loc[ordered_index].groupby(clean_username_column).cumcount() + 1
    )

    numbered_rows = int(df[post_number_column].notna().sum())
    df = df.drop(
        columns=[
            source_order_column,
            clean_username_column,
            timestamp_numeric_column,
        ]
    )

    return df, numbered_rows, dropped_rows


def main():
    args = parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")

    output_csv = resolve_output_path(args.input_csv, args.output)
    if output_csv.resolve() == args.input_csv.resolve():
        raise SystemExit("--output cannot be the same file as the input CSV")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    required = {args.username_column, args.timestamp_column, args.text_column}
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(
            f"{args.input_csv} is missing required column(s): {', '.join(missing)}"
        )
    if args.post_number_column in columns and not args.overwrite_column:
        raise SystemExit(
            f"{args.input_csv} already has a {args.post_number_column!r} "
            "column. Pass --overwrite-column to replace it."
        )

    df = pd.read_csv(args.input_csv)
    output_df, numbered_rows, dropped_rows = clean_comments(
        df,
        args.username_column,
        args.timestamp_column,
        args.text_column,
        args.post_number_column,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print(f"Saved {output_csv}")
    print(f"Rows read: {len(df):,}")
    print(f"Rows dropped: {dropped_rows:,}")
    print(f"Rows written: {len(output_df):,}")
    print(f"Rows assigned {args.post_number_column}: {numbered_rows:,}")


if __name__ == "__main__":
    main()
