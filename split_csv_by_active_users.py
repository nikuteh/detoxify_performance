import argparse
import re
from pathlib import Path

import pandas as pd


INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create one CSV per user for users with at least a given number "
            "of comments in a subreddit CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV with a username column.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where per-user CSV files should be written.",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=100,
        help="Minimum number of comments required for a user CSV. Default: 100.",
    )
    parser.add_argument(
        "--username-column",
        default="username",
        help="Column containing usernames. Default: username.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include [deleted] and [removed] usernames if they meet the cutoff.",
    )
    return parser.parse_args()


def safe_filename(username):
    name = str(username).strip()
    name = INVALID_FILENAME_CHARS.sub("_", name)
    name = name.strip("._")
    return name or "unknown_user"


def main():
    args = parse_args()

    if args.min_comments < 1:
        raise SystemExit("--min-comments must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    if args.username_column not in columns:
        raise SystemExit(
            f"{args.input_csv} is missing username column: {args.username_column}"
        )

    df = pd.read_csv(args.input_csv)
    df[args.username_column] = df[args.username_column].fillna("").astype(str)
    df = df[df[args.username_column].str.strip() != ""]

    if not args.include_deleted:
        deleted_names = {"[deleted]", "[removed]"}
        df = df[~df[args.username_column].str.lower().isin(deleted_names)]

    comment_counts = df[args.username_column].value_counts()
    qualifying_users = comment_counts[comment_counts >= args.min_comments]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    used_filenames = set()
    written = 0
    for username, count in qualifying_users.items():
        filename_base = safe_filename(username)
        filename = f"{filename_base}.csv"
        suffix = 2
        while filename in used_filenames:
            filename = f"{filename_base}_{suffix}.csv"
            suffix += 1
        used_filenames.add(filename)

        user_rows = df[df[args.username_column] == username]
        output_path = args.output_dir / filename
        user_rows.to_csv(output_path, index=False)
        written += 1
        print(f"Wrote {output_path} ({count} comments)")

    print(f"Found {len(qualifying_users)} users with {args.min_comments}+ comments")
    print(f"Wrote {written} CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
