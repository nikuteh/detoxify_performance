import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank users by highest average Detoxify toxicity score."
    )
    parser.add_argument(
        "user_csv_folder",
        type=Path,
        help="Folder containing one CSV file per user.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ranking CSV. Default: top_10_average_toxicity.csv in the input folder.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of users to include. Default: 10.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Score column to average. Default: toxicity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.user_csv_folder.is_dir():
        raise SystemExit(f"Folder does not exist: {args.user_csv_folder}")
    if args.top_n < 1:
        raise SystemExit("--top-n must be at least 1")

    output_csv = args.output or args.user_csv_folder / "top_10_average_toxicity.csv"
    rows = []

    for csv_path in sorted(args.user_csv_folder.glob("*.csv")):
        if csv_path.resolve() == output_csv.resolve():
            continue

        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except pd.errors.EmptyDataError:
            continue

        if args.score_column not in columns:
            continue

        usecols = [args.score_column]
        if "username" in columns:
            usecols.append("username")

        df = pd.read_csv(csv_path, usecols=usecols)
        scores = pd.to_numeric(df[args.score_column], errors="coerce").dropna()
        if scores.empty:
            continue

        if "username" in df.columns and df["username"].notna().any():
            username = str(df["username"].dropna().mode().iloc[0])
        else:
            username = csv_path.stem

        rows.append(
            {
                "rank": None,
                "username": username,
                "average_toxicity": scores.mean(),
                "comment_count": len(scores),
                "source_file": str(csv_path),
            }
        )

    if not rows:
        raise SystemExit(
            f"No CSVs with a usable {args.score_column!r} column found in "
            f"{args.user_csv_folder}"
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(["average_toxicity", "comment_count"], ascending=[False, False])
        .head(args.top_n)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    ranking["average_toxicity"] = ranking["average_toxicity"].round(6)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_csv, index=False)

    print(f"Saved {output_csv}")
    print(ranking[["rank", "username", "average_toxicity", "comment_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
