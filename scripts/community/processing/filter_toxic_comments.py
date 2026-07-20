import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SUBREDDITS_DIR = PROJECT_ROOT / "data" / "subreddits"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write all comments above a Detoxify toxicity threshold to CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV containing Detoxify prediction columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Default: inferred from the input CSV filename.",
    )
    parser.add_argument(
        "--score-column",
        default="toxicity",
        help="Score column to filter on. Default: toxicity.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Keep comments with score greater than this value. Default: 0.5.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows to read at a time. Default: 100000.",
    )
    return parser.parse_args()


def format_threshold_for_filename(threshold):
    return f"{threshold:g}".replace(".", "_")


def infer_output_path(input_csv, score_column, threshold):
    threshold_label = format_threshold_for_filename(threshold)
    filename = f"{input_csv.stem}_{score_column}_above_{threshold_label}.csv"

    try:
        input_csv.resolve().relative_to(DATA_SUBREDDITS_DIR)
    except ValueError:
        return input_csv.with_name(filename)

    return input_csv.with_name(filename)


def filter_comments(input_csv, output_csv, score_column, threshold, chunk_size):
    total_rows = 0
    valid_scores = 0
    kept_rows = 0
    wrote_header = False

    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        total_rows += len(chunk)
        scores = pd.to_numeric(chunk[score_column], errors="coerce")
        valid_scores += scores.notna().sum()

        filtered = chunk[scores > threshold].copy()
        if filtered.empty:
            continue

        filtered.to_csv(
            output_csv,
            mode="w" if not wrote_header else "a",
            index=False,
            header=not wrote_header,
        )
        wrote_header = True
        kept_rows += len(filtered)

    if not wrote_header:
        columns = pd.read_csv(input_csv, nrows=0).columns
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    return {
        "total_rows": total_rows,
        "valid_scores": int(valid_scores),
        "kept_rows": kept_rows,
    }


def main():
    args = parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    if args.score_column not in columns:
        raise SystemExit(
            f"{args.input_csv} is missing score column: {args.score_column}"
        )

    output_csv = args.output or infer_output_path(
        args.input_csv,
        args.score_column,
        args.threshold,
    )
    if output_csv.resolve() == args.input_csv.resolve():
        raise SystemExit("--output cannot be the same file as the input CSV")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    counts = filter_comments(
        args.input_csv,
        output_csv,
        args.score_column,
        args.threshold,
        args.chunk_size,
    )

    print(f"Saved {output_csv}")
    print(f"Total rows read: {counts['total_rows']:,}")
    print(f"Rows with valid {args.score_column}: {counts['valid_scores']:,}")
    print(
        f"Rows with {args.score_column} > {args.threshold:g}: "
        f"{counts['kept_rows']:,}"
    )


if __name__ == "__main__":
    main()
