import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Detoxify on a comments CSV and append prediction columns."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV containing a comment text column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path. Default: input filename with "
            "_detoxify_unbiased_predictions.csv appended."
        ),
    )
    parser.add_argument(
        "--text-column",
        default="comment_text",
        help="Column containing comment text. Default: comment_text.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Timestamp column used to add a date column. Default: timestamp.",
    )
    parser.add_argument(
        "--model",
        default="unbiased",
        help="Detoxify model name. Default: unbiased.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("auto", "cpu", "cuda"),
        help="Device for Detoxify. Default: auto.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of comments per Detoxify batch. Default: 64.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of CSV rows to read and write at a time. Default: 5000.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only score this many rows. Useful for quick tests.",
    )
    return parser.parse_args()


def resolve_output_path(input_csv, output):
    if output is not None:
        return output

    return input_csv.with_name(
        f"{input_csv.stem}_detoxify_unbiased_predictions.csv"
    )


def resolve_device(device):
    if device != "auto":
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def predict_texts(model, texts, batch_size):
    predictions = None

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_predictions = model.predict(batch)

        if predictions is None:
            predictions = {column: [] for column in batch_predictions}

        for column, values in batch_predictions.items():
            predictions[column].extend(list(values))

    return predictions or {}


def add_date_column(df, timestamp_column):
    if timestamp_column not in df.columns or "date" in df.columns:
        return df

    timestamps = pd.to_numeric(df[timestamp_column], errors="coerce")
    df["date"] = pd.to_datetime(timestamps, unit="s", utc=True)
    return df


def main():
    args = parse_args()

    if not args.input_csv.is_file():
        raise SystemExit(f"Input file does not exist: {args.input_csv}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    output_csv = resolve_output_path(args.input_csv, args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = set(pd.read_csv(args.input_csv, nrows=0).columns)
    if args.text_column not in columns:
        raise SystemExit(
            f"{args.input_csv} is missing text column: {args.text_column}"
        )

    from detoxify import Detoxify

    device = resolve_device(args.device)
    print(f"Loading Detoxify model: {args.model} on {device}")
    model = Detoxify(args.model, device=device)

    reader = pd.read_csv(
        args.input_csv,
        chunksize=args.chunk_size,
        nrows=args.limit,
    )

    rows_scored = 0
    wrote_header = False

    for chunk in reader:
        texts = chunk[args.text_column].fillna("").astype(str).tolist()
        predictions = predict_texts(model, texts, args.batch_size)

        for column, values in predictions.items():
            chunk[column] = values

        chunk = add_date_column(chunk, args.timestamp_column)

        chunk.to_csv(
            output_csv,
            mode="w" if not wrote_header else "a",
            index=False,
            header=not wrote_header,
        )
        wrote_header = True
        rows_scored += len(chunk)
        print(f"Scored {rows_scored:,} rows")

    print(f"Saved {output_csv}")
    print(f"Rows scored: {rows_scored:,}")


if __name__ == "__main__":
    main()
