import argparse
import csv
import io
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


CSV_COLUMNS = ["Subreddit", "username", "timestamp", "comment_text", "parent_id"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Reddit comments .zst dump into the same CSV format as "
            "the project files in data/subreddits/<Subreddit>/."
        )
    )
    parser.add_argument(
        "zst_file",
        type=Path,
        help="Input Reddit comments .zst file containing newline-delimited JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Default: input filename with .csv extension.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only convert this many comments. Useful for quick checks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress after this many input rows. Default: 100000.",
    )
    return parser.parse_args()


@contextmanager
def zst_text_reader(zst_file, allow_early_stop=False):
    try:
        import zstandard as zstd
    except ImportError:
        zstd = None

    if zstd is not None:
        with zst_file.open("rb") as compressed:
            reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(
                compressed
            )
            text_reader = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            try:
                yield text_reader
            finally:
                text_reader.close()
        return

    zstd_command = shutil.which("zstd")
    if zstd_command is None:
        raise SystemExit(
            "Install the Python package 'zstandard' or the command-line tool "
            "'zstd' to read .zst files."
        )

    process = subprocess.Popen(
        [zstd_command, "-dc", str(zst_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        yield process.stdout
    finally:
        if process.stdout is not None:
            process.stdout.close()
        _, stderr = process.communicate()
        if process.returncode not in (0, None):
            early_stop_returncodes = {1, 70, 141, -13}
            early_stop_messages = ("Broken pipe", "Write error")
            if allow_early_stop and (
                process.returncode in early_stop_returncodes
                or any(message in stderr for message in early_stop_messages)
                or stderr.strip() == "zstd:"
            ):
                return
            raise SystemExit(
                stderr.strip() or f"zstd exited with {process.returncode}"
            )


def clean_comment_text(value):
    if value is None:
        return ""
    return str(value).replace("\r", "").replace("\n", "")


def format_timestamp(value):
    if value is None:
        return ""
    return str(value)


def comment_to_row(comment):
    return {
        "Subreddit": comment.get("subreddit", ""),
        "username": comment.get("author", ""),
        "timestamp": format_timestamp(comment.get("created_utc", "")),
        "comment_text": clean_comment_text(comment.get("body", "")),
        "parent_id": comment.get("parent_id", ""),
    }


def main():
    args = parse_args()

    if not args.zst_file.is_file():
        raise SystemExit(f"Input file does not exist: {args.zst_file}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be at least 1")

    output_csv = args.output or args.zst_file.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    written_rows = 0
    skipped_rows = 0

    with zst_text_reader(
        args.zst_file,
        allow_early_stop=args.limit is not None,
    ) as lines, output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for line in lines:
            if args.limit is not None and written_rows >= args.limit:
                break

            input_rows += 1
            if not line.strip():
                continue

            try:
                comment = json.loads(line)
            except json.JSONDecodeError:
                skipped_rows += 1
                continue

            writer.writerow(comment_to_row(comment))
            written_rows += 1

            if input_rows % args.progress_every == 0:
                print(f"Read {input_rows:,} rows; wrote {written_rows:,} comments")

    print(f"Saved {output_csv}")
    print(f"Input rows read: {input_rows:,}")
    print(f"Comments written: {written_rows:,}")
    if skipped_rows:
        print(f"Malformed rows skipped: {skipped_rows:,}")


if __name__ == "__main__":
    main()
