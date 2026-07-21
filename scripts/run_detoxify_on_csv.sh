#!/bin/bash
#SBATCH --job-name=run_detoxify_on_csv
#SBATCH --output=detoxify_results%j.log
#SBATCH --error=detoxify_error%j.log

#SBATCH --partition=gpuq
#SBATCH  --gres=gpu:1 
#SBATCH  --mem=16G 
#SBATCH  --cpus-per-task=4 
#SBATCH  --time=02:00:00 

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 data/subreddits/<Subreddit>/<comments_dump>.zst"
  exit 1
fi

ZST_INPUT="$1"
if [ ! -f "$ZST_INPUT" ]; then
  echo "Input .zst file does not exist: $ZST_INPUT"
  exit 1
fi

case "$ZST_INPUT" in
  *.zst) ;;
  *)
    echo "Input file must end in .zst: $ZST_INPUT"
    exit 1
    ;;
esac

ZST_DIR="$(cd "$(dirname "$ZST_INPUT")" && pwd)"
ZST_FILE="${ZST_DIR}/$(basename "$ZST_INPUT")"
SUBREDDIT="$(basename "$ZST_DIR")"
SUBREDDIT_SLUG="$(printf '%s' "$SUBREDDIT" | tr '[:upper:]' '[:lower:]')"
COMMENTS_BASE="${SUBREDDIT_SLUG}_comments"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wwalsh/detoxify_performance}"
if [ ! -d "${PROJECT_ROOT}/scripts" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$PROJECT_ROOT"

DATA_DIR="data/subreddits/${SUBREDDIT}"
RAW_CSV="${DATA_DIR}/${COMMENTS_BASE}.csv"
CLEANED_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned.csv"
PREDICTIONS_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned_detoxify_unbiased_predictions.csv"
TOXIC_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned_toxicity_above_0_5.csv"

mkdir -p "$DATA_DIR"

python -u scripts/format_reddit_comments_zst.py \
  "$ZST_FILE" \
  --output "$RAW_CSV"

python -u scripts/community/processing/csv_cleaner.py \
  "$RAW_CSV" \
  --output "$CLEANED_CSV" \
  --drop-url-comments

python -u scripts/run_detoxify_on_csv.py \
  "$CLEANED_CSV" \
  --output "$PREDICTIONS_CSV"

python -u scripts/community/processing/filter_toxic_comments.py \
  "$PREDICTIONS_CSV" \
  --output "$TOXIC_CSV" \
  --threshold 0.5
