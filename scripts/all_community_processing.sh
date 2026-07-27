#!/bin/bash
#SBATCH --job-name=all_community_processing
#SBATCH --output=all_community_processing_results%j.log
#SBATCH --error=all_community_processing_error%j.log

#SBATCH --partition=gpuq
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wwalsh/detoxify_performance}"
if [ ! -d "${PROJECT_ROOT}/scripts" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$PROJECT_ROOT"

DATA_DIR="${ALL_COMMUNITY_DATA_DIR:-data/subreddits}"
OUTPUT_CSV="${ALL_COMMUNITY_OUTPUT_CSV:-data/subreddits/all_community_metrics.csv}"
PLOT_DIR="${ALL_COMMUNITY_PLOT_DIR:-visualizations/all_communities}"
THRESHOLD="${TOXICITY_THRESHOLD:-0.5}"
TROLL_THRESHOLD="${TROLL_THRESHOLD:-0.25}"
TOP_ACTIVE_USERS="${TOP_ACTIVE_USERS:-100}"
DOGPILE_MIN_TOXIC_REPLIES="${DOGPILE_MIN_TOXIC_REPLIES:-2}"

echo "Running all-community processing"
echo "Data directory: ${DATA_DIR}"
echo "Output CSV: ${OUTPUT_CSV}"
echo "Plot directory: ${PLOT_DIR}"
echo "Toxicity threshold: ${THRESHOLD}"
echo "Troll threshold: ${TROLL_THRESHOLD}"

python -u scripts/community/visualization/plot_all_community_metrics.py \
  "$@" \
  --data-dir "$DATA_DIR" \
  --output "$OUTPUT_CSV" \
  --plot-dir "$PLOT_DIR" \
  --threshold "$THRESHOLD" \
  --troll-threshold "$TROLL_THRESHOLD" \
  --top-active-users "$TOP_ACTIVE_USERS" \
  --dogpile-min-toxic-replies "$DOGPILE_MIN_TOXIC_REPLIES"

echo "All-community processing complete"
