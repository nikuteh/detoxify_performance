#!/bin/bash
#SBATCH --job-name=community_processing_test
#SBATCH --output=community_processing_test_results%j.log
#SBATCH --error=community_processing_test_error%j.log

#SBATCH --partition=gpuq
#SBATCH  --mem=16G
#SBATCH  --cpus-per-task=4
#SBATCH  --time=02:00:00

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <Subreddit>"
  exit 1
fi

SUBREDDIT="$1"
SUBREDDIT_SLUG="$(printf '%s' "$SUBREDDIT" | tr '[:upper:]' '[:lower:]')"
COMMENTS_BASE="${SUBREDDIT_SLUG}_comments"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wwalsh/detoxify_performance}"
if [ ! -d "${PROJECT_ROOT}/scripts" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "$PROJECT_ROOT"

DATA_DIR="data/subreddits/${SUBREDDIT}"
VIS_DIR="visualizations/subreddits/${SUBREDDIT}"
PREDICTIONS_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned_detoxify_unbiased_predictions.csv"

if [ ! -f "$PREDICTIONS_CSV" ]; then
  echo "Missing predictions CSV: $PREDICTIONS_CSV"
  echo "Run: sbatch scripts/run_detoxify_on_csv.sh ${DATA_DIR}/${COMMENTS_BASE}.zst"
  exit 1
fi

mkdir -p "$DATA_DIR" "$VIS_DIR"

THRESHOLD="${TOXICITY_THRESHOLD:-0.5}"

echo "Running community processing test analyses for ${SUBREDDIT}"
echo "Input: ${PREDICTIONS_CSV}"
echo "Outputs: ${DATA_DIR} and ${VIS_DIR}"
echo "Toxicity threshold: ${THRESHOLD}"

# Baseline toxicity over time.
python -u scripts/community/visualization/plot_toxicity_over_time.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_toxicity_over_time.png" \
  --title "${SUBREDDIT} Toxicity Over Time"

# All users: percent toxic by each user's chronological post/comment number.
python -u scripts/community/visualization/plot_all_users_toxicity_by_post_number.py \
  "$PREDICTIONS_CSV" \
  --average-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_average_toxicity_by_post_number.png" \
  --percent-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_percent_toxic_by_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_all_users_toxicity_by_post_number.csv" \
  --threshold "$THRESHOLD"

# All users: percent toxic over elapsed user time.
python -u scripts/community/visualization/plot_all_users_toxicity_over_user_time.py \
  "$PREDICTIONS_CSV" \
  --average-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_average_toxicity_over_user_time.png" \
  --percent-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_percent_toxic_over_user_time.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_all_users_toxicity_over_user_time.csv" \
  --threshold "$THRESHOLD"

# Comment vs response toxicity by parent post number.
python -u scripts/community/visualization/cultural_violence_test_4.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_comment_vs_response_toxicity_by_parent_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_comment_vs_response_toxicity_by_parent_post_number.csv" \
  --title "${SUBREDDIT} Comment vs Response Toxicity by Parent Post Number"

# Idea 1: toxicity spike timeline.
python -u scripts/community/visualization/plot_toxicity_over_time.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_1_toxicity_spike_timeline.png" \
  --title "${SUBREDDIT} Toxicity Spike Timeline" \
  --time-bin "${SPIKE_TIME_BIN:-MS}"

# Idea 3: thread escalation by observed reply-chain depth.
python -u scripts/community/visualization/plot_thread_escalation.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_3_thread_escalation.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_3_thread_escalation.csv" \
  --title "${SUBREDDIT} Thread Escalation by Reply Depth" \
  --threshold "$THRESHOLD" \
  --max-depth "${THREAD_ESCALATION_MAX_DEPTH:-20}" \
  --min-comments-per-depth "${THREAD_ESCALATION_MIN_COMMENTS:-1}"

# Idea 6: dogpiling candidates.
python -u scripts/community/visualization/plot_dogpiling.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_6_dogpiling_candidates.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_6_dogpiling_candidates.csv" \
  --title "${SUBREDDIT} Dogpiling Candidates" \
  --threshold "$THRESHOLD" \
  --min-toxic-replies "${DOGPILING_MIN_TOXIC_REPLIES:-2}" \
  --top-n "${DOGPILING_TOP_N:-20}"

# Idea 13: established/high-volume user authority test.
python -u scripts/community/visualization/plot_user_authority_toxicity.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_13_user_authority_toxicity.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_13_user_authority_toxicity.csv" \
  --user-output "${DATA_DIR}/${COMMENTS_BASE}_idea_13_user_authority_users.csv" \
  --title "${SUBREDDIT} Toxicity by User Activity" \
  --threshold "$THRESHOLD"

echo "Community processing test analyses complete for ${SUBREDDIT}"
