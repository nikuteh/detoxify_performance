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

# Idea 1: toxicity spike timeline.
python -u scripts/community/visualization/plot_toxicity_over_time.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_1_toxicity_spike_timeline.png" \
  --title "${SUBREDDIT} Toxicity Spike Timeline" \
  --time-bin "${SPIKE_TIME_BIN:-MS}"

# Toxic comment volume by month.
python -u scripts/community/visualization/plot_toxic_comment_volume_over_time.py \
  "$PREDICTIONS_CSV" \
  --bar-output "${VIS_DIR}/${COMMENTS_BASE}_toxic_comment_volume_by_month_bar.png" \
  --line-output "${VIS_DIR}/${COMMENTS_BASE}_toxic_comment_volume_by_month_line.png" \
  --percent-line-output "${VIS_DIR}/${COMMENTS_BASE}_toxic_comment_percent_by_month_line.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_toxic_comment_volume_by_month.csv" \
  --bar-title "${SUBREDDIT} Toxic Comment Volume by Month" \
  --line-title "${SUBREDDIT} Toxic Comment Volume by Month" \
  --percent-line-title "${SUBREDDIT} Percent Toxic Comments by Month" \
  --threshold "$THRESHOLD" \
  --time-bin "${TOXIC_VOLUME_TIME_BIN:-MS}"

# Idea 3: thread escalation by observed reply-chain depth.
python -u scripts/community/visualization/plot_thread_escalation.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_3_thread_escalation.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_3_thread_escalation.csv" \
  --title "${SUBREDDIT} Thread Escalation by Reply Depth" \
  --threshold "$THRESHOLD" \
  --max-depth "${THREAD_ESCALATION_MAX_DEPTH:-20}" \
  --min-comments-per-depth "${THREAD_ESCALATION_MIN_COMMENTS:-1}"

# Ideas 4 and 5: toxic reply contagion and parent-child toxicity deltas.
python -u scripts/community/visualization/plot_parent_child_toxicity.py \
  "$PREDICTIONS_CSV" \
  --contagion-output "${VIS_DIR}/${COMMENTS_BASE}_idea_4_toxic_reply_contagion.png" \
  --delta-output "${VIS_DIR}/${COMMENTS_BASE}_idea_5_parent_child_toxicity_delta.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_4_5_parent_child_toxicity.csv" \
  --title-prefix "${SUBREDDIT}" \
  --threshold "$THRESHOLD"

# Idea 6: dogpiling candidates.
python -u scripts/community/visualization/plot_dogpiling.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_6_dogpiling_candidates.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_6_dogpiling_candidates.csv" \
  --title "${SUBREDDIT} Dogpiling Candidates" \
  --threshold "$THRESHOLD" \
  --min-toxic-replies "${DOGPILING_MIN_TOXIC_REPLIES:-2}" \
  --top-n "${DOGPILING_TOP_N:-20}"

# Idea 7: whether toxic comments attract more direct replies.
python -u scripts/community/visualization/plot_toxicity_engagement.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_7_toxicity_engagement.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_7_toxicity_engagement.csv" \
  --title "${SUBREDDIT} Toxicity and Direct Reply Engagement" \
  --threshold "$THRESHOLD"

# Idea 11: user lifecycle toxicity over elapsed user time.
python -u scripts/community/visualization/plot_all_users_toxicity_over_user_time.py \
  "$PREDICTIONS_CSV" \
  --average-output "${VIS_DIR}/${COMMENTS_BASE}_idea_11_all_users_average_toxicity_over_user_time.png" \
  --percent-output "${VIS_DIR}/${COMMENTS_BASE}_idea_11_all_users_percent_toxic_over_user_time.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_11_all_users_toxicity_over_user_time.csv" \
  --top-active-average-output "${VIS_DIR}/${COMMENTS_BASE}_idea_11_top_100_active_users_average_toxicity_by_week.png" \
  --top-active-summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_11_top_100_active_users_average_toxicity_by_week.csv" \
  --top-active-average-title "${SUBREDDIT} Top 100 Active Users: Average Toxicity by Week" \
  --top-active-users "${USER_LIFECYCLE_TOP_ACTIVE_USERS:-100}" \
  --threshold "$THRESHOLD" \
  --time-unit "${USER_LIFECYCLE_TIME_UNIT:-week}" \
  --max-time-number "${USER_LIFECYCLE_MAX_TIME:-100}"

# Idea 13: established/high-volume user authority test.
python -u scripts/community/visualization/plot_user_authority_toxicity.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_13_user_authority_toxicity.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_13_user_authority_toxicity.csv" \
  --user-output "${DATA_DIR}/${COMMENTS_BASE}_idea_13_user_authority_users.csv" \
  --title "${SUBREDDIT} Toxicity by User Activity" \
  --threshold "$THRESHOLD"

# Idea 15: toxicity concentration among users.
python -u scripts/community/visualization/plot_toxicity_concentration.py \
  "$PREDICTIONS_CSV" \
  --curve-output "${VIS_DIR}/${COMMENTS_BASE}_idea_15_toxicity_concentration_curve.png" \
  --top-users-output "${VIS_DIR}/${COMMENTS_BASE}_idea_15_top_toxicity_contributors.png" \
  --top-percent-users-output "${VIS_DIR}/${COMMENTS_BASE}_idea_15_top_toxicity_contributors_by_percent.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_15_toxicity_concentration.csv" \
  --title-prefix "${SUBREDDIT}" \
  --threshold "$THRESHOLD" \
  --top-n "${CONCENTRATION_TOP_N:-20}" \
  --min-comments-for-percent "${CONCENTRATION_MIN_COMMENTS_FOR_PERCENT:-1}"

# Idea 16: average toxic-comment counts for top active users vs all users.
python -u scripts/community/visualization/plot_top_active_toxic_comment_average.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_vs_all_toxic_comment_average.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_16_top_active_vs_all_toxic_comment_average.csv" \
  --histogram-output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_toxic_comments_by_post_number.png" \
  --normalized-histogram-output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_toxic_comments_by_post_number_normalized.png" \
  --histogram-summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_16_top_active_toxic_comments_by_post_number.csv" \
  --title "${SUBREDDIT} Top Active Users vs All Users: Average Toxic Comment Count" \
  --threshold "$THRESHOLD" \
  --top-active-users "${TOP_ACTIVE_TOXIC_AVERAGE_USERS:-100}" \
  --max-post-number "${TOP_ACTIVE_TOXIC_POST_NUMBER_MAX:-100}"

echo "Community processing test analyses complete for ${SUBREDDIT}"
