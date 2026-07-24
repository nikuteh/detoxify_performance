#!/bin/bash
#SBATCH --job-name=community_processing
#SBATCH --output=community_processing_results%j.log
#SBATCH --error=community_processing_error%j.log

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
USERS_DIR="${DATA_DIR}/users"
TROLLS_COMMUNITY_CSV="data/subreddits/trolls_community.csv"
TROLLS_COMMUNITY_PNG="visualizations/trolls_community_percent_trolls.png"

CLEANED_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned.csv"
PREDICTIONS_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned_detoxify_unbiased_predictions.csv"
TOXIC_CSV="${DATA_DIR}/${COMMENTS_BASE}_cleaned_toxicity_above_0_5.csv"

if [ ! -f "$PREDICTIONS_CSV" ]; then
  echo "Missing predictions CSV: $PREDICTIONS_CSV"
  echo "Run: sbatch scripts/run_detoxify_on_csv.sh ${DATA_DIR}/${COMMENTS_BASE}.zst"
  exit 1
fi

if [ ! -f "$CLEANED_CSV" ]; then
  echo "Missing cleaned comments CSV: $CLEANED_CSV"
  exit 1
fi

mkdir -p "$DATA_DIR" "$VIS_DIR" "$USERS_DIR"

THRESHOLD="${TOXICITY_THRESHOLD:-0.5}"

echo "Running community processing analyses for ${SUBREDDIT}"
echo "Input: ${PREDICTIONS_CSV}"
echo "Outputs: ${DATA_DIR} and ${VIS_DIR}"
echo "Toxicity threshold: ${THRESHOLD}"

# compute toxicity percentages for all comments in a csv
python -u scripts/community/processing/compute_toxicity_percentages.py \
  "$PREDICTIONS_CSV" \
  --output "${DATA_DIR}/${COMMENTS_BASE}_toxicity_percentages.csv" \
  --plot-output "${VIS_DIR}/${COMMENTS_BASE}_toxicity_percentages.png"

python -u scripts/community/processing/compute_toxicity_percentages.py \
  "$PREDICTIONS_CSV" \
  --threshold 0.8 \
  --output "${DATA_DIR}/${COMMENTS_BASE}_toxicity_percentages_threshold_0.8.csv" \
  --plot-output "${VIS_DIR}/${COMMENTS_BASE}_toxicity_percentages_threshold_0.8.png"

python -u scripts/community/processing/filter_toxic_comments.py \
  "$PREDICTIONS_CSV" \
  --output "$TOXIC_CSV" \
  --threshold 0.5

python -u scripts/community/processing/troll_count.py \
  "$PREDICTIONS_CSV" \
  --output "$TROLLS_COMMUNITY_CSV" \
  --plot-output "$TROLLS_COMMUNITY_PNG" \
  --subreddit "$SUBREDDIT" \
  --threshold "${TROLL_THRESHOLD:-0.25}"

python -u scripts/community/processing/split_csv_by_active_users.py \
  "$PREDICTIONS_CSV" \
  "$USERS_DIR" \
  --min-comments "${ACTIVE_USER_MIN_COMMENTS:-100}"

python -u scripts/community/processing/rank_users_by_average_toxicity.py \
  "$USERS_DIR" \
  --output "${USERS_DIR}/top_10_average_toxicity.csv" \
  --top-n 10

python -u scripts/community/visualization/plot_toxicity_over_time.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_toxicity_scatter_over_time.png" \
  --title "${SUBREDDIT} Toxicity Scatter Over Time" \
  --scatter-only \
  --dot-size 20 \
  --dot-alpha 0.7

python -u scripts/community/visualization/plot_posts_over_time.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_posts_over_time.png" \
  --title "${SUBREDDIT} Comments Over Time" \
  --ylabel "Comments per month" \
  --time-bin MS

python -u scripts/community/visualization/plot_top_toxic_users_average_toxicity.py \
  "$USERS_DIR" \
  --output "${VIS_DIR}/top_10_average_toxicity_per_post_scatter.png" \
  --top-n 10 \
  --min-users-per-post 1

python -u scripts/community/visualization/plot_top_users_toxicity_by_post_number.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_top_100_users_toxicity_by_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_top_100_users_toxicity_by_post_number.csv" \
  --title "${SUBREDDIT} Top 100 Users: Average Toxicity by Post Number" \
  --top-n 100 \
  --min-users-per-post 1 \
  --max-post-number "${TOP_USERS_POST_NUMBER_MAX:-250}"

python -u scripts/community/visualization/cultural_violence_test_2.py \
  "$TOXIC_CSV" \
  "$CLEANED_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_cultural_violence_parent_post_numbers.png" \
  --likelihood-output "${VIS_DIR}/${COMMENTS_BASE}_cultural_violence_toxic_responses_per_comment_percent_by_parent_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_cultural_violence_parent_post_numbers.csv" \
  --title "${SUBREDDIT} Cultural Violence Parent Post Numbers"

python -u scripts/community/visualization/cultural_violence_test_3.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_average_response_toxicity_by_parent_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_average_response_toxicity_by_parent_post_number.csv" \
  --title "${SUBREDDIT} Average Response Toxicity by Parent Post Number"

python -u scripts/community/visualization/plot_response_toxicity_by_user_post_number.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_response_toxicity_by_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_response_toxicity_by_post_number.csv" \
  --users "${RESPONSE_USERS:-100}" \
  --min-comments "${RESPONSE_MIN_COMMENTS:-200}" \
  --comments-per-user "${RESPONSE_COMMENTS_PER_USER:-500}"

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

# Ideas 4 and 5: toxic reply contagion and parent-child toxicity deltas.
python -u scripts/community/visualization/plot_parent_child_toxicity.py \
  "$PREDICTIONS_CSV" \
  --contagion-output "${VIS_DIR}/${COMMENTS_BASE}_idea_4_toxic_reply_contagion.png" \
  --delta-output "${VIS_DIR}/${COMMENTS_BASE}_idea_5_parent_child_toxicity_delta.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_4_5_parent_child_toxicity.csv" \
  --title-prefix "${SUBREDDIT}" \
  --threshold "$THRESHOLD"

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

# Idea 16: toxic-comment counts and percentages for top active users vs all users.
python -u scripts/community/visualization/plot_top_active_toxic_comment_average.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_vs_all_toxic_comment_average.png" \
  --percent-output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_vs_all_percent_toxic_comments.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_16_top_active_vs_all_toxic_comment_average.csv" \
  --histogram-output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_toxic_comments_by_post_number.png" \
  --normalized-histogram-output "${VIS_DIR}/${COMMENTS_BASE}_idea_16_top_active_toxic_comments_by_post_number_normalized.png" \
  --histogram-summary-output "${DATA_DIR}/${COMMENTS_BASE}_idea_16_top_active_toxic_comments_by_post_number.csv" \
  --title "${SUBREDDIT} Top Active Users vs All Users: Average Toxic Comment Count" \
  --percent-title "${SUBREDDIT} Top Active Users vs All Users: Percent Toxic Comments" \
  --threshold "$THRESHOLD" \
  --top-active-users "${TOP_ACTIVE_TOXIC_AVERAGE_USERS:-100}" \
  --max-post-number "${TOP_ACTIVE_TOXIC_POST_NUMBER_MAX:-100}"

echo "Community processing analyses complete for ${SUBREDDIT}"
