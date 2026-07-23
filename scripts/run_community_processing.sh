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
find "$VIS_DIR" -type f -name '*.png' -delete

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
  --output "${VIS_DIR}/${COMMENTS_BASE}_toxicity_over_time.png" \
  --title "${SUBREDDIT} Toxicity Over Time"

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

python -u scripts/community/visualization/plot_all_users_toxicity_by_post_number.py \
  "$PREDICTIONS_CSV" \
  --average-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_average_toxicity_by_post_number.png" \
  --percent-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_percent_toxic_by_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_all_users_toxicity_by_post_number.csv"

python -u scripts/community/visualization/plot_all_users_toxicity_over_user_time.py \
  "$PREDICTIONS_CSV" \
  --average-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_average_toxicity_over_user_time.png" \
  --percent-output "${VIS_DIR}/${COMMENTS_BASE}_all_users_percent_toxic_over_user_time.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_all_users_toxicity_over_user_time.csv"

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

python -u scripts/community/visualization/cultural_violence_test_4.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_comment_vs_response_toxicity_by_parent_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_comment_vs_response_toxicity_by_parent_post_number.csv" \
  --title "${SUBREDDIT} Comment vs Response Toxicity by Parent Post Number"

python -u scripts/community/visualization/plot_response_toxicity_by_user_post_number.py \
  "$PREDICTIONS_CSV" \
  --output "${VIS_DIR}/${COMMENTS_BASE}_response_toxicity_by_post_number.png" \
  --summary-output "${DATA_DIR}/${COMMENTS_BASE}_response_toxicity_by_post_number.csv" \
  --users "${RESPONSE_USERS:-100}" \
  --min-comments "${RESPONSE_MIN_COMMENTS:-200}" \
  --comments-per-user "${RESPONSE_COMMENTS_PER_USER:-500}"
