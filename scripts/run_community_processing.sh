#!/bin/bash
#SBATCH --job-name=run_detoxify_on_csv
#SBATCH --output=detoxify_results%j.log
#SBATCH --error=detoxify_error%j.log

#SBATCH --partition=gpuq
#SBATCH  --gres=gpu:1 
#SBATCH  --mem=16G 
#SBATCH  --cpus-per-task=4 
#SBATCH  --time=02:00:00 

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 ${SUBREDDIT} ${COMMENTS_BASE}"
  exit 1
fi

SUBREDDIT="$1"
COMMENTS_BASE="$2"

# compute toxicity percentages for all comments in a csv
python -u scripts/community/processing/compute_toxicity_percentages.py \
  "data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_detoxify_unbiased_predictions_cleaned.csv" \
  --threshold 0.8 \
  --plot-output "visualizations/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_toxicity_percentages_threshold_0.8.png"

#REMOVE THIS LATER
python scripts/community/processing/filter_toxic_comments.py \
  data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_detoxify_unbiased_predictions.csv \
  --output data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_toxicity_above_0_5.csv


python -u /scripts/community/visualization/cultural_violence_test.py
  data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_cleaned_toxicity_above_0_5.csv \
  data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_cleaned.csv \
  --output visualizations/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_cultural_violence_parent_post_numbers.png \
  --summary-output data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_cultural_violence_parent_post_numbers.csv