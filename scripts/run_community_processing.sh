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
  echo "Usage: $0 <Subreddit> <comments_base>"
  exit 1
fi

SUBREDDIT="$1"
COMMENTS_BASE="$2"

# compute toxicity percentages for all comments in a csv
python -u /home/wwalsh/detoxify_performance/scripts/community/processing/compute_toxicity_percentages.py \
  "data/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_detoxify_unbiased_predictions.csv" \
  --threshold 0.8 \
  --plot-output "visualizations/subreddits/${SUBREDDIT}/${COMMENTS_BASE}_toxicity_percentages_threshold_0.8.png"

