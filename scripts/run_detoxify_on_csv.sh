#!/bin/bash
#SBATCH --job-name=run_detoxify_on_csv
#SBATCH --output=detoxify_results%j.log
#SBATCH --error=detoxify_error%j.log

#SBATCH --partition=gpuq
#SBATCH  --gres=gpu:1 
#SBATCH  --mem=16G 
#SBATCH  --cpus-per-task=4 
#SBATCH  --time=02:00:00 

python -u /home/wwalsh/detoxify_performance/scripts/run_detoxify_on_csv.py data/subreddits/Ramen/ramen_comments.csv --output data/subreddits/Ramen/ramen_comments_detoxify_unbiased_predictions.csv