# Detoxify Subreddit Workflow

This project processes Reddit subreddit comment data into Detoxify toxicity
scores, community-level summaries, and individual-user summaries.

Run commands from the project root:

```bash
cd /Users/williamwalsh/Desktop/REU/detoxify_performance
source .venv/bin/activate
```

## Requirements

The scripts use Python plus `pandas`, `numpy`, `matplotlib`, `torch`, and
`detoxify`.

For `.zst` files, `scripts/format_reddit_comments_zst.py` uses either the
Python package `zstandard` or the command-line tool `zstd`.

```bash
pip install pandas numpy matplotlib zstandard torch detoxify
```

## Project Layout

Use one folder per subreddit:

```text
data/subreddits/<Subreddit>/
  Raw dumps, converted comments CSVs, Detoxify prediction CSVs, and per-user CSVs.

visualizations/subreddits/<Subreddit>/
  Generated plot images.

scripts/
  format_reddit_comments_zst.py
  run_detoxify_on_csv.py
  community/processing/
  community/visualization/
  users/visualization/
```

In the examples below:

```text
<Subreddit>     Folder-safe subreddit name, such as My_Subreddit.
<comments_base> Base filename for the comment dump, such as my_subreddit_comments.
<Username>      Reddit username to inspect individually.
```

## Data Flow

For each subreddit, the full workflow is:

```text
raw comment dump or plain-text source
  -> data/subreddits/<Subreddit>/<comments_base>.csv
  -> data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv
  -> community-level CSV summaries and PNG visualizations
  -> data/subreddits/<Subreddit>/users/*.csv
  -> individual-user CSV summaries and PNG visualizations
```

The Detoxify prediction CSV should contain these score columns:

```text
toxicity,severe_toxicity,obscene,identity_attack,insult,threat,sexual_explicit
```

## 1. Prepare The Data

Create the subreddit folders:

```bash
mkdir -p data/subreddits/<Subreddit>
mkdir -p visualizations/subreddits/<Subreddit>
```

If your raw source is a Reddit comments `.zst` dump containing newline-delimited
JSON, convert it to the project CSV format:

```bash
python scripts/format_reddit_comments_zst.py \
  data/subreddits/<Subreddit>/<comments_base>.zst \
  --output data/subreddits/<Subreddit>/<comments_base>.csv
```

The converted comments CSV has this shape:

```text
Subreddit,comment_id,username,timestamp,comment_text,parent_id
```

If your source is already plain text or another CSV format, convert it into that
same CSV shape before running Detoxify. The important columns for later scripts
are `comment_id`, `username`, `timestamp`, `comment_text`, and `parent_id`.

For a quick conversion test:

```bash
python scripts/format_reddit_comments_zst.py \
  data/subreddits/<Subreddit>/<comments_base>.zst \
  --limit 100
```

## 2. Run Detoxify

Score every comment and write a predictions CSV:

```bash
python scripts/run_detoxify_on_csv.py \
  data/subreddits/<Subreddit>/<comments_base>.csv \
  --output data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv
```

For a quick Detoxify test:

```bash
python scripts/run_detoxify_on_csv.py \
  data/subreddits/<Subreddit>/<comments_base>.csv \
  --limit 100
```

Use a smaller batch size if your computer runs out of memory:

```bash
python scripts/run_detoxify_on_csv.py \
  data/subreddits/<Subreddit>/<comments_base>.csv \
  --batch-size 16
```

## 3. Community-Level Processing

Set the predictions path once mentally as:

```text
data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv
```

Compute the percent of comments above a toxicity threshold for each Detoxify
score type:

```bash
python scripts/community/processing/compute_toxicity_percentages.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output data/subreddits/<Subreddit>/<comments_base>_toxicity_percentages.csv \
  --plot-output visualizations/subreddits/<Subreddit>/<comments_base>_toxicity_percentages.png
```

Use a stricter toxicity cutoff:

```bash
python scripts/community/processing/compute_toxicity_percentages.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --threshold 0.8 \
  --plot-output visualizations/subreddits/<Subreddit>/<comments_base>_toxicity_percentages_threshold_0.8.png
```

Save every comment with `toxicity` above `0.5` to its own CSV:

```bash
python scripts/community/processing/filter_toxic_comments.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output data/subreddits/<Subreddit>/<comments_base>_toxicity_above_0_5.csv
```

Clean the community CSV by keeping only comment replies with `parent_id`
starting with `t1_`, dropping deleted users, removing URLs from comment text,
and adding each user's chronological subreddit comment number:

```bash
python scripts/community/processing/csv_cleaner.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output data/subreddits/<Subreddit>/<comments_base>_cleaned.csv
```

Split the prediction CSV into one CSV per active user:

```bash
python scripts/community/processing/split_csv_by_active_users.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  data/subreddits/<Subreddit>/users \
  --min-comments 100
```

Use a different active-user cutoff:

```bash
python scripts/community/processing/split_csv_by_active_users.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  data/subreddits/<Subreddit>/users \
  --min-comments 50
```

Rank users by average toxicity:

```bash
python scripts/community/processing/rank_users_by_average_toxicity.py \
  data/subreddits/<Subreddit>/users \
  --output data/subreddits/<Subreddit>/users/top_10_average_toxicity.csv \
  --top-n 10
```

Rank more users:

```bash
python scripts/community/processing/rank_users_by_average_toxicity.py \
  data/subreddits/<Subreddit>/users \
  --top-n 25
```

## 4. Community-Level Visualizations

Plot overall toxicity over time:

```bash
python scripts/community/visualization/plot_toxicity_over_time.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output visualizations/subreddits/<Subreddit>/<comments_base>_toxicity_over_time.png \
  --title "<Subreddit> Toxicity Over Time"
```

Plot toxicity over time as individual comment dots:

```bash
python scripts/community/visualization/plot_toxicity_over_time.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output visualizations/subreddits/<Subreddit>/<comments_base>_toxicity_scatter_over_time.png \
  --title "<Subreddit> Toxicity Scatter Over Time" \
  --scatter-only \
  --dot-size 20 \
  --dot-alpha 0.7
```

Change the over-time averaging interval:

```bash
python scripts/community/visualization/plot_toxicity_over_time.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --time-bin QS
```

Plot comment volume over time:

```bash
python scripts/community/visualization/plot_posts_over_time.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output visualizations/subreddits/<Subreddit>/<comments_base>_posts_over_time.png \
  --title "<Subreddit> Comments Over Time" \
  --ylabel "Comments per month" \
  --time-bin MS
```

Plot average toxicity by post/comment number for the top toxic users:

```bash
python scripts/community/visualization/plot_top_toxic_users_average_toxicity.py \
  data/subreddits/<Subreddit>/users \
  --output visualizations/subreddits/<Subreddit>/top_10_average_toxicity_per_post_scatter.png \
  --top-n 10
```

Plot only post/comment numbers where at least 5 of the top users contributed:

```bash
python scripts/community/visualization/plot_top_toxic_users_average_toxicity.py \
  data/subreddits/<Subreddit>/users \
  --min-users-per-post 5
```

Plot average Detoxify scores by post/comment number for the top users in the
full community CSV:

```bash
python scripts/community/visualization/plot_top_users_toxicity_by_post_number.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output visualizations/subreddits/<Subreddit>/<comments_base>_top_100_users_toxicity_by_post_number.png \
  --summary-output data/subreddits/<Subreddit>/<comments_base>_top_100_users_toxicity_by_post_number.csv \
  --title "<Subreddit> Top 100 Users: Average Toxicity by Post Number" \
  --top-n 100
```

Plot average toxicity and percent of comments above `0.5` toxicity over each
user's time in the subreddit:

```bash
python scripts/community/visualization/plot_all_users_toxicity_over_user_time.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --average-output visualizations/subreddits/<Subreddit>/<comments_base>_all_users_average_toxicity_over_user_time.png \
  --percent-output visualizations/subreddits/<Subreddit>/<comments_base>_all_users_percent_toxic_over_user_time.png \
  --summary-output data/subreddits/<Subreddit>/<comments_base>_all_users_toxicity_over_user_time.csv
```

Plot the parent-comment post-number distribution for toxic comments:

```bash
python scripts/community/visualization/cultural_violence_test.py \
  data/subreddits/<Subreddit>/<comments_base>_cleaned_toxicity_above_0_5.csv \
  data/subreddits/<Subreddit>/<comments_base>_cleaned.csv \
  --output visualizations/subreddits/<Subreddit>/<comments_base>_cultural_violence_parent_post_numbers.png \
  --summary-output data/subreddits/<Subreddit>/<comments_base>_cultural_violence_parent_post_numbers.csv
```

Plot average toxicity and percent toxic by post/comment number for 100 randomly
sampled users with at least 200 comments:

```bash
python scripts/community/visualization/plot_random_users_toxicity_by_post_number.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --average-output visualizations/subreddits/<Subreddit>/<comments_base>_random_users_average_toxicity_by_post_number.png \
  --percent-output visualizations/subreddits/<Subreddit>/<comments_base>_random_users_percent_toxic_by_post_number.png \
  --summary-output data/subreddits/<Subreddit>/<comments_base>_random_users_toxicity_by_post_number.csv
```

Plot the average toxicity of direct replies to 100 randomly sampled active
users' comments by each user's post/comment number:

```bash
python scripts/community/visualization/plot_response_toxicity_by_user_post_number.py \
  data/subreddits/<Subreddit>/<comments_base>_detoxify_unbiased_predictions.csv \
  --output visualizations/subreddits/<Subreddit>/<comments_base>_response_toxicity_by_post_number.png \
  --summary-output data/subreddits/<Subreddit>/<comments_base>_response_toxicity_by_post_number.csv \
  --users 100 \
  --min-comments 200 \
  --comments-per-user 500
```

## 5. Individual-User Processing And Visualization

After splitting users into `data/subreddits/<Subreddit>/users`, plot one user's
toxicity percentages by type:

```bash
python scripts/users/visualization/plot_user_toxicity_over_time.py \
  <Username> \
  data/subreddits/<Subreddit>/users \
  --output visualizations/subreddits/<Subreddit>/users/<Username>_toxicity_percentages.png
```

You can also pass a user CSV directly:

```bash
python scripts/users/visualization/plot_user_toxicity_over_time.py \
  data/subreddits/<Subreddit>/users/<Username>.csv
```

Optionally save the user's percentage summary as CSV:

```bash
python scripts/users/visualization/plot_user_toxicity_over_time.py \
  <Username> \
  data/subreddits/<Subreddit>/users \
  --summary-output data/subreddits/<Subreddit>/users/<Username>_toxicity_percentages.csv \
  --output visualizations/subreddits/<Subreddit>/users/<Username>_toxicity_percentages.png
```
