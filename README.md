# Detoxify Subreddit Workflow

Run commands from the project root:

```bash
cd /Users/williamwalsh/Desktop/REU/detoxify_performance
```

## Requirements

The scripts use Python plus `pandas`, `numpy`, `matplotlib`, `torch`, and
`detoxify`.

For `.zst` files, `format_reddit_comments_zst.py` uses either the Python package
`zstandard` or the command-line tool `zstd`.

```bash
pip install pandas numpy matplotlib zstandard torch detoxify
```

## File Flow

For each subreddit, the workflow is:

```text
comments.zst
  -> comments.csv
  -> comments_detoxify_unbiased_predictions.csv
  -> per-user CSV files
  -> rankings and plots
```

The ranking and user-plot scripts require Detoxify score columns:

```text
toxicity,severe_toxicity,obscene,identity_attack,insult,threat,sexual_explicit
```

The repo already has Asahi Detoxify predictions. For a new subreddit, run
`run_detoxify_on_csv.py` after creating the plain comments CSV.

## Asahi Linux

Convert the Reddit `.zst` comments dump to the Asahi-style CSV:

```bash
python format_reddit_comments_zst.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments.zst \
  --output asahi_linux_reddit/subreddits25/AsahiLinux_comments.csv
```

Run Detoxify on the comments CSV:

```bash
python run_detoxify_on_csv.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments.csv \
  --output asahi_linux_reddit/subreddits25/AsahiLinux_comments_detoxify_unbiased_predictions.csv
```

Plot overall subreddit toxicity over time as a scatter plot:

```bash
python plot_toxicity_over_time.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments_detoxify_unbiased_predictions.csv \
  --output subreddits/asahi_linux/asahi_linux_toxicity_scatter_over_time.png \
  --title "asahi_linux Toxicity Scatter Over Time" \
  --scatter-only \
  --dot-size 20 \
  --dot-alpha 0.7
```

Plot the number of posts/comments over time:

```bash
python plot_posts_over_time.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments_detoxify_unbiased_predictions.csv \
  --output subreddits/asahi_linux/asahi_linux_posts_over_time.png \
  --title "asahi_linux Posts Over Time" \
  --ylabel "Comments per month" \
  --time-bin MS
```

Compute the percent of comments above `0.5` for each toxicity type:

```bash
python compute_toxicity_percentages.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments_detoxify_unbiased_predictions.csv \
  --output asahi_linux_reddit/subreddits25/AsahiLinux_toxicity_percentages.csv \
  --plot-output asahi_linux_reddit/subreddits25/AsahiLinux_toxicity_percentages.png
```

Split Detoxify predictions into one CSV per active user:

```bash
python split_csv_by_active_users.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments_detoxify_unbiased_predictions.csv \
  subreddits/asahi_linux \
  --min-comments 100
```

Rank the top 10 users by average toxicity:

```bash
python rank_users_by_average_toxicity.py \
  subreddits/asahi_linux \
  --output subreddits/asahi_linux/top_10_average_toxicity.csv \
  --top-n 10
```

Plot average toxicity by post/comment number for the top 10 users:

```bash
python plot_top_toxic_users_average_toxicity.py \
  subreddits/asahi_linux \
  --output subreddits/asahi_linux/top_10_average_toxicity_per_post_scatter.png \
  --top-n 10
```

Plot average Detoxify scores by post/comment number for the top 100 users:

```bash
python plot_top_users_toxicity_by_post_number.py \
  asahi_linux_reddit/subreddits25/AsahiLinux_comments_detoxify_unbiased_predictions.csv \
  --output subreddits/asahi_linux/asahi_linux_top_100_users_toxicity_by_post_number.png \
  --summary-output subreddits/asahi_linux/asahi_linux_top_100_users_toxicity_by_post_number.csv \
  --title "asahi_linux Top 100 Users: Average Toxicity by Post Number" \
  --top-n 100
```

Plot one user's toxicity percentages by type:

```bash
python plot_user_toxicity_over_time.py \
  intulor \
  subreddits/asahi_linux/users
```

You can also pass a user CSV directly:

```bash
python plot_user_toxicity_over_time.py \
  subreddits/asahi_linux/users/intulor.csv
```

## Ramen

Convert the Ramen `.zst` comments dump to CSV:

```bash
python format_reddit_comments_zst.py \
  subreddits/ramen/ramen_comments.zst \
  --output subreddits/ramen/ramen_comments.csv
```

Run Detoxify on the comments CSV:

```bash
python run_detoxify_on_csv.py \
  subreddits/ramen/ramen_comments.csv \
  --output subreddits/ramen/ramen_comments_detoxify_unbiased_predictions.csv
```

Then plot overall subreddit toxicity over time:

```bash
python plot_toxicity_over_time.py \
  subreddits/ramen/ramen_comments_detoxify_unbiased_predictions.csv \
  --output subreddits/ramen/ramen_toxicity_over_time.png \
  --title "ramen Toxicity Over Time"
```

Compute the percent of comments above `0.5` for each toxicity type:

```bash
python compute_toxicity_percentages.py \
  subreddits/ramen/ramen_comments_detoxify_unbiased_predictions.csv \
  --output subreddits/ramen/ramen_toxicity_percentages.csv \
  --plot-output subreddits/ramen/ramen_toxicity_percentages.png
```

Split Detoxify predictions into one CSV per active user:

```bash
python split_csv_by_active_users.py \
  subreddits/ramen/ramen_comments_detoxify_unbiased_predictions.csv \
  subreddits/ramen/users \
  --min-comments 100
```

Rank the top 10 users by average toxicity:

```bash
python rank_users_by_average_toxicity.py \
  subreddits/ramen/users \
  --output subreddits/ramen/users/top_10_average_toxicity.csv \
  --top-n 10
```

Plot average toxicity by post/comment number for the top 10 users:

```bash
python plot_top_toxic_users_average_toxicity.py \
  subreddits/ramen/users \
  --output subreddits/ramen/users/top_10_average_toxicity_per_post_scatter.png \
  --top-n 10
```

Plot one user's toxicity percentages by type:

```bash
python plot_user_toxicity_over_time.py \
  USERNAME \
  subreddits/ramen/users
```

## Useful Options

Quickly test a `.zst` conversion with only the first few comments:

```bash
python format_reddit_comments_zst.py subreddits/ramen/ramen_comments.zst --limit 100
```

Quickly test Detoxify on only the first 100 comments:

```bash
python run_detoxify_on_csv.py subreddits/ramen/ramen_comments.csv --limit 100
```

Use a smaller Detoxify batch size if your computer runs out of memory:

```bash
python run_detoxify_on_csv.py subreddits/ramen/ramen_comments.csv --batch-size 16
```

Use a stricter toxicity cutoff:

```bash
python compute_toxicity_percentages.py \
  INPUT_predictions.csv \
  --threshold 0.8 \
  --plot-output toxicity_percentages_threshold_0.8.png
```

Use a different active-user cutoff:

```bash
python split_csv_by_active_users.py INPUT.csv OUTPUT_DIR --min-comments 50
```

Rank more than 10 users:

```bash
python rank_users_by_average_toxicity.py subreddits/ramen/users --top-n 25
```

Plot top-user averages where at least 5 of the top users have that post number:

```bash
python plot_top_toxic_users_average_toxicity.py \
  subreddits/ramen/users \
  --min-users-per-post 5
```

Change the over-time averaging interval:

```bash
python plot_toxicity_over_time.py INPUT_predictions.csv --time-bin QS
```
