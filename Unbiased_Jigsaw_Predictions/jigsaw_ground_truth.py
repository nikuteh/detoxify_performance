import pandas as pd
from detoxify import Detoxify
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch

# select nvidia gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 5k samples
df = pd.read_csv("./jigsaw-unintended-bias/train.csv").sample(n=5000, random_state=42)

# Load the model
model = Detoxify('unbiased', device=device)

# Batch processing
batch_size = 64  
all_predictions = []
comments = df['comment_text'].tolist()

print("Running predictions on GPU...")
for i in tqdm(range(0, len(comments), batch_size)):
    batch = comments[i:i + batch_size]
    pred = model.predict(batch)
    all_predictions.extend(pred['toxicity'])

df['pred_toxicity'] = all_predictions
mse = mean_squared_error(df['target'], df['pred_toxicity'])
print(f"\nBaseline Mean Squared Error: {mse:.4f}")

