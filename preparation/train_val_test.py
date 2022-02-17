"""Dataset split"""
# Removes row entries where `tweets_keywords_3` not set

# %%
import os

import pandas as pd

# Set constants to data inputs and desired outputs
DATA = "../data/original/Org-Retweeted-Vectors_preproc.csv"
OUT_DIR = "../data/intermediary/data_split/"

# %%
df = pd.read_csv(DATA, parse_dates=["created_at"]).iloc[:, 1:]
print("Dataset length:", len(df))

remaining = df[~df.tweets_keywords_3_in_degree.isna()]
print("Length after NaN tweets_keywords_3 removal:", len(remaining))

# %%
# already sorted by date
N = len(remaining)
train_perc = 0.80
val_perc = 0.10
test_perc = 0.10

train_idx = int(N * train_perc)
val_idx = train_idx + int(N * val_perc)

train = remaining.id_str[0:train_idx]
val = remaining.id_str[train_idx: val_idx]
test = remaining.id_str[val_idx:]

# %%
print("Training length:", len(train))
print("Validation length:", len(val))
print("Test length:", len(val))

assert len(train) + len(val) + len(test) == len(remaining)
# %%
print("Storing data splits to", OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "train_id_str.txt"), "w") as f:
    for v in train.values:
        f.write(str(v) + "\n")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "val_id_str.txt"), "w") as f:
    for v in val.values:
        f.write(str(v) + "\n")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "test_id_str.txt"), "w") as f:
    for v in test.values:
        f.write(str(v) + "\n")
# %%
