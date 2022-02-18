# %%
import numpy as np
import pandas as pd

CONTENT_VECS = "../data/intermediary/bert_vecs.npy"
TRAIN_IDS = "../data/intermediary/data_split/train_id_str.txt"
VAL_IDS = "../data/intermediary/data_split/val_id_str.txt"
TEST_IDS = "../data/intermediary/data_split/test_id_str.txt"
DATA = "../data/original/Org-Retweeted-Vectors_preproc.csv"

OUT_TRAIN = "../data/prepared/content_train_df.pkl"
OUT_VAL = "../data/prepared/content_val_df.pkl"
OUT_TEST = "../data/prepared/content_test_df.pkl"

# %%
with open(CONTENT_VECS, "rb") as f:
    vecs = np.load(f)
data = pd.read_csv(DATA, parse_dates=["created_at"]).iloc[:, 1:]
data.info()
# %%
with open(TRAIN_IDS) as f:
    train_ids = f.read().splitlines()
print("Number of train ids:", len(train_ids))
with open(VAL_IDS) as f:
    val_ids = f.read().splitlines()
print("Number of validation ids:", len(val_ids))
with open(TEST_IDS) as f:
    test_ids = f.read().splitlines()
print("Number of test ids:", len(test_ids))
# %%
data = data.astype({"id_str": str})
data_train = data[data.id_str.isin(train_ids)]
data_val = data[data.id_str.isin(val_ids)]
data_test = data[data.id_str.isin(test_ids)]

train_vecs = vecs[data_train.index]
val_vecs = vecs[data_val.index]