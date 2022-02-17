# %%
import numpy as np
import pandas as pd

# %%
TRAIN = "../data/prepared/train_features.pkl"
VAL = "../data/prepared/val_features.pkl"
TEST = "../data/prepared/test_features.pkl"

CONTENT_VECS = "../data/intermediary/bert_vecs.npy"

OUT_TRAIN = "../data/prepared/full_train_df.pkl"
OUT_VAL = "../data/prepared/full_val_df.pkl"
OUT_TEST = "../data/prepared/full_test_df.pkl"

OUT_TRAIN_CONTENT = "../data/prepared/content_train_df.pkl"
OUT_VAL_CONTENT = "../data/prepared/content_val_df.pkl"
OUT_TEST_CONTENT = "../data/prepared/content_test_df.pkl"

# %%
with open(CONTENT_VECS, "rb") as f:
    vecs = np.load(f)

# %%
train_df = pd.read_pickle(TRAIN)
val_df = pd.read_pickle(VAL)
test_df = pd.read_pickle(TEST)
# %%
train_vecs = vecs[train_df.index]
val_vecs = vecs[val_df.index]
test_vecs = vecs[test_df.index]
# %%
train_vecs_df = pd.DataFrame(train_vecs, index=train_df.index)
val_vecs_df = pd.DataFrame(val_vecs, index=val_df.index)
test_vecs_df = pd.DataFrame(test_vecs, index=test_df.index)
# %%
train_df_full = pd.concat([train_df, train_vecs_df], axis=1)
val_df_full = pd.concat([val_df, val_vecs_df], axis=1)
test_df_full = pd.concat([test_df, test_vecs_df], axis=1)
# %%
# SAVE DF with full features
train_df_full.to_pickle(OUT_TRAIN)
val_df_full.to_pickle(OUT_VAL)
test_df_full.to_pickle(OUT_TEST)
# %%
train_vecs_df["retweet_label"] = train_df["retweet_label"]
val_vecs_df["retweet_label"] = val_df["retweet_label"]
test_vecs_df["retweet_label"] = test_df["retweet_label"]

train_vecs_df["id_str"] = train_df["id_str"]
val_vecs_df["id_str"] = val_df["id_str"]
test_vecs_df["id_str"] = test_df["id_str"]

# %%
# SAVE DF with content features only
train_vecs_df.to_pickle(OUT_TRAIN_CONTENT)
val_vecs_df.to_pickle(OUT_VAL_CONTENT)
test_vecs_df.to_pickle(OUT_TEST_CONTENT)

# %%
