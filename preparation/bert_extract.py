"""BERT vector extraction"""
# https://huggingface.co/InfoCoV/Cro-CoV-cseBERT

# %%
import numpy as np
import pandas as pd
from sentence_transformers import models
from sentence_transformers import SentenceTransformer

# Set DATA and VECTORS to data inputs and outputs
DATA = "../data/original/Org-Retweeted-Vectors_preproc.csv"
VECTORS = "../data/intermediary/bert_vecs.npy"

# %%
word_embedding_model = models.Transformer('InfoCoV/Cro-CoV-cseBERT')
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False)

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], device='cuda:0')

# %%
df = pd.read_csv(DATA, parse_dates=["created_at"]).iloc[:, 1:]

# %%
# does batching internally, default 32
texts_emb = model.encode(df.full_text.values)
# %%
with open(VECTORS, "wb") as f:
    np.save(f, texts_emb)
