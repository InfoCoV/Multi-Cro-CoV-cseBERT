# %%
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer


# %%
DATA = "../data/orginal/Org-Retweeted-Vectors_preproc.csv"

TRAIN = "../data/intermediary/data_split/train_id_str.txt"
VAL = "../data/intermediary/data_split/val_id_str.txt"
TEST = "../data/intermediary/data_split/test_id_str.txt"

OUT_TRAIN = "../data/prepared/train_features.pkl"
OUT_VAL = "../data/prepared/val_features.pkl"
OUT_TEST = "../data/prepared/test_features.pkl"


# %%
def retweet_categories(retweet_count):
    if retweet_count == 1:
        return 0
    elif retweet_count >= 2:
        return 1
    else:
        raise ValueError


def add_labels(df):
    df["retweet_label"] = data.retweet_count.map(retweet_categories)
    return df


def remove_cols(df):
    return df.drop(
        ["created_at", "full_text", "favorite_count", "entities.hashtags",
         "user_id_str", "user.screen_name", "covid_keywords",
         "mentioned_users_ids", "mentioned_users_usernames",
         "tweets_keywords_3_louvian_class",
         "folowing_users_graph_louvian_class",
         "users_reply_louvian_class",
         "retweet_count"], axis=1)


def set_types(df):
    return df.astype({
         "entities.urls": int,
         "entities.media": int,
         "user_in_net": int,
         "has_covid_keyword": int,
         })


def set_nan_indicators(df):
    df["user.followers_isna"] = df["user.followers_count"].isna().astype(int)
    df["users_mention_isna"] = df["users_mention_in_degree"].isna().astype(int)
    df["following_users_isna"] = df[
        "folowing_users_graph_in_degree"].isna().astype(int)
    df["users_reply_isna"] = df["users_reply_in_degree"].isna().astype(int)
    return df


def add_logs(df):
    df["log1p_num_hashtags"] = df["number_of_hashtags"].map(np.log1p)
    df["log1p_followers_count"] = df["user.followers_count"].map(np.log1p)
    df["log1p_friends_count"] = df["user.friends_count"].map(np.log1p)
    df["log1p_statuses_count"] = df["user.statuses_count"].map(np.log1p)
    df["log1p_num_mentioned"] = df["number_of_mentioned_users"].map(np.log1p)
    return df


class MainFeatureProcess(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_cols = [
            "user.followers_count",
            "user.friends_count",
            "user.statuses_count",
            "folowing_users_graph_in_degree",
            "folowing_users_graph_out_degree",
            "folowing_users_graph_in_strength",
            "folowing_users_graph_out_strength",
            "folowing_users_graph_eigenvector_in",
            "folowing_users_graph_eigenvector_out",
            "folowing_users_graph_katz_in",
            "folowing_users_graph_katz_in",
            "folowing_users_graph_katz_out",
            "folowing_users_graph_clustering"
        ]
        self.zero_cols = [
            'users_mention_in_degree',
            'users_mention_out_degree',
            'users_mention_in_strength',
            'users_mention_out_strength',
            'users_mention_eigenvector_in',
            'users_mention_eigenvector_out',
            'users_mention_katz_in',
            'users_mention_katz_out',
            'users_mention_clustering',
            'users_reply_in_degree',
            'users_reply_out_degree',
            'users_reply_in_strength',
            'users_reply_out_strength',
            'users_reply_eigenvector_in',
            'users_reply_eigenvector_out',
            'users_reply_katz_in',
            'users_reply_katz_out',
            'users_reply_clustering'
        ]

        self.log_cols = [
            "number_of_hashtags", "user.followers_count", "user.friends_count",
            "user.statuses_count", "number_of_mentioned_users"]

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame)
        self.median_imputer = SimpleImputer(strategy="median")
        self.median_imputer.fit(X[self.median_cols])
        self.constant_imputer = SimpleImputer(
            strategy="constant", fill_value=0)
        self.constant_imputer.fit(X[self.zero_cols])
        return self

    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)
        X = X.copy()
        X = add_labels(X)
        X = remove_cols(X)
        X = set_types(X)
        X = set_nan_indicators(X)
        median_impute = self.median_imputer.transform(X[self.median_cols])
        X[self.median_cols] = median_impute
        zero_impute = self.constant_imputer.transform(X[self.zero_cols])
        X[self.zero_cols] = zero_impute
        X = add_logs(X)
        X.drop(self.log_cols, axis=1, inplace=True)
        return X


class FeatureScaling(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame)
        self.network_cols = [
            c for c in X.columns if
            c.startswith("tweets_keywords_3") or
            c.startswith("users_mention") or
            c.startswith("folowing_users") or
            c.startswith("users_reply")]
        self.other_cols = [c for c in X.columns if c.startswith("log1p")]
        self.robust_scaler = RobustScaler().fit(X[self.other_cols])
        self.network_transformer = PowerTransformer().fit(X[self.network_cols])
        return self

    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)
        X = X.copy()
        X[self.other_cols] = self.robust_scaler.transform(X[self.other_cols])
        X[self.network_cols] = self.network_transformer.transform(
            X[self.network_cols])
        return X


# %%
data = pd.read_csv(DATA, parse_dates=["created_at"]).iloc[:, 1:]
data = data.astype({"id_str": str})
data.info()
# %%
with open(TRAIN) as f:
    train_ids = f.read().splitlines()
with open(VAL) as f:
    val_ids = f.read().splitlines()
with open(TEST) as f:
    test_ids = f.read().splitlines()


# %%
data_train = data[data.id_str.isin(train_ids)].copy()
print(len(data_train))

data_val = data[data.id_str.isin(val_ids)].copy()
print(len(data_val))

data_test = data[data.id_str.isin(test_ids)].copy()
print(len(data_test))

# DATA PROCESSING
# %%
main_process = MainFeatureProcess()
tr_data = main_process.fit_transform(data_train)
# %%
scaling = FeatureScaling()
tr_data_2 = scaling.fit_transform(tr_data)
# %%
v_data = main_process.transform(data_val)
v_data_2 = scaling.transform(v_data)
# %%
te_data = main_process.transform(data_test)
te_data_2 = scaling.transform(te_data)

# %%
tr_data_2.to_pickle(OUT_TRAIN)
# %%
v_data_2.to_pickle(OUT_VAL)
# %%
te_data_2.to_pickle(OUT_TEST)


# CHECKS
# %%
cols = [c for c in tr_data_2.columns if c.startswith("tweets_keywords")]
tr_data_2[cols].describe()
# %%
tr_data[cols].describe()
# %%
cols = [c for c in tr_data_2.columns if c.startswith("users_mention")]
tr_data_2[cols].describe()
# %%
cols = [c for c in tr_data_2.columns if c.startswith("folowing_users")]
tr_data_2[cols].describe()
# %%
cols = [c for c in tr_data_2.columns if c.startswith("users_reply")]
tr_data_2[cols].describe()
# %%
