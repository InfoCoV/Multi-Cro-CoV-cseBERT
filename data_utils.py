import logging
import pandas as pd


from sklearn.preprocessing import Normalizer


LOGGER = logging.getLogger(__name__)


def setup_data(config):
    LOGGER.info(f"Loading f{config.data.train}")
    train_df = pd.read_pickle(config.data.train)
    train_df_labels = train_df.retweet_label
    train_df.drop(["retweet_label", "id_str"], axis=1, inplace=True)

    LOGGER.info(f"Loading f{config.data.val}")
    val_df = pd.read_pickle(config.data.val)
    val_df_labels = val_df.retweet_label
    val_df.drop(["retweet_label", "id_str"], axis=1, inplace=True)

    vec_cols = list(range(768))
    scaler = None
    LOGGER.info(f"Normalizing content: {config.normalize_content}")
    if config.normalize_content:
        scaler = Normalizer()
        transformed = scaler.fit_transform(train_df[vec_cols].values)
        train_df[vec_cols] = transformed

        transformed_val = scaler.transform(val_df[vec_cols].values)
        val_df[vec_cols] = transformed_val

    return {
        "train_data": train_df,
        "val_data": val_df,
        "train_labels": train_df_labels,
        "val_labels": val_df_labels,
        "scaler": scaler}
