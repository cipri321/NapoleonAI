from typing import Tuple

import numpy as np
import tensorflow as tf
import pandas as pd


def convert_to_one_hot(labels: pd.Series):
    no_labels = int(labels.value_counts().count())
    one_hot_labels = tf.one_hot(labels.values, no_labels)
    return one_hot_labels


def get_dataset_from_df(
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        one_hot_label: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if (isinstance(df, pd.DataFrame) and df.empty) or (not isinstance(df, pd.DataFrame) and not df):
        return None, None
    df = df[[text_column, label_column]]
    labels = np.array(df[label_column].values)
    if one_hot_label:
        labels = np.array(convert_to_one_hot(df[label_column]))

    return np.array(df[text_column].values), labels
