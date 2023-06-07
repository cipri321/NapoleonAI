from typing import List, Optional, Dict, Tuple

import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from napoleonai.settings import DEFAULT_IMAGE_SIZE


def get_dataset(
        x_train,
        y_train,
        no_inputs
) -> Tuple[Tuple, Tuple]:
    """
    Creates a dataset that consists of pairs of (value, label) or just values if y_train is None

    :param x_train: values dataset
    :param y_train: labels dataset
    :param no_inputs: dimension of values dataset - length of x_train
    :return: Dataset
    """
    input_ds = []

    for i in range(no_inputs):
        tensors = tf.convert_to_tensor([x[i] for x in x_train])
        input_ds.append(tf.data.Dataset.from_tensor_slices(tensors, name=f'column_{i}'))

    input_ds = tf.data.Dataset.zip(tuple(input_ds), name='zipped_dataset')
    if y_train is None:
        return input_ds.batch(128)
    labels = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_train))
    train = tf.data.Dataset.zip((input_ds, labels)).batch(128)
    return train


def create_multimodal_dataset(
        df: pd.DataFrame,
        text_columns: List[str],
        image_columns: List[str],
        category_column: Optional[str],
        category_map: Optional[Dict],
        images_path=None,
        one_hot: bool = False,
        image_size: Tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
) -> tuple[tuple, tuple]:
    """
    Creates a multimodal dataset based on the given text and image columns
    It reads images from disk and converts them to tensors
    It can create either sparse or one_hot labels

    :param image_size: size of the images to be resized
    :param df: Dataframe, which contains the complete text data and the locations of images relative to images_path
    :param text_columns: names of the columns which contain text
    :param image_columns: names of the columns which contain image paths
    :param category_column: names of the category column
    :param category_map: mapping from the str category to a number from 0 to no_labels
    :param images_path: path to which all the values of image locations are relative to
    :param one_hot: true, if the created label should be one hot, false, otherwise
    :return:
    """
    x, y = [], []
    if category_column and not category_map:
        cat_map = {}
        no_cats = 0
    for idx, row in tqdm(df.iterrows()):
        images = []
        for img_col in image_columns:
            img = Image.open(images_path + '/' + row[img_col])
            img = img.resize(image_size)
            images.append(tf.convert_to_tensor(img))

        texts = []
        for text_col in text_columns:
            texts.append(tf.constant(row[text_col] if not pd.isna(row[text_col]) else '', dtype=tf.string))

        x.append(
            (
                *texts,
                *images
            )
        )
        if category_column:
            if not category_map:
                if str(row[category_column]) not in cat_map:
                    cat_map[str(row[category_column])] = no_cats
                    no_cats += 1
                y.append(cat_map[str(row[category_column])])
            else:
                y.append(category_map[str(row[category_column])])
    if category_column and category_map:
        no_cats = len(category_map.keys())
    if category_column and one_hot:
        print(no_cats)
        y = tf.one_hot(y, no_cats)
        print(len(y.numpy()[0]), y.numpy()[0])
    no_inputs = len(text_columns) + len(image_columns)
    if category_column:
        return get_dataset(x, y, no_inputs=no_inputs)
    else:
        return get_dataset(x, None, no_inputs=no_inputs)
