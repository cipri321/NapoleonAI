import pathlib
from typing import Tuple

import numpy
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import time


def load_dataset(
        image_size: int,
        batch_size: int,
        raw_data_path: str,
        validation_split: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates a tensorflow dataset of images from directory raw_data_path
    The directory should have the structure:
    |cat1
        |img1
        |img2
        |...
    |cat2
        |img1
        |img2
        |..
    |...
    :param image_size: the size at which the image is processed(width=height=image_size)
    :param batch_size: the size of the batch of the dataset
    :param raw_data_path: the path to the image folder
    :param validation_split: the proportion of the dataset that is kept for validation
    :return: Tuple - two datasets, first for train, second for validation
    """
    data_dir = pathlib.Path(raw_data_path)
    train, validation = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=int(time.time()),
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode='categorical',
        validation_split=validation_split,
        subset='both'
    )

    return train, validation


def generate_dataset_stats(dataset: tf.data.Dataset, name='') -> int:
    """
    Generates dataset stats using matplotlib

    :param dataset: tensorflow dataset
    :param name: str, name displayed in the matplotlib plot
    :return: no of labels existing in the dataset
    """
    category_stats = {}
    no_images = 0
    for batched_image_tensor, batched_one_hot_label in tqdm(dataset, desc=f'Dataset Stats - {name}'):
        for image_tensor, one_hot_label in zip(batched_image_tensor, batched_one_hot_label):
            label = np.where(one_hot_label == 1)
            if len(label) == 0 or len(label) > 1:
                raise ValueError('Invalid one hot label')
            label = label[0][0]
            if label not in category_stats:
                category_stats[label] = 0
            category_stats[label] += 1
            no_images += 1

    sorted_stats = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)

    x_sort, y_sort = [item[0] for item in sorted_stats], [item[1] for item in sorted_stats]

    plt.plot(y_sort)
    plt.title(f'{name} - Category distribution - sorted')
    plt.show()

    stats_len = len(sorted_stats)

    fig, ax = plt.subplots()
    x_first, y_first = [str(x) for x in x_sort[:min(int(stats_len*0.25), 10)]], y_sort[:min(int(stats_len*0.25), 10)]
    bars = ax.bar(x_first, y_first)
    ax.bar_label(bars, labels=['%.2f' % (p / no_images * 100) for p in y_first])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.title(f'{name} - First 10 categories')
    plt.show()

    fig, ax = plt.subplots()
    x_last, y_last = [str(x) for x in x_sort[stats_len-min(int(stats_len * 0.25), 10):]], y_sort[stats_len-min(int(stats_len * 0.25), 10):]
    bars = ax.bar(x_last, y_last)
    ax.bar_label(bars, labels=['%.2f' % (p / no_images * 100) for p in y_last])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.title(f'{name} - Last 10 categories')
    plt.show()

    return len(category_stats.keys())
