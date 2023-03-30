from typing import List

import tensorflow as tf
import pandas as pd

from napoleonai.settings import DEFAULT_IMAGE_SIZE, DEFAULT_NO_CHANNELS


def get_combined_model(
        text_models: List[tf.keras.Model],
        image_models: List[tf.keras.Model],
        no_labels: int
):
    """
    Creates a multimodal classification model based on multiple text models and image models

    :param text_models: List of text feature extractors
    :param image_models: List of image feature extractors
    :param no_labels: int, no of labels
    :return: multimodal classification keras model
    """
    text_inputs = []
    image_inputs = []

    for idx, textModel in enumerate(text_models):
        textModel.trainable = False
        text_inputs.append(tf.keras.Input((), dtype=tf.string, name=f'input_text_{idx}'))

    for idx, imageModel in enumerate(image_models):
        imageModel.trainable = False
        image_inputs.append(
            tf.keras.Input((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_NO_CHANNELS), name=f'input_image_{idx}'))

    features = [
        model(input_layer)
        for input_layer, model in zip(text_inputs + image_inputs, text_models + image_models)
    ]

    concatenate_layer = tf.keras.layers.Concatenate()(features)

    dense2 = tf.keras.layers.Dense(128, activation='sigmoid')(concatenate_layer)
    dense3 = tf.keras.layers.Dense(64, activation='sigmoid')(dense2)
    output = tf.keras.layers.Dense(no_labels, activation='softmax')(dense3)

    return tf.keras.Model(inputs=text_inputs + image_inputs, outputs=output)


def get_multimodal_classification_model(
        text_models: List[tf.keras.Model],
        image_models: List[tf.keras.Model],
        no_labels: int
) -> tf.keras.Model:
    """
    Creates a multimodal classification model based on multiple text models and image models

    :param text_models: List of text feature extractors
    :param image_models: List of image feature extractors
    :param no_labels: int, no of labels
    :return: multimodal classification keras model
    """
    print(no_labels)
    model = get_combined_model(text_models, image_models, no_labels)
    # model.summary()
    return model
