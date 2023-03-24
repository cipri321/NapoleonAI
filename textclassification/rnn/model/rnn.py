from typing import Optional, Tuple

import tensorflow as tf
import numpy as np

from settings import DEFAULT_RNN_MODEL_NAME, DEFAULT_TEXT_INPUT_RAW_SUFFIX, DEFAULT_TEXT_ENCODING_LAYER_SUFFIX, \
    DEFAULT_EMBEDDING_LAYER_SUFFIX, DEFAULT_RNN_LAYER_NAME, DEFAULT_FEATURES_LAYER_NAME, \
    DEFAULT_CLASSIFIER_ACTIVATION_FUNCTION
from textclassification.common.classifier import create_classifier
from textclassification.common.common_types import Vocabulary


def get_rnn_layer_from_layer_type(rnn_layer_type: str) -> tf.keras.layers.Layer:
    """
    Returns a Keras layer class based on the parameter rnn_layer_type

    :param rnn_layer_type: str - lstm of gru
    :return: Keras layer - used to create an rnn layer
    """
    layer_dict = {
        'lstm': tf.keras.layers.LSTM,
        'gru': tf.keras.layers.GRU
    }
    return layer_dict[rnn_layer_type]


def get_rnn_for_text_classification(
        model_save_path: Optional[str],
        model_name: Optional[str],
        pretrained_word_embeddings: Optional[Tuple[tf.keras.layers.Embedding, Vocabulary]],
        word_embedding_size: Optional[int],
        vocabulary_dataset: Optional[tf.data.Dataset],
        rnn_layers_sizes: Tuple[int, ...],
        rnn_cell_type: str,
        dense_layers: Tuple[int, ...],
        no_labels: int
) -> tf.keras.Model:
    """
    Creates a text classification model based on an rnn for feature extraction
    and a classifier consisting of dense layers
    It can either use pretrained word embeddings(Embedding layer+Vocabulary) or
    it can learn them from scratch

    :param model_save_path: str - the function will return the model located at model_save_path
    :param model_name: str - name of the keras model, it will also be used for naming the layers
    :param pretrained_word_embeddings:
    Embedding layer + Vocabulary, the function will use these pretrained word embeddings
    None, otherwise
    :param word_embedding_size: int, size of the word embeddings
    :param vocabulary_dataset: tensorflow Dataset, a dataset for learning a vocabulary if no embeddings are provided
    :param rnn_layers_sizes: list of int, the sizes of each bidirectional rnn layer
    !!! For list [1, 2, 3] the number of parameters for each layer is [2, 4, 6], because it is bidirectional !!!
    :param rnn_cell_type: str - lstm or gru
    :param dense_layers: list of int, size of each dense layer
    :param no_labels: int, number of unique labels
    :return: Keras model used for text classification
    """
    if model_save_path:
        return tf.keras.models.load_model(model_save_path)

    model_name = model_name if model_name else DEFAULT_RNN_MODEL_NAME

    input_text = tf.keras.layers.Input(
        shape=(),
        dtype=tf.string,
        name=f'{model_name}_{DEFAULT_TEXT_INPUT_RAW_SUFFIX}'
    )

    if pretrained_word_embeddings:
        encoding_layer = tf.keras.layers.TextVectorization(
            name=f'{model_name}_{DEFAULT_TEXT_ENCODING_LAYER_SUFFIX}',
            vocabulary=pretrained_word_embeddings[1],
            max_tokens=len(pretrained_word_embeddings[1])+100
        )
        embedding_layer = pretrained_word_embeddings[0]
    else:
        encoding_layer = tf.keras.layers.TextVectorization(
            name=f'{model_name}_{DEFAULT_TEXT_ENCODING_LAYER_SUFFIX}'
        )
        if not vocabulary_dataset.any():
            raise ValueError('If no word embeddings are provided, a vocabulary dataset should be provided')
        encoding_layer.adapt(vocabulary_dataset)
        embedding_layer = tf.keras.layers.Embedding(
            len(encoding_layer.get_vocabulary()),
            word_embedding_size,
            mask_zero=True,
            name=f'{model_name}_{DEFAULT_EMBEDDING_LAYER_SUFFIX}'
        )

    encoded_tensor: tf.Tensor = encoding_layer(input_text)
    embedding_tensor: tf.Tensor = embedding_layer(encoded_tensor)

    if len(rnn_layers_sizes) == 0:
        raise ValueError('There should be at least one rnn layer')

    bidir_tensor = embedding_tensor
    rnn_layer = get_rnn_layer_from_layer_type(rnn_cell_type)

    for i in range(len(rnn_layers_sizes) - 1):
        bidir_layer: tf.keras.layers.Layer = tf.keras.layers.Bidirectional(
            rnn_layer(rnn_layers_sizes[i], return_sequences=True),
            name=f'{model_name}_{DEFAULT_RNN_LAYER_NAME}_{i + 1}'
        )
        bidir_tensor: tf.Tensor = bidir_layer(bidir_tensor)

    last_bidir_layer = tf.keras.layers.Bidirectional(
        rnn_layer(rnn_layers_sizes[-1]),
        name=f'{model_name}_{DEFAULT_RNN_LAYER_NAME}_{len(rnn_layers_sizes)}'
    )
    bidir_tensor: tf.Tensor = last_bidir_layer(bidir_tensor)
    feature_layer: tf.keras.layers.Layer = tf.keras.layers.Lambda(
        lambda x: x,
        name=f'{model_name}_{DEFAULT_FEATURES_LAYER_NAME}'
    )
    features: tf.Tensor = feature_layer(bidir_tensor)

    classifier = create_classifier(
        model_name=model_name,
        input_dim=features.shape[1],
        dense_layers=dense_layers,
        no_labels=no_labels
    )

    output = classifier(features)

    return tf.keras.Model(inputs=input_text, outputs=output)
