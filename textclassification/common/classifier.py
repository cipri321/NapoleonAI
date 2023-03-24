from typing import Tuple, Optional

import tensorflow as tf

from settings import DEFAULT_CLASSIFIER_NAME, DEFAULT_CLASSIFIER_INPUT, DEFAULT_CLASSIFIER_ACTIVATION_FUNCTION, \
    DEFAULT_CLASSIFIER_DENSE_LAYER, DEFAULT_CLASSIFIER_SOFTMAX_LAYER, DEFAULT_CLASSIFIER_FUNCTIONAL_LAYER


def create_classifier(
        model_name: Optional[str],
        input_dim: int,
        dense_layers: Tuple[int],
        no_labels: int
) -> tf.keras.Model:
    """
    Creates a keras Model that can be used as a classifier on top of a feature extractor

    :param model_name: str - name of the classifier and its layers
    :param input_dim: int - number of input features
    :param dense_layers: List of int - number of neurons in each dense layer
    :param no_labels: int - number of labels, also the number of neurons of a softmax layer(last layer)
    :return: Keras model used of classification
    """

    if not model_name:
        model_name = DEFAULT_CLASSIFIER_NAME

    classifier_input = tf.keras.Input(
        shape=(input_dim,),
        name=f'{model_name}_{DEFAULT_CLASSIFIER_INPUT}'
    )

    prev_tensor = classifier_input
    for i in range(len(dense_layers)):
        dense_layer = tf.keras.layers.Dense(
            dense_layers[i],
            activation=DEFAULT_CLASSIFIER_ACTIVATION_FUNCTION,
            name=f'{model_name}_{DEFAULT_CLASSIFIER_DENSE_LAYER}_{i}')
        prev_tensor = dense_layer(prev_tensor)

    softmax_layer = tf.keras.layers.Dense(
        no_labels,
        activation='softmax',
        name=f'{model_name}_{DEFAULT_CLASSIFIER_SOFTMAX_LAYER}')

    output = softmax_layer(prev_tensor)

    return tf.keras.Model(
        inputs=classifier_input,
        outputs=output,
        name=f'{model_name}_{DEFAULT_CLASSIFIER_FUNCTIONAL_LAYER}'
    )
