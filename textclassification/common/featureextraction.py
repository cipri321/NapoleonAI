import tensorflow as tf
from settings import DEFAULT_CLASSIFIER_FUNCTIONAL_LAYER, DEFAULT_FEATURES_LAYER_NAME


def get_feature_extractor_from_model(
        model: tf.keras.Model
) -> tf.keras.Model:
    """
    Creates a model that doesn't contain the functional layer called classifier

    :param model: Keras model that should contain a functional layer called classifier
    :return: Keras model without the classifier
    """
    if DEFAULT_CLASSIFIER_FUNCTIONAL_LAYER not in model.layers[-1].name:
        raise ValueError('There is no classifier in this model')
    feature_layer = None
    for layer in model.layers:
        if DEFAULT_FEATURES_LAYER_NAME in layer.name:
            feature_layer = layer
            break
    if not feature_layer:
        raise ValueError('There is no feature layer in the model')
    return tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
