import tensorflow as tf

from settings import DEFAULT_FEATURES_LAYER_NAME


def erase_classifier_layer(model: tf.keras.Model) -> tf.keras.Model:
    """
    Erases the classifier functional layer from a keras model

    :param model: image classification keras model
    :return: keras model without the classification layer
    """
    if 'classifier' not in model.layers[-1].name:
        raise ValueError('The last layer is not classifier')

    feature_layer = None
    for layer in model.layers:
        if 'features' in layer.name:
            feature_layer = layer
            break
    if not feature_layer:
        raise ValueError('There is no feature layer in the model')
    return tf.keras.Model(inputs=model.input, outputs=feature_layer.output)


def get_feature_extractor_from_saved_model(savepath: str) -> tf.keras.Model:
    """
    Returns a model without the classifier functional layer

    :param savepath: str, path to a saved keras model
    :return: keras model that doesn't contain a classifier layer
    """
    model = tf.keras.models.load_model(savepath)
    return erase_classifier_layer(model)
