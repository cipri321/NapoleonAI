from typing import Dict
import tensorflow as tf

from napoleonai.controller.image_model import create_image_model
from napoleonai.controller.multimodal_model import create_multimodal_classification_model
from napoleonai.controller.text_model import create_text_model


def create_model(configuration: Dict) -> tf.keras.Model:
    """
    Creates a tensorflow model based on the configuration object

    :parameter configuration: dictionary
    :return: Keras model based on configuration
    """
    if 'task' not in configuration:
        raise ValueError('There should be a task in the configuration')
    if configuration['task'] == 'text classification':
        return create_text_model(configuration)
    elif configuration['task'] == 'image classification':
        return create_image_model(configuration)
    elif configuration['task'] == 'multimodal classification':
        return create_multimodal_classification_model(configuration)
    raise ValueError('No task with that name')
