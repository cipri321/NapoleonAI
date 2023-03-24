from typing import Dict, List

from common.validation_functions import validate_configuration
from imageclassification.image_types import ModelConfiguration
from imageclassification.train_models import get_models


def validate_image_configuration(configuration):
    mandatory_data = {
        'data': Dict,
        'model_name': str,
        'type': str,
        'is_pretrained': bool,
        'train_layers_keywords': List
    }
    validate_configuration(configuration, mandatory_data)


def validate_image_data(configuration):
    mandatory_data = {
        'image_size': int,
        'train_folder': str
    }
    validate_configuration(configuration, mandatory_data)


def create_image_model(configuration):
    """
    Create a keras model used for image classification based on a configuration file
    and trains it for classification task

    :param configuration: dictionary
    :return: None
    """
    validate_image_configuration(configuration)
    validate_image_data(configuration['data'])

    get_models(
        configuration['data']['img_size'],
        configuration['data']['train_folder'],
        (ModelConfiguration(
            name=configuration['model_name'],
            model_type_name=configuration['type'],
            is_pretrained=configuration['is_pretrained'],
            train_layers_keywords=configuration['train_layers_keywords']
        ),)
    )