from typing import Tuple, List

import tensorflow as tf
from dataclasses import dataclass, field
from settings import DEFAULT_IMAGE_SIZE, DEFAULT_NO_CLASSES, DEFAULT_NO_CHANNELS, DEFAULT_IMAGE_INPUT_RAW_SUFFIX
from .image_types import ModelInstantiator, DataPreprocessor, ModelConfiguration
from .image_models_dictionary import CONF


@dataclass
class TrainableModel:
    name: str
    model_instantiator: ModelInstantiator
    data_preprocessor: DataPreprocessor
    image_size: int = DEFAULT_IMAGE_SIZE
    no_classes: int = DEFAULT_NO_CLASSES
    train_layers_keywords: Tuple[str, ...] = field(default_factory=tuple)
    is_pretrained: bool = False

    def get_model(self) -> tf.keras.Model:
        """
        Creates a keras Model using the architecture specified by model_instantiator, creates input and preprocesses it
        It can use pretrained models and specify which layers have to be frozen

        :return: keras Model based in this object's configuration
        """

        raw_input = tf.keras.layers.Input(
            shape=[self.image_size, self.image_size, 3],
            dtype=tf.uint8,
            name=f'{self.name}_{DEFAULT_IMAGE_INPUT_RAW_SUFFIX}'
        )

        x = tf.cast(raw_input, tf.float32, name=f'{self.name}_cast_image_input')

        preprocessed_data = self.data_preprocessor(x)

        base = self.model_instantiator(
            # name=self.name,
            include_top=False,
            weights='imagenet' if self.is_pretrained else None,
            input_shape=(self.image_size, self.image_size, 3),
            pooling='avg',
        )

        if self.is_pretrained:
            for layer in base.layers:
                should_train = False
                for keyword in self.train_layers_keywords:
                    if keyword in layer.name:
                        should_train = True
                        break
                layer.trainable = should_train

        features: tf.Tensor = base(preprocessed_data)
        feature_layer = tf.keras.layers.Lambda(lambda x: x, name=f'{self.name}_features')(features)

        classifier_input = tf.keras.Input(
            shape=features.shape[1:],
            name=f'{self.name}_classifier_input'
        )

        classifier_output = tf.keras.layers.Dense(
            self.no_classes,
            activation='softmax',
            name=f'{self.name}_classifier_output')(classifier_input)

        classifier = tf.keras.Model(inputs=classifier_input, outputs=classifier_output, name=f'{self.name}_classifier')

        model_output = classifier(feature_layer)
        new_model = tf.keras.Model(inputs=raw_input, outputs=model_output)

        return new_model


@dataclass
class ModelWithInfo:
    model: tf.keras.Model
    name: str


def create_models(image_size: int, no_classes: int, models: Tuple[ModelConfiguration, ...])\
        -> List[ModelWithInfo]:
    """
    Creates models based on ModelConfigurations

    :param image_size: int
    :param no_classes: int
    :param models: tuple of ModelConfigurations
    :return: List of created Keras models and their info(ModelWithInfo type)
    """
    common_args = {
        'image_size': image_size,
        'no_classes': no_classes,
    }
    models: List[ModelWithInfo] = [
        ModelWithInfo(
            model=TrainableModel(
                name=model_conf.name,
                model_instantiator=CONF[model_conf.model_type_name][0],
                data_preprocessor=CONF[model_conf.model_type_name][1],
                is_pretrained=model_conf.is_pretrained,
                train_layers_keywords=model_conf.train_layers_keywords,
                **common_args
            ).get_model(),
            name=model_conf.name
        ) for model_conf in models
    ]
    return models
