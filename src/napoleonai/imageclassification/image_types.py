import typing
from dataclasses import dataclass
from typing import Callable, List, Union, Protocol, Tuple

import numpy as np

from napoleonai.settings import DEFAULT_IMAGE_SIZE, DEFAULT_NO_CLASSES, DEFAULT_NO_CHANNELS
import tensorflow as tf


class ModelInstantiator(Protocol):
    def __call__(
            self,
            name: str,
            include_top: bool = True,
            weights: Union[str, None] = None,
            input_shape: Tuple = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, DEFAULT_NO_CHANNELS),
            pooling: str = 'avg',
            classes: int = DEFAULT_NO_CLASSES
    ) -> tf.keras.Model:
        ...


class DataPreprocessor(Protocol):
    def __call__(
            self,
            name: str,
            data: Union[np.ndarray, tf.Tensor]
    ) -> Union[np.ndarray, tf.Tensor]:
        ...


ModelDataPreprocessorPair = typing.NewType('ModelDataPreprocessorPair', Tuple[tf.keras.Model, DataPreprocessor])


@dataclass
class ModelConfiguration:
    name: str
    model_type_name: str
    is_pretrained: bool
    train_layers_keywords: Tuple[str, ...]
