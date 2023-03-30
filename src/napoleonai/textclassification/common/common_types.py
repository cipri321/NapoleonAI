from typing import Protocol, Union, Tuple, Optional, List, Dict, NewType
import tensorflow as tf

Vocabulary = List[str]


class GensimEmbeddingLayerGenerator(Protocol):
    def __call__(
            self,
            pretrained_model_path: Optional[str],
            sentences: Optional[List[List[str]]],
            layer_name: Optional[str],
            vector_size: int,
            save_path: Optional[str]
    ) -> Tuple[tf.keras.layers.Embedding, Vocabulary]:
        ...
