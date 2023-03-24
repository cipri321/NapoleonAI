from typing import Optional, Tuple, Iterable

import gensim.models
import numpy as np
from gensim.models import Doc2Vec

from settings import DEFAULT_TEXT_INPUT_RAW_SUFFIX, DEFAULT_TEXT_ENCODING_LAYER_SUFFIX, DEFAULT_FEATURES_LAYER_NAME
from textclassification.common.classifier import create_classifier
from textclassification.common.common_types import Vocabulary
from textclassification.doc2vec.dataprocessing.datapreprocessing import process_text
from textclassification.doc2vec.dataprocessing.datasetgeneration import get_tagged_documents_from_text_dataset
from textclassification.common.wordembeddings import get_embedding_layer_from_weights
import tensorflow as tf


def get_doc2vec(
        method: str,
        vector_size: int,
        train_dataset: np.ndarray,
        save_path: Optional[str],
        load_path: Optional[str],
        already_tokenised: bool
) -> gensim.models.Doc2Vec:
    """
    Returns a doc2vec gensim model trained on a train dataset
    It can be either a new model or an existing model further trained on train dataset

    :param method: dm or
    :param vector_size: size of the embedding vector for each document
    :param train_dataset: array of arrays of strings or array of strings
    :param save_path: str - path where the gensim model should be saved
    :param load_path: str - path of a gensim model to be further trained
    :param already_tokenised: true - if the docs are already tokenised, false - otherwise
    :return: trained Doc2vec model
    """
    tagged_docs = None
    if isinstance(train_dataset, Iterable) and train_dataset.any():
        tagged_docs = get_tagged_documents_from_text_dataset(text_dataset=train_dataset,
                                                             already_tokenised=already_tokenised)
    if load_path:
        doc2vec = Doc2Vec.load(load_path)
        if tagged_docs:
            doc2vec.train(tagged_docs, epochs=30, total_examples=doc2vec.corpus_count)
    else:
        doc2vec = Doc2Vec(tagged_docs, dm=1, vector_size=vector_size, window=10, min_count=1, workers=5, epochs=30)
    if save_path:
        doc2vec.save(save_path)
    return doc2vec


def get_doc2vec_embedding_layer(
        doc2vec_model: Doc2Vec,
        original_dataset: np.ndarray,
        vector_size: int,
        layer_name: Optional[str],
        already_tokenised: bool
) -> Tuple[tf.keras.layers.Embedding, Vocabulary]:
    """
    Creates a keras embedding layer and a vocabulary consisting of available documents
    NO NEW(NOT LEARNT) DOCUMENTS CAN BE EVALUATED USING THIS LAYER

    :param doc2vec_model: gensim Doc2vec model
    :param original_dataset: the text dataset from which documents can be learnt
    :param vector_size: int, size of the document embeddings
    :param layer_name: str, name of the keras embedding layer
    :param already_tokenised: true or false, if the original_dataset is already tokenised
    :return: keras embedding layer and its corresponding vocabulary
    """
    vocab: Vocabulary = []
    freqs = dict()
    to_delete = []
    keyed_vectors = doc2vec_model.dv

    for idx, doc in enumerate(original_dataset):
        if not already_tokenised:
            processed_doc = process_text(doc)
        else:
            processed_doc = ' '.join(doc)
        if idx != keyed_vectors.index_to_key[idx]:
            raise ValueError(f"Something wrong, {processed_doc} wasn't processed correctly")

        if processed_doc == '':
            to_delete.append(idx)
            continue

        if processed_doc in freqs:
            freqs[processed_doc].append(idx)
            to_delete.append(idx)
        else:
            freqs[processed_doc] = [idx]
            vocab.append(processed_doc)

    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    new_weights = []

    for doc in vocab:
        new_weights.append(weights[freqs[doc][0]])

    weights = np.array(new_weights)

    layer = get_embedding_layer_from_weights(weights, layer_name)

    return layer, vocab


def get_doc2vec_classifier(
        model_name: str,
        vocab_dataset: np.ndarray,
        already_tokenised: bool,
        doc2vec_save_path: Optional[str],
        load_path: Optional[str],
        dense_layers: Tuple[int, ...],
        no_labels: int
):
    """
    Creates a text classification model that extracts features using the doc2vec algorithm

    :param model_name: str - name of the model and underlying layers
    :param vocab_dataset: - an array of strings or an array of arrays of strings
    :param already_tokenised:
    true - vocab dataset is an array of arrays of strings
    false - vocab dataset is an array of strings(sentences)
    :param doc2vec_save_path: str - path where the gensim model should be saved
    :param load_path: str - path where a pretrained gensim model exists
    :param dense_layers: List of int - number of neurons for each dense layer
    :param no_labels: int - number of layers
    :return: Keras Model used to classify text
    """
    doc2vec = get_doc2vec(
        'dm',
        vector_size=256,
        train_dataset=vocab_dataset,
        save_path=doc2vec_save_path,
        load_path=load_path,
        already_tokenised=already_tokenised
    )

    input_text = tf.keras.layers.Input(
        shape=(),
        dtype=tf.string,
        name=f'{model_name}_{DEFAULT_TEXT_INPUT_RAW_SUFFIX}'
    )

    embedding_layer, vocab = get_doc2vec_embedding_layer(doc2vec, vocab_dataset, 256, f'{model_name}_embedding_layer',
                                                         already_tokenised=already_tokenised)
    embedding_layer.trainable = False

    encoding_layer = tf.keras.layers.TextVectorization(
        name=f'{model_name}_{DEFAULT_TEXT_ENCODING_LAYER_SUFFIX}',
        vocabulary=vocab,
        split=None
    )

    encoded_tensor: tf.Tensor = encoding_layer(input_text)

    embedding_tensor: tf.Tensor = embedding_layer(encoded_tensor)

    feature_layer: tf.keras.layers.Layer = tf.keras.layers.Lambda(
        lambda x: x,
        name=f'{model_name}_{DEFAULT_FEATURES_LAYER_NAME}'
    )
    features: tf.Tensor = feature_layer(embedding_tensor)

    classifier = create_classifier(
        model_name=model_name,
        input_dim=features.shape[1],
        dense_layers=dense_layers,
        no_labels=no_labels
    )

    output = classifier(features)

    return tf.keras.Model(inputs=input_text, outputs=output)
