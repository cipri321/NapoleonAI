import gensim.models
import numpy as np
from gensim.models import Word2Vec, Doc2Vec, FastText
import tensorflow as tf
from typing import Optional, List, Dict, Tuple, Callable
import logging
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from textclassification.common.common_types import GensimEmbeddingLayerGenerator, Vocabulary
from settings import DEFAULT_WORD2VEC_WINDOW_SIZE, DEFAULT_WORD2VEC_MIN_COUNT, DEFAULT_NO_WORKERS, \
    DEFAULT_WORD2VEC_EMBEDDING_SIZE, DEFAULT_WORD2VEC_SAVE_PATH, DEFAULT_WORD2VEC_NO_EPOCHS, \
    DEFAULT_FASTTEXT_EMBEDDING_SIZE, DEFAULT_FASTTEXT_SAVE_PATH, DEFAULT_FASTTEXT_NO_EPOCHS, \
    DEFAULT_FASTTEXT_WINDOW_SIZE, DEFAULT_FASTTEXT_MIN_COUNT, DEFAULT_EMBEDDING_NAME


def get_list_of_sentences_from_txt_file(filepath: str, stem_function: Callable[[str], str]) -> List[List[str]]:
    """
    Considers that each line in file located at filepath is a sentence,
    parses(lowercase and splits at space) and stems each word

    :param filepath: str, location where the file of the corpus is
    :param stem_function: function that stems a word
    :return: a list of lists of words -
    outer list is the sentences, each list inside it is a parsed and processed sentence
    """
    res = []

    def process_function(x: str):
        return x.strip().lower() if not stem_function else stem_function(x.strip().lower())

    with open(filepath, 'rt') as f:
        lines = f.readlines()
        for sentence in tqdm(lines):
            res.append([process_function(word) for word in sentence.split()])
    return res


def get_embedding_layer_and_vocab(
        model_type: str,
        pretrained_model_path: Optional[str],
        train_corpus_path: Optional[str],
        layer_name: Optional[str],
        vector_size: Optional[int],
        save_path: Optional[str],
        stem_function: Optional[Callable[[str], str]]
) -> Tuple[tf.keras.layers.Embedding, Vocabulary]:
    """
    Creates an embedding layer and Vocabulary based on a pretrained model and/or a train corpus path
    (each sentence/separate context is on a new line) using a word embedding learning algo(Word2Vec/Fasttext)
    If both pretrained_model_path and train_corpus_path are provided, then the gensim model will be further
    trained using the file provided

    :param model_type: str - word2vec or fasttext
    :param pretrained_model_path: str or None - path to a saved gensim model
    :param train_corpus_path: str - path to a text file where each sentence(separate context) is on a new line
    :param layer_name: str - name for the embedding layer
    :param vector_size: int - size of the word embeddings
    :param save_path: str - path where the gensim model is saved
    :param stem_function: function to get the stem of a word
    :return: Keras Embedding layer and corresponding vocabulary
    """
    sentences = None
    if train_corpus_path:
        logging.debug('get_embedding_layer_and_vocab called with a training corpus file')
        sentences = get_list_of_sentences_from_txt_file(train_corpus_path, stem_function=stem_function)
        logging.info('training corpus file processed')

    layer_name = layer_name if layer_name else DEFAULT_EMBEDDING_NAME

    word_embedding_generator_functions: Dict[str, GensimEmbeddingLayerGenerator] = {
        "word2vec": get_gensim_word2vec_embedding_layer,
        "fasttext": get_gensim_fasttext_embedding_layer,
    }

    default_word_embedding_vector_size = {
        "word2vec": DEFAULT_WORD2VEC_EMBEDDING_SIZE,
        "fasttext": DEFAULT_FASTTEXT_EMBEDDING_SIZE
    }

    default_model_save_path = {
        "word2vec": DEFAULT_WORD2VEC_SAVE_PATH,
        "fasttext": DEFAULT_FASTTEXT_SAVE_PATH
    }

    return word_embedding_generator_functions[model_type](
        pretrained_model_path,
        sentences,
        layer_name,
        vector_size if vector_size else default_word_embedding_vector_size[model_type],
        save_path if save_path else f'{default_model_save_path[model_type]}/{layer_name}'
    )


def get_embedding_layer_from_weights(
        weights: np.ndarray,
        layer_name: str
) -> tf.keras.layers.Embedding:
    """
    Get embedding layer from a weights ~matrix~: each line is an embedding vector

    :param weights: np.ndarray - an array of embedding vectors
    :param layer_name: str - the name of the Keras embedding layer
    :return: Keras embedding layer
    """
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

    # Text vectorization layer adds two values at the beginning of the vocabulary '' and '[UNK]'
    print(weights.shape)
    emptyvalue = initializer(shape=(weights.shape[1],)).numpy()
    unknownvalue = initializer(shape=(weights.shape[1],)).numpy()

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0] + 2,
        output_dim=weights.shape[1],
        weights=[np.concatenate([[emptyvalue], [unknownvalue], weights])],
        trainable=True,
        name=layer_name
    )
    return layer


def get_embedding_and_vocabulary_from_keyed_vectors(
        keyed_vectors: gensim.models.KeyedVectors,
        layer_name: str
) -> Tuple[tf.keras.layers.Embedding, Vocabulary]:
    """
    Creates a Keras embedding layer and a Vocabulary from a Gensim KeyedVectors object

    :param keyed_vectors: Gensim KeyedVectors object
    :param layer_name: str - name for the embedding layer
    :return: Keras Embedding layer and corresponding vocabulary
    """
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    return get_embedding_layer_from_weights(weights, layer_name), index_to_key


def get_gensim_word2vec_embedding_layer(
        pretrained_model_path: Optional[str],
        sentences: Optional[List[List[str]]],
        layer_name: Optional[str],
        vector_size: int,
        save_path: Optional[str]
) -> Tuple[tf.keras.layers.Embedding, Vocabulary]:
    """
    Returns an Embedding layer and corresponding vocabulary using the Word2Vec algorithm,
    based on a pretrained gensim model and/or a new corpus(sentences)

    :param pretrained_model_path: str - path to a file containing a training corpus
    :param sentences: List of sentences, which are a list of words
    :param layer_name: str - name of the embedding layer
    :param vector_size: int
    :param save_path: str - path where the gensim model will be saved
    :return: Keras embedding layer and corresponding Vocabulary
    """
    if not pretrained_model_path and not sentences:
        raise ValueError('There should be either a pretrained word2vec model or training data')
    if pretrained_model_path:
        logging.debug('Loading a pretrained word2vec model')
        word2vec = Word2Vec.load(pretrained_model_path)
        logging.info(f'Loaded word2vec model from {pretrained_model_path}')
        if sentences:
            logging.debug('further training a word2vec model')
            word2vec.train(sentences, total_examples=len(sentences), epochs=DEFAULT_WORD2VEC_NO_EPOCHS)
            logging.info('Further trained word2vec model')
    else:
        logging.debug('Started training a new word2vec model')
        word2vec = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=DEFAULT_WORD2VEC_WINDOW_SIZE,
            min_count=DEFAULT_WORD2VEC_MIN_COUNT,
            workers=DEFAULT_NO_WORKERS,
            epochs=DEFAULT_WORD2VEC_NO_EPOCHS
        )
        logging.info('Trained a new word2vec model')

    if save_path:
        word2vec.save(save_path)
        logging.info('saved word2vec model')

    return get_embedding_and_vocabulary_from_keyed_vectors(
        keyed_vectors=word2vec.wv,
        layer_name=layer_name
    )


def get_gensim_fasttext_embedding_layer(
        pretrained_model_path: Optional[str],
        sentences: Optional[List[List[str]]],
        layer_name: Optional[str],
        vector_size: int,
        save_path: Optional[str]
) -> Tuple[tf.keras.layers.Embedding, Vocabulary]:
    """
    Returns an Embedding layer and corresponding vocabulary using the Fasttext algorithm,
    based on a pretrained gensim model and/or a new corpus(sentences)

    :param pretrained_model_path: str - path to a file containing a training corpus
    :param sentences: List of sentences, which are a list of words
    :param layer_name: str - name of the embedding layer
    :param vector_size: int
    :param save_path: str - path where the gensim model will be saved
    :return: Keras embedding layer and corresponding Vocabulary
    """
    if not pretrained_model_path and not sentences:
        raise ValueError('There should be either a pretrained fasttext model or training data')
    if pretrained_model_path:
        fasttext = FastText.load(pretrained_model_path)
        if sentences:
            fasttext.train(sentences, total_examples=len(sentences), epochs=DEFAULT_FASTTEXT_NO_EPOCHS)
    else:
        fasttext = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=DEFAULT_FASTTEXT_WINDOW_SIZE,
            min_count=DEFAULT_FASTTEXT_MIN_COUNT,
            workers=DEFAULT_NO_WORKERS,
            epochs=DEFAULT_FASTTEXT_NO_EPOCHS
        )

    if save_path:
        fasttext.save(save_path)

    return get_embedding_and_vocabulary_from_keyed_vectors(
        keyed_vectors=fasttext.wv,
        layer_name=layer_name
    )
