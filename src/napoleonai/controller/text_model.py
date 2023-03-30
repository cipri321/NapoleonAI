from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np
import tensorflow as tf

from napoleonai.common.categorymapping import get_category_mapping_from_file
from napoleonai.common.stem_function import get_stem_function
from napoleonai.common.validation_functions import validate_configuration
from napoleonai.textclassification.common.datapreprocessing import preprocess_csv
from napoleonai.textclassification.common.datasetgeneration import get_dataset_from_df
from napoleonai.textclassification.common.wordembeddings import get_embedding_layer_and_vocab
from napoleonai.textclassification.doc2vec.models.doc2vec import get_doc2vec_classifier
from napoleonai.textclassification.rnn.model.rnn import get_rnn_for_text_classification


def validate_data(data_conf: Dict):
    if 'training_file' not in data_conf and 'test_file' not in data_conf:
        raise ValueError('either training or test should be in data configuration')

    mandatory_data = {
        'concat_columns': bool,
        'category_column': str,
        'text_columns': Iterable,
        'language': str
    }
    validate_configuration(data_conf, mandatory_data)


def create_datasets_from_conf(configuration: Dict) -> \
        Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create tuple of train, test datasets from configuration object

    :param configuration: dict
    :return: (train, test) -> any one can be None
    """
    if 'data' not in configuration:
        raise ValueError('There should be data in the configuration')
    cat_map = None
    if 'category_map' in configuration['data']:
        cat_map, no_labels = get_category_mapping_from_file(configuration['data']['category_map'])
    validate_data(configuration['data'])
    df_train, df_test = None, None
    if 'training_file' in configuration['data']:
        df_train = preprocess_csv(
            filename=configuration['data']['training_file'],
            concat_columns=configuration['data']['concat_columns'],
            category_column=configuration['data']['category_column'],
            text_columns=configuration['data']['text_columns'],
            category_map=cat_map,
            stem_function=get_stem_function(configuration['data']['language'])
        )
    if 'test_file' in configuration['data']:
        df_test = preprocess_csv(
            filename=configuration['data']['test_file'],
            concat_columns=configuration['data']['concat_columns'],
            category_column=configuration['data']['category_column'],
            text_columns=configuration['data']['text_columns'],
            category_map=cat_map,
            stem_function=get_stem_function(configuration['data']['language'])
        )

    train_text_dataset, train_label_dataset = get_dataset_from_df(df_train, 'Text', 'Category', one_hot_label=True)
    test_text_dataset, test_label_dataset = get_dataset_from_df(df_test, 'Text', 'Category', one_hot_label=True)

    return (train_text_dataset, train_label_dataset), (test_text_dataset, test_label_dataset)


def check_pretrained_word_embeddings(configuration: Dict):
    """
    Checks the configuration for pretrained word embeddings is valid

    :param configuration: Dict
    :return: -
    """
    if not 'pretrained_gensim_model' and not 'train_corpus_path':
        raise ValueError('Either a pretrained model or a train corpus should be given')

    mandatory_data = {
        'word_embeddings': str,
        'word_embedding_size': int,
        'word_embedding_save_path': str,
        'language': (str, '; if there are multiple languages present, write "ANY"')
    }
    validate_configuration(configuration, mandatory_data)


def validate_rnn_model_layers(configuration):
    mandatory_data = {
        'rnn_layer_sizes': List,
        'dense_layer_sizes': List,
        'rnn_cell': str,
    }
    validate_configuration(configuration, mandatory_data)


def validate_doc2vec_layers(configuration):
    mandatory_data = {
        'dense_layer_sizes': List,
    }
    validate_configuration(configuration, mandatory_data)


def validate_compilation_data(configuration):
    mandatory_data = {
        'optimizer': str,
        'loss': str
    }
    validate_configuration(configuration, mandatory_data)


def validate_training_conf(configuration):
    mandatory_data = {
        'epochs': int,
        'batch_size': int
    }
    validate_configuration(configuration, mandatory_data)


def create_text_model(configuration: Dict) -> tf.keras.Model:
    """
    Creates a text classification model based on the configuration object

    :param configuration: dict
    :return: Keras model based on configuration
    """
    (train_text_dataset, train_label_dataset), (test_text_dataset, test_label_dataset) = create_datasets_from_conf(
        configuration)
    if 'type' not in configuration:
        raise ValueError('There should be a type in a text model configuration')
    if configuration['type'] == 'rnn':
        embedding_layer, vocab = None, None
        if 'word_embeddings' in configuration:
            check_pretrained_word_embeddings(configuration)
            embedding_layer, vocab = get_embedding_layer_and_vocab(
                pretrained_model_path=configuration['pretrained_gensim_model']
                if 'pretrained_gensim_model' in configuration else None,
                model_type=configuration['word_embeddings'],
                train_corpus_path=configuration['train_corpus_path']
                if 'train_corpus_path' in configuration else None,
                layer_name=f"{configuration['model_name']}_embedding_layer",
                vector_size=configuration['word_embedding_size'],
                stem_function=get_stem_function(configuration['language']),
                save_path=configuration['word_embedding_save_path']
            )
        vocabulary_dataset = None
        if train_text_dataset.any() and test_text_dataset.any():
            vocabulary_dataset = np.concatenate([train_text_dataset, test_text_dataset], axis=0)
        elif train_text_dataset:
            vocabulary_dataset = train_text_dataset

        validate_rnn_model_layers(configuration)
        model = get_rnn_for_text_classification(
            model_save_path=configuration['pretrained_model_path']
            if 'pretrained_model_path' in configuration else None,
            model_name=configuration['model_name'],
            pretrained_word_embeddings=(embedding_layer, vocab) if embedding_layer else None
            if 'pretrained_word_embeddings' in configuration else None,
            word_embedding_size=configuration['word_embedding_size'],
            vocabulary_dataset=vocabulary_dataset,
            rnn_layers_sizes=configuration['rnn_layer_sizes'],
            dense_layers=configuration['dense_layer_sizes'],
            rnn_cell_type=configuration['rnn_cell'],
            no_labels=configuration['data']['no_labels']
        )

    elif configuration['type'] == 'doc2vec':
        validate_doc2vec_layers(configuration)
        model = get_doc2vec_classifier(
            'experiment_doc2vec',
            np.concatenate([train_text_dataset, test_text_dataset], axis=0),
            doc2vec_save_path=None,
            load_path=None,
            dense_layers=configuration['dense_layer_sizes'],
            no_labels=configuration['data']['no_labels'],
            already_tokenised=False
        )

    else:
        raise ValueError("Unknown text model type")

    validate_compilation_data(configuration)
    model.compile(
        optimizer=configuration['optimizer'],
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        loss=configuration['loss']
    )

    model.summary()

    if 'training_file' in configuration['data']:
        validate_training_conf(configuration)
        model.fit(
            train_text_dataset,
            train_label_dataset,
            batch_size=configuration['batch_size'],
            validation_split=0.1,
            epochs=configuration['epochs'],
            workers=5,
            use_multiprocessing=True
        )

    if 'test_file' in configuration['data']:
        model.evaluate(test_text_dataset, test_label_dataset)

    if 'save_path' in configuration:
        model.save(configuration['save_path'])

    return model
