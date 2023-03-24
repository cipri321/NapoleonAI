import numpy as np
import tensorflow as tf
from textclassification.common.datapreprocessing import *
from textclassification.common.datasetgeneration import get_dataset_from_df
from textclassification.common.wordembeddings import get_embedding_layer_and_vocab
from textclassification.rnn.model.rnn import get_rnn_for_text_classification
from common.categorymapping import write_category_mapping_to_file, get_category_mapping_from_file
from nltk.stem.snowball import FrenchStemmer
import pandas as pd


def rnn_word_embedding_experiment():

    stem_function = FrenchStemmer().stem

    embedding_layer, vocab = get_embedding_layer_and_vocab(
        pretrained_model_path='/Users/cipri/PycharmProjects/coreLicenta/saved/models/word2vec/french_rakuten_256',
        model_type='word2vec',
        train_corpus_path=None,
        layer_name='rnn_word2vec_256_description_embedding_layer',
        vector_size=256,
        save_path='saved/models/word2vec/french_rakuten_256',
        stem_function=stem_function
    )

    cat_map, no_labels = get_category_mapping_from_file('/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map')
    df_train = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv',
        concat_columns=True,
        category_column='Category',
        text_columns=['Title', 'Description'],
        category_map=cat_map,
        stem_function=stem_function
    )

    df_test = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv',
        concat_columns=True,
        category_column='Category',
        text_columns=['Title', 'Description'],
        category_map=cat_map,
        stem_function=stem_function
    )

    train_text_dataset, train_label_dataset = get_dataset_from_df(df_train, 'Text', 'Category', one_hot_label=True)
    test_text_dataset, test_label_dataset = get_dataset_from_df(df_test, 'Text', 'Category', one_hot_label=True)

    print(train_text_dataset[:10])

    model = get_rnn_for_text_classification(
        model_save_path=None,
        model_name='rnn_title_description_word2vec_256',
        pretrained_word_embeddings=(embedding_layer, vocab),
        word_embedding_size=256,
        vocabulary_dataset=None,
        rnn_layers_sizes=(256, 128),
        rnn_cell_type='lstm',
        dense_layers=(256, 128),
        no_labels=27
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        loss='categorical_crossentropy'
    )

    model.summary()

    model.fit(
        train_text_dataset,
        train_label_dataset,
        batch_size=128,
        validation_split=0.1,
        epochs=10,
        workers=5,
        use_multiprocessing=True
    )

    model.evaluate(test_text_dataset, test_label_dataset)

    model.save('saved/models/rnn/title_description_word2vec_256_rnn')
