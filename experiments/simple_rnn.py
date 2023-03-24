import numpy as np
import tensorflow as tf
from nltk.stem.snowball import FrenchStemmer

from common.categorymapping import get_category_mapping_from_file
from textclassification.common.datapreprocessing import *
from textclassification.common.datasetgeneration import get_dataset_from_df
from textclassification.rnn.model.rnn import get_rnn_for_text_classification
import pandas as pd


def simple_rnn_experiment():
    stem_function = FrenchStemmer().stem

    cat_map, no_labels = get_category_mapping_from_file(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map')

    df = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten.csv',
        concat_columns=False,
        category_column='Category',
        text_columns=['Description'],
        category_map=cat_map,
        stem_function=stem_function
    )

    vocab_dataset, _ = get_dataset_from_df(df, 'Text', 'Category', one_hot_label=True)


    df_train = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv',
        concat_columns=False,
        category_column='Category',
        text_columns=['Description'],
        category_map=cat_map,
        stem_function=stem_function
    )

    df_test = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv',
        concat_columns=False,
        category_column='Category',
        text_columns=['Description'],
        category_map=cat_map,
        stem_function=stem_function
    )

    train_text_dataset, train_label_dataset = get_dataset_from_df(df_train, 'Text', 'Category', one_hot_label=True)
    test_text_dataset, test_label_dataset = get_dataset_from_df(df_test, 'Text', 'Category', one_hot_label=True)

    model = get_rnn_for_text_classification(
        model_save_path=None,
        model_name='simple_rnn',
        pretrained_word_embeddings=None,
        word_embedding_size=128,
        vocabulary_dataset=vocab_dataset,
        rnn_layers_sizes=(64, 64),
        rnn_cell_type='lstm',
        dense_layers=(32,),
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
        validation_split=0.15,
        epochs=10,
        workers=5,
        use_multiprocessing=True
    )

    model.evaluate(test_text_dataset, test_label_dataset)

    model.save('saved/models/rnn/simple_rnn_description')