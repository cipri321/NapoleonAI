import numpy as np
import tensorflow as tf
from nltk.stem.snowball import FrenchStemmer

from common.categorymapping import get_category_mapping_from_file
from textclassification.common.datapreprocessing import preprocess_csv
from textclassification.common.datasetgeneration import get_dataset_from_df
from textclassification.common.wordembeddings import get_list_of_sentences_from_txt_file
from textclassification.doc2vec.models.doc2vec import get_doc2vec, get_doc2vec_embedding_layer, get_doc2vec_classifier
from textclassification.doc2vec.dataprocessing.datasetgeneration import get_tagged_documents_from_text_dataset

def doc2vec_experiment():
    stem_function = FrenchStemmer().stem

    cat_map, no_labels = get_category_mapping_from_file('/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map')
    df_train = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv',
        concat_columns=False,
        category_column='Category',
        text_columns=['Title'],
        category_map=cat_map,
        stem_function=stem_function
    )

    df_test = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv',
        concat_columns=False,
        category_column='Category',
        text_columns=['Title'],
        category_map=cat_map,
        stem_function=stem_function
    )

    train_text_dataset, train_label_dataset = get_dataset_from_df(df_train, 'Text', 'Category', one_hot_label=True)
    test_text_dataset, test_label_dataset = get_dataset_from_df(df_test, 'Text', 'Category', one_hot_label=True)
    doc2vec, encoding = get_doc2vec_classifier(
        'title_doc2vec',
        np.concatenate([train_text_dataset, test_text_dataset], axis=0),
        already_tokenised=False,
        doc2vec_save_path='saved/models/doc2vec/shitdoc2vec',
        load_path=None,
        dense_layers=(128, 128),
        no_labels=27
    )

    print(encoding.predict(test_text_dataset[:20]))

    doc2vec.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        loss='categorical_crossentropy'
    )

    doc2vec.summary()

    doc2vec.fit(
        train_text_dataset,
        train_label_dataset,
        batch_size=128,
        validation_split=0.1,
        epochs=10,
        workers=5,
        use_multiprocessing=True
    )

    doc2vec.evaluate(test_text_dataset, test_label_dataset)

    doc2vec.save('saved/models/doc2vec_classifier/title')