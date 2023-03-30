import pandas as pd
import tensorflow as tf
from nltk.stem.snowball import FrenchStemmer

from napoleonai.common.categorymapping import get_category_mapping_from_file
from napoleonai.textclassification.common.datapreprocessing import preprocess_csv
from napoleonai.textclassification.common.featureextraction import get_feature_extractor_from_model
from napoleonai.multimodalclassification.model.model import get_multimodal_classification_model
from napoleonai.multimodalclassification.dataprocessing.datasetgeneration import create_multimodal_dataset


def multimodal_experiment():
    stem_function = FrenchStemmer().stem

    cat_map, no_labels = get_category_mapping_from_file(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map')

    description_model = tf.keras.models.load_model('saved/models/rnn/description_word2vec_256_rnn')
    title_model = tf.keras.models.load_model('saved/models/rnn/title_word2vec_256_rnn')

    title_feature_extractor = get_feature_extractor_from_model(title_model)
    description_feature_extractor = get_feature_extractor_from_model(description_model)

    multi = get_multimodal_classification_model(
        [title_feature_extractor, description_feature_extractor],
        [],
        27
    )

    df_train = pd.read_csv('/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv')

    df_test = pd.read_csv('/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv')

    df_text_train = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv',
        concat_columns=True,
        category_column='Category',
        text_columns=['Description'],
        category_map=cat_map,
        stem_function=stem_function
    )
    df_train['Description'] = df_text_train['Text']

    df_text_test = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv',
        concat_columns=True,
        category_column='Category',
        text_columns=['Description'],
        category_map=cat_map,
        stem_function=stem_function
    )
    df_test['Description'] = df_text_test['Text']

    df_text_train = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv',
        concat_columns=True,
        category_column='Category',
        text_columns=['Title'],
        category_map=cat_map,
        stem_function=stem_function
    )
    df_train['Title'] = df_text_train['Text']

    df_text_test = preprocess_csv(
        '/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv',
        concat_columns=True,
        category_column='Category',
        text_columns=['Title'],
        category_map=cat_map,
        stem_function=stem_function
    )
    df_test['Title'] = df_text_test['Text']


    dataset_train = create_multimodal_dataset(
        df_train,
        text_columns=['Title', 'Description'],
        image_columns=[],
        category_column='Category',
        category_map=cat_map,
        one_hot=True,
        images_path='/Users/cipri/Downloads/archive (5)/images/images/image_train'
    )

    dataset_test = create_multimodal_dataset(
        df_test,
        text_columns=['Title', 'Description'],
        image_columns=[],
        category_column='Category',
        one_hot=True,
        category_map=cat_map,
        images_path='/Users/cipri/Downloads/archive (5)/images/images/image_train'
    )

    multi.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        loss='categorical_crossentropy',
        run_eagerly=True
    )
    multi.summary()

    # train_tf_data = tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(data) for data in dataset_train[0]]))
    # train_tf_labels = tf.data.Dataset.from_tensor_slices(dataset_train[1])
    # train_tf = tf.data.Dataset.zip(train_tf_data, train_tf_labels)

    multi.fit(
        dataset_train,
        epochs=5,
        workers=5,
        use_multiprocessing=True
    )

    multi.evaluate(dataset_test)
