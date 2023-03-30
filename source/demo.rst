Demo
====

    This is an example of multimodal classification based on already existing text and image feature extractors


1. Code to use the run method of a Napoleon object with a yml configuration file

.. code-block:: python3

    from napoleon import Napoleon
    napoleon_instance = Napoleon()
    napoleon_instance.run('path_to_yml_configuration')


2. Configuration file example


.. code-block:: yaml

    task: multimodal classification
    data:
      training_file: PATH_TO_TRAINING_FILE
      test_file: PATH_TO_TEST_FILE
      inference_file: PATH_TO_INFERENCE_FILE
      inference_results: PATH_TO_INFERENCE_RESULTS
      category_column: Category
      category_map: PATH_TO_A_CATEGORY_MAPPING_FILE
      language: french
      images_path: ROOT_FOLDER_FOR_IMAGES
    modes:
      - model: PATH_TO_TENSORFLOW_TEXT_CLASSIFICATION_MODEL
        is_feature_extractor: false
        type: text
        data_column: Title
      - model: PATH_TO_TENSORFLOW_TEXT_FEATURE_EXTRACTOR
        is_feature_extractor: true
        type: text
        data_column: Description
      - model: PATH_TO_TENSORFLOW_IMAGE_CLASSIFICATION_MODEL
        is_feature_extractor: false
        type: image
        data_column: ImagePath
    epochs: 1
    batch_size: 128
    save_path: PATH_WHERE_THE_MODEL_SHOULD_BE_SAVED

3. Input data CSV file excerpt

+--------------------------------------+---------------------------------------------------------------+----------------------------------------+----------+
| Title                                | Description                                                   | ImageName                              | Category |
+======================================+===============================================================+========================================+==========+
| Bleu Gamepad Nintendo Wii U          | PILOT STYLE Touch Pen de marque Speedlink est 1 stylet 2      | image_938777978_product_201115110.jpg  | 50       |
+--------------------------------------+---------------------------------------------------------------+----------------------------------------+----------+

4. Manually using functions in the Napoleon library

.. code-block:: python

    import pandas as pd
    import tensorflow as tf
    from nltk.stem.snowball import FrenchStemmer

    from common.categorymapping import get_category_mapping_from_file
    from textclassification.common.datapreprocessing import preprocess_csv
    from textclassification.common.featureextraction import get_feature_extractor_from_model
    from multimodalclassification.model.model import get_multimodal_classification_model
    from multimodalclassification.dataprocessing.datasetgeneration import create_multimodal_dataset


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

        multi.fit(
            dataset_train,
            epochs=5,
            workers=5,
            use_multiprocessing=True
        )

        multi.evaluate(dataset_test)
