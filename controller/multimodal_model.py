from typing import Union, List, Tuple, Optional

import pandas as pd
import tensorflow as tf

from common.categorymapping import get_category_mapping_from_file
from common.stem_function import get_stem_function
from common.validation_functions import validate_configuration
from imageclassification.feature_extraction import get_feature_extractor_from_saved_model
from multimodalclassification.dataprocessing.datasetgeneration import create_multimodal_dataset
from multimodalclassification.model.model import get_multimodal_classification_model
from textclassification.common.datapreprocessing import preprocess_csv
from textclassification.common.featureextraction import get_feature_extractor_from_model


def validate_model_conf(model_conf):
    mandatory_data = {
        'model': str,
        'is_feature_extractor': bool,
        'type': str,
        'data_column': Union[str, List]
    }
    validate_configuration(model_conf, mandatory_data)


def create_df_for_col(data_type, data_column, data_conf, cat_map) \
        -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Creates dataframes for train, test and inference(depending on the configuration)
    each composed of the data_column and the corresponding label

    :param data_type: str, either text or image
    :param data_column: str, name of label category
    :param data_conf: dict, data configuration dictionary
    :param cat_map: dict, category mapping(it needs to be from 0 to n)
    :return: tuple of three (potential) dataframes: training, test, inference
    """
    df_train, df_test, df_inference = None, None, None
    if data_type == 'text':
        if 'training_file' in data_conf:
            df_train = preprocess_csv(
                data_conf['training_file'],
                concat_columns=data_conf['concat_columns']
                if 'concat_columns' in data_conf else True,
                text_columns=data_column
                if isinstance(data_column, list) else [data_column],
                category_map=cat_map,
                stem_function=get_stem_function(data_conf['language']),
                category_column=data_conf['category_column']
            )
        if 'test_file' in data_conf:
            df_test = preprocess_csv(
                data_conf['test_file'],
                concat_columns=data_conf['concat_columns']
                if 'concat_columns' in data_conf else True,
                text_columns=data_column
                if isinstance(data_column, list) else [data_column],
                category_map=cat_map,
                stem_function=get_stem_function(data_conf['language']),
                category_column=data_conf['category_column']
            )
        if 'inference_file' in data_conf:
            df_inference = preprocess_csv(
                data_conf['inference_file'],
                concat_columns=data_conf['concat_columns']
                if 'concat_columns' in data_conf else True,
                text_columns=data_column
                if isinstance(data_column, list) else [data_column],
                category_map=None,
                stem_function=get_stem_function(data_conf['language']),
                category_column=None
            )
    elif data_type == 'image':
        if 'training_file' in data_conf:
            df = pd.read_csv(data_conf['training_file'])
            df_train = df[[data_conf['category_column']]]
            df_train['Image'] = df[data_column]
        if 'test_file' in data_conf:
            df = pd.read_csv(data_conf['test_file'])
            df_test = df[[data_conf['category_column']]]
            df_test['Image'] = df[data_column]
        if 'inference_file' in data_conf:
            df = pd.read_csv(data_conf['inference_file'])
            df_inference = pd.DataFrame()
            df_inference['Image'] = df[data_column]
    return df_train, df_test, df_inference


def valid_df(df):
    try:
        df.any()
        return True
    except Exception:
        return False


def get_ordered_categories(results, cat_map):
    results = [(idx, res) for idx, res in enumerate(results)]
    results.sort(key=lambda x: x[1], reverse=True)
    reverse_cat_map = {v: k for k, v in cat_map.items()}
    results = [reverse_cat_map[idx] for idx, res in results]
    return results


def create_multimodal_classification_model(configuration) -> tf.keras.Model:
    """
    Creates a multimodal classification model using a configuration dictionary

    :param configuration: dictionary
    :return: Keras model
    """
    cat_map, no_labels = get_category_mapping_from_file(
        configuration['data']['category_map'])

    text_models = []
    image_models = []

    text_cols = []
    image_cols = []

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_inference = pd.DataFrame()

    for idx, model_conf in enumerate(configuration['modes']):
        validate_model_conf(model_conf)
        model = tf.keras.models.load_model(model_conf['model'])
        if model_conf['is_feature_extractor']:
            feature_extractor = model
        else:
            if model_conf['type'] == 'text':
                feature_extractor = get_feature_extractor_from_model(model)
            else:
                feature_extractor = get_feature_extractor_from_saved_model(model_conf['model'])
        if model_conf['type'] == 'text':
            text_models.append(feature_extractor)
        else:
            image_models.append(feature_extractor)

        col_train, col_test, col_inference = create_df_for_col(model_conf['type'], model_conf['data_column'],
                                                               configuration['data'],
                                                               cat_map)

        if model_conf['type'] == 'text':
            col = f'text_col_{idx}'
            text_cols.append(col)
            if valid_df(col_train):
                df_train[col] = col_train['Text']
            if valid_df(col_test):
                df_test[col] = col_test['Text']
            if valid_df(col_inference):
                df_inference[col] = col_inference['Text']
        elif model_conf['type'] == 'image':
            col = f'image_col_{idx}'
            image_cols.append(col)
            if valid_df(col_train):
                df_train[col] = col_train['Image']
            if valid_df(col_test):
                df_test[col] = col_test['Image']
            if valid_df(col_inference):
                df_inference[col] = col_inference['Image']

    # df_train=df_train.sample(1000)
    # df_test=df_test.sample(100)

    if 'training_file' in configuration['data']:
        df_train['Category'] = pd.read_csv(
            configuration['data']['training_file']
        )[configuration['data']['category_column']]
        multi_train_ds = create_multimodal_dataset(
            df_train,
            text_columns=text_cols,
            image_columns=image_cols,
            category_column='Category',
            one_hot=True,
            category_map=cat_map,
            images_path=configuration['data']['images_path']
        )

    if 'test_file' in configuration['data']:
        df_test['Category'] = pd.read_csv(
            configuration['data']['test_file']
        )[configuration['data']['category_column']]
        multi_test_ds = create_multimodal_dataset(
            df_test,
            text_columns=text_cols,
            image_columns=image_cols,
            category_column='Category',
            one_hot=True,
            category_map=cat_map,
            images_path=configuration['data']['images_path']
        )

    if 'inference_file' in configuration['data']:
        multi_inference_ds = create_multimodal_dataset(
            df_inference,
            text_columns=text_cols,
            image_columns=image_cols,
            category_column=None,
            category_map=None,
            images_path=configuration['data']['images_path']
        )

    if 'model_load_path' in configuration:
        multi_model = tf.keras.models.load_model(configuration['model_load_path'])
    else:
        multi_model = get_multimodal_classification_model(
            text_models,
            image_models,
            no_labels
        )

    multi_model.summary()

    multi_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        loss='categorical_crossentropy',
        run_eagerly=True
    )

    if 'training_file' in configuration['data']:
        multi_model.fit(
            multi_train_ds,
            epochs=configuration['epochs'],
            workers=5,
            use_multiprocessing=True
        )

    if 'test_file' in configuration['data']:
        multi_model.evaluate(
            multi_test_ds
        )

    if 'inference_file' in configuration['data']:
        results = []
        for i in multi_inference_ds:
            res = multi_model(i)
            results += list(res)
        # results = multi_model.predict(multi_inference_ds)
        res_df = pd.DataFrame(
            data={'Cat. Probs.': [' '.join(get_ordered_categories(res, cat_map)) for res in results]}
        )

        res_df.to_csv(configuration['data']['inference_results'])
    if 'save_path' in configuration:
        multi_model.save(configuration['save_path'])

    return multi_model