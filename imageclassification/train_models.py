from typing import Tuple

import tensorflow as tf
from imageclassification.create_models import ModelWithInfo, create_models
from imageclassification.dataset_generation import load_dataset, generate_dataset_stats
from settings import DEFAULT_BATCH_SIZE, DEFAULT_NO_EPOCHS, DEFAULT_LOGS_FOLDER, DEFAULT_MODELS_FOLDER, \
    DEFAULT_IMAGE_SIZE
from .image_types import ModelConfiguration


def train_model(
        model: ModelWithInfo,
        train_dataset: tf.data.Dataset,
        validate_dataset: tf.data.Dataset,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_NO_EPOCHS,
        logs_folder: str = DEFAULT_LOGS_FOLDER,
        model_folders: str = DEFAULT_MODELS_FOLDER
) -> ModelWithInfo:
    """
    Trains a model using a given dataset and validation set

    :param model: ModelWithInfo
    :param train_dataset: training image dataset
    :param validate_dataset: validation dataset
    :param batch_size: int
    :param epochs: int
    :param logs_folder: str, path where it should output the logs
    :param model_folders: str, path where it should save the logs
    :return:
    """
    print(f'==========================================\n'
          f'        TRAINING {model.name}             \n'
          f'==========================================\n'
          )
    model.model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    model.model.summary()
    model.model.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validate_dataset,
        callbacks=
        [
            tf.keras.callbacks.CSVLogger(f'{logs_folder}/{model.name}.csv'),
        ]
    )
    model.model.save(f'{model_folders}/{model.name}')
    return model


def get_models(
        img_size,
        data_folder,
        models: Tuple[ModelConfiguration, ...],
        epochs: int = DEFAULT_NO_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Creates and trains multiple image classification models that are configured using
    the ModelConfiguration protocol
    training folder has to have the folder structure
    label1
    |   img1
    |   img2
    |   ...
    label2
    |   img1
    |   img2
    |   ...
    ...
    labeln
    |   img1
    |   img2
    |   ...

    :param img_size: int, all images are (img_size, img_size, 3)
    :param data_folder: str, path to where the pictures are
    :param models: list of ModelConfiguration
    :param epochs: int, number of epochs used for training
    :param batch_size: int, batch size used for training
    :return: None
    """
    train, validation = load_dataset(img_size, DEFAULT_IMAGE_SIZE, data_folder)
    no_categories_train = generate_dataset_stats(train, 'training')
    no_categories_validate = generate_dataset_stats(validation, 'validation')

    models = create_models(img_size, max(no_categories_train, no_categories_validate), models=models)

    for model in models:
        train_model(
            model,
            train_dataset=train,
            validate_dataset=validation,
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=DEFAULT_NO_EPOCHS,
            logs_folder=f'{DEFAULT_LOGS_FOLDER}/image',
            model_folders=f'{DEFAULT_MODELS_FOLDER}/image'
        )