Configuration
=============

Image Classification models
---------------------------

Image models are most often used as auxiliary aids in multimodal classification, as they are very costly to train and do not provide tremendous value to the accuracy of the classification

Image models can be very easily trained in Napoleon, using a yaml file.

Example:
^^^^^^^^

.. code-block:: yaml

    task: image classification
    type: EfficientNetV2B3
    model_name: efficient_net_rakuten_pretrained
    is_pretrained: true
    train_layers_keywords:
      - top
      - block6
    data:
      train_folder: PATH_TO_TRAIN_FOLDER
      img_size: 224
    epochs: 1

Description:
^^^^^^^^^^^^
* *task* -> image classification
* *type* -> this can be any architecture included in `Tensorflow applications <https://www.tensorflow.org/api_docs/python/tf/keras/applications>`_
* *model_name* -> the name of the Keras model and will be included in the naming of its layers
* *is_pretrained* -> true, if the model's weights should be initialized using some obtained after training on ImageNet dataset, false, otherwise
* *train_layers_keywords* -> only layers that contain these keywords will be trained, the other will be frozen
* *data* ->

    * train_folder ->
        * path to a folder containing a training dataset(it will be automatically separated in training+validation)
        * the folder should have the following structure

            | root
            |   label1
            |       img1
            |       img2
            |       ...
            |       imgn
            |   label2
            |       img1
            |       img2
            |       ...
            |       imgn
            |   ...

    * img_size -> the model will be trained with images of the size **(img_size, img_size, 3)**. If the images are not of that specific size, it will resize them automatically, but the new files will **not** be saved on disk
* *epochs* -> the number of epochs the model will be trained


Text Classification models
---------------------------

Example:
^^^^^^^^

* RNN

.. code-block:: yaml

    task: text classification
    type: rnn
    rnn_cell: lstm
    model_name: rnn_lstm_model
    train_corpus_path: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/french_corpus_ecommerce
    pretrained_model_path: /Users/cipri/PycharmProjects/coreLicenta/saved/models/rnn/title_description_word2vec_256_rnn
    data:
      test_file: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv
      category_mapping: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map
      no_labels: 27
      concat_columns: true
      text_columns:
        - Title
        - Description
      category_column: Category
      language: french
      category_map: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map
    rnn_layer_sizes:
      - 128
      - 128
    dense_layer_sizes:
      - 128
      - 32
    epochs: 10
    word_embeddings: word2vec
    word_embedding_size: 128
    optimizer: 'adam'
    batch_size: 128
    loss: 'categorical_crossentropy'
    save_path: /Users/cipri/PycharmProjects/coreLicenta/saved/models/rnn/title_description_word2vec_256_rnn

* Doc2Vec
    .. code-block::yaml
        ...

Description:
^^^^^^^^^^^^
* *task* -> text classification
* *type* -> Can be either "rnn" or "doc2vec", depending on the algorithm used for text feature extraction
* *rnn_cell* -> Can be either "lstm" or "gru", refers to the type of rnn cell used in the Neural Network. Should only be completed if "type" is "rnn"
* *model_name* -> the name of the Keras model and will be included in the naming of its layers
* *train_corpus_path* ->
    * the path to a file with varied language to learn word embeddings. This field is mandatory only if an algorithm to learn word embeddings is specified
    * the file should contain a paragraphs that belong to the same context on the same line(text that is on different lines will be considered unrelated)
* *pretrained_model_path* -> path to a Tensorflow Keras model for text classification that will be further trained on the provided data
* *data* ->
    * training_file -> path to a csv file, the data in this file will be used for training
    * test_file -> path to a csv file, the data in this file will be used only for evaluation, **not** for training
    * inference_file -> path to a csv file, the data in this file will be used for inference, it does not need a label column
    * category_mapping -> path to a text file, each line contains the name of the label and its corresponding index(from 0 to total number of labels)
    * no_labels -> total number of labels
    * text_columns -> list of text columns from the csv that will be used for classification
    * concat_columns -> true, the columns specified in text_columns will be concatenated, false, they will be processed separately
    * category_column -> name of the label column
    * language -> main language of the text file(used for stemming)
* *rnn_layer_sizes* -> the number of rnn cells in each rnn layer(as they are bidirectional, they size will appear double in the model's summary)
* *dense_layer_sizes* -> the number of neurons in each classifier dense layer
* *word_embeddings* -> word2vec or fasttext, depending on which you want to use to learn word embeddings
* *word_embedding_size* -> the size of the word embeddings
* *epochs* -> the number of epochs the model will be trained
* *batch_size* -> the size of the batches in training
* *loss* -> the loss function used in training
* *save_path* -> the path at which the Tensorflow model will be saved

Multi-modal Classification
--------------------------
Example
^^^^^^^

.. code-block:: yaml

    task: multimodal classification
    model_load_path: /Users/cipri/PycharmProjects/coreLicenta/saved/models/multimodal/titleRnn_descriptionRnn_imageEfficientNet
    data:
      training_file: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_train.csv
      test_file: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv
      inference_file: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_test.csv
      inference_results: /Users/cipri/PycharmProjects/coreLicenta/results/rakuten_inference_results.csv
      category_column: Category
      category_map: /Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/rakuten_category_map
      language: french
      images_path: /Users/cipri/Downloads/archive (5)/images/images/image_train
    modes:
      - model: /Users/cipri/PycharmProjects/coreLicenta/saved/models/rnn/title_word2vec_256_rnn
        is_feature_extractor: false
        type: text
        data_column: Title
      - model: /Users/cipri/PycharmProjects/coreLicenta/saved/models/doc2vec_classifier/title
        is_feature_extractor: false
        type: text
        data_column: Title
      - model: /Users/cipri/PycharmProjects/coreLicenta/models/image/efficient_net_rakuten
        is_feature_extractor: false
        type: image
        data_column: ImagePath
    epochs: 1
    batch_size: 128
    save_path: saved/models/multimodal/titleRnn_titledoc2vec_imageEfficientNet

Description
^^^^^^^^^^^

* *task* -> multimodal classification
* *model_load_path* -> path to a previously created multi-modal classification model that needs to be further trained, evaluated or used for inference
* *data* ->
    * training_file -> path to a csv file, the data contained in this file will be used for training only
    * test_file -> path to a csv file, the data contained in this file will be used for evaluation only
    * inference_file -> path to a csv file, the data contained in this file will be used for inference, it needs to have the same structure as training and test files, with the exception of the category column which can be missing
    * category_column -> column whose values should be used as labels
    * category_map -> path to a text file, each line contains the name of the label and its corresponding index(from 0 to total number of labels)
    * language -> main language of the text columns
    * images_path -> root of the directory in which the images are located
* *modes* -> contains a list of elements with the following structure
    * model: path to a previously generated text or image model
    * is_feature_extractor: true, the model doesn't contain dense classification layers at the end, false, the dense layers at the end of the neural network will be dropped
    * type: text or image
    * data_column: the name of the column in the CSV file that is considered the input of this model
* *epochs* -> number of epochs the model will be trained
* *batch_size* -> batch size used for training
* *save_path* -> the path at which the Tensorflow model will be saved


Training, Testing and Inference CSV file structure
--------------------------------------------------

.. csv-table:: CSV Structure
   :header: "Text1", "...", "TextN", "Image1", "...", "ImageN", "Category"
   :widths: 30, 10, 30, 30, 10, 30, 20

   "TEXT_CONTENT", "...", "TEXT_CONTENT", "IMAGE_NAME", "...", "IMAGE_NAME", "LABEL"

* **TEXT_CONTENT** can be any raw text content. If the provided text is html content, text will be extracted using `BeautifulSoup <https://beautiful-soup-4.readthedocs.io/en/latest/>`_
* **IMAGE_NAME** is the relative path from folder specified in "images_path" field(inside "data" field)