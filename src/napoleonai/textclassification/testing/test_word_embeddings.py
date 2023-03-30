from napoleonai.textclassification.common.wordembeddings import get_embedding_layer_and_vocab


def test_word2vec():
    embedding_layer, vocab = get_embedding_layer_and_vocab(
        pretrained_model_path=None,
        model_type='word2vec',
        train_corpus_path='/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/french_unsupervised_books.txt',
        layer_name='test_embedding_layer',
        vector_size=128,
        save_path='saved/models/word2vec/firstmodel'
    )
    print(embedding_layer.name)
    print(len(vocab))


def test_fasttext():
    embedding_layer, vocab = get_embedding_layer_and_vocab(
        pretrained_model_path=None,
        model_type='fasttext',
        train_corpus_path='/Users/cipri/PycharmProjects/coreLicenta/assets/trainingdata/french_unsupervised_books.txt',
        layer_name='test_embedding_layer',
        vector_size=128,
        save_path='saved/models/fasttext/firstmodel'
    )
    print(embedding_layer.name)
    print(len(vocab))


def test():
    # test_word2vec()
    test_fasttext()
