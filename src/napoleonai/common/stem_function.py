from typing import Callable

from nltk.stem.snowball import FrenchStemmer, EnglishStemmer


def get_stem_function(lang: str) -> Callable[[str], str]:
    """
    Returns a stemming function based on the nltk library if the language exists,
    if not, it will return the identical function

    :param lang: str - the language
    :return: function - stemming function
    """
    stemmer_functions = {
        'french': FrenchStemmer().stem,
        'english': EnglishStemmer().stem
    }
    if lang not in stemmer_functions:
        return lambda x: x
    return stemmer_functions[lang]