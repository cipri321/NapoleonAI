import re
import string
from typing import List, Any, Set, Optional, Dict, Union, Tuple, Callable

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pandas import DataFrame

nltk.download('punkt')

ugly_chars: str = r'|'.join(map(lambda x: re.escape(x), string.punctuation + ' '))


def concatenate_text_columns(df: pd.DataFrame, text_columns: List[str]) -> pd.Series:
    """
    Concatenates columns specified in text_columns

    :param df: pandas Dataframe which has to include columns in text_columns
    :param text_columns: columns to be concatenated
    :return: pandas Series which contains the concatenated columns
    """
    for text_column in text_columns:
        df[text_column] = df[text_column].fillna('')
    return df[text_columns].agg('<STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP>'.join, axis=1)


def clean_text(text: Any, language: str, stem_function: Optional[Callable[[str], str]]) -> str:
    """
    Erases html tags, lowercases string, erases stopwords

    :param stem_function:
    :param text:
    :param language:
    :return:
    """
    text = str(text)
    text = BeautifulSoup(text).text
    stop_words: Set[str] = set(stopwords.words(language))
    text = " ".join(
        map(
            lambda x: x.strip().lower() if not stem_function else stem_function(x.strip().lower()),
            filter(
                lambda x: x not in stop_words and len(x) > 0,
                re.split(ugly_chars, text)
            )
        )
    )

    return text


def preprocess_csv(
        filename: str,
        concat_columns: bool,
        category_column: Optional[str],
        text_columns: List[str],
        category_map: Optional[Dict],
        stem_function: Optional[Callable[[str], str]]
) -> Union[tuple[DataFrame, Union[dict[Any, int], dict]], DataFrame]:
    """
    Creates a pandas dataframe which has two columns: Text and Category. Text has the input data for training, Category has the label
    e.g. csv_file =
    |col1 | col2| cat|
    |text1|text2|cat1|
    |text3|text4|cat2|
    concat_columns = True, text_columns=['col1', 'col2']
    result=
    |Text                                                                    |Category|
    |text1<STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP>text2|cat1    |
    |text3<STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP> <STOP>text4|cat2    |

    concat_columns = False, text_columns=['col1', 'col2']
    result=
    |Text |Category|
    |text1|cat1    |
    |text2|cat1    |
    |text3|cat2    |
    |text4|cat3    |

    :param stem_function:
    :param category_map:
    :param filename: path for the csv_file
    :param concat_columns: True if you want the text values in text_columns to be concatenated in Text column, False if you want to add a new row for each text value in the columns
    :param category_column: column which will be mapped to column Category in the result
    :param text_columns: text columns which will be used to create the Text column
    :return: pandas Dataframe
    """

    df: pd.DataFrame = pd.read_csv(filename)

    if concat_columns:
        df['Text'] = concatenate_text_columns(df, text_columns)
        if category_column:
            df = df[[category_column, 'Text']]
        else:
            df = df[['Text']]
        df['Text'] = df['Text'].apply(lambda x: clean_text(x, language='french', stem_function=stem_function))

    else:
        data = {
            # 'Category': [],
            'Text': []
        }
        if category_column:
            data['Category'] = []
        for idx, row in df.iterrows():
            for text_column in text_columns:
                if category_column:
                    data['Category'].append(row['Category'])
                data['Text'].append(clean_text(row[text_column], language='french', stem_function=stem_function))
        df = pd.DataFrame(data=data)

    df = df.dropna()
    if category_column:
        if not category_map:
            cat_map = {}
            no_cats = 0
            for cat in df['Category'].values:
                if cat not in cat_map:
                    cat_map[cat] = no_cats
                    no_cats += 1
        else:
            cat_map = category_map

    if category_column:
        df['Category'] = df['Category'].apply(lambda x: cat_map[str(x)])
        if not category_map:
            return df, cat_map
    return df
