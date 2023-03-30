from typing import List

import numpy as np
from gensim.models.doc2vec import TaggedDocument

from napoleonai.textclassification.doc2vec.dataprocessing.datapreprocessing import process_text


def get_tagged_documents_from_text_dataset(
        text_dataset: np.ndarray,
        already_tokenised: bool
) -> List:
    if not already_tokenised:
        tagged_docs = [TaggedDocument(process_text(text).split(), [idx]) for idx, text in enumerate(text_dataset)]
    else:
        tagged_docs = [TaggedDocument(doc, [idx]) for idx, doc in enumerate(text_dataset)]
    return tagged_docs

