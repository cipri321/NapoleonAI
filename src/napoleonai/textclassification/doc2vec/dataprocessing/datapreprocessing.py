import re
import string

DEFAULT_STRIP_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'


def process_text(text: str):
    if not isinstance(text, str):
        text=''
    text = text.lower()
    text = re.sub(DEFAULT_STRIP_REGEX, '', text)
    return text
