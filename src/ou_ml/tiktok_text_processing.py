import string
from typing import Any

import emoji
from emot.emo_unicode import UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english'))


def transform_emoji_name(name: str) -> str:
    name = name.lower()
    name = name.replace(': ', '_')
    name = name.replace(' ', '_')
    name = name.replace('\'', '_')
    name = name.replace(',', '_')
    name = name.replace('-', '_')
    return ':' + name + ':'


def get_all_emoji_emoticon() -> dict:
    all_emoji_emoticons = {**EMOTICONS_EMO, **UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS,
                           **{k: v['en'].lower() for k, v in emoji.EMOJI_DATA.items()}}
    all_emoji_emoticons = {k: transform_emoji_name(v.lower().replace("'", "").replace(":", "").strip()) for k, v in
                           all_emoji_emoticons.items()}

    return all_emoji_emoticons


def get_emoji_tokens() -> Any:
    return get_all_emoji_emoticon().values()


def replace_emoji_w_token(comment: str) -> str:
    for k, v in get_all_emoji_emoticon().items():
        comment = comment.replace(k, v)
    return comment


def remove_Stopwords(text: str) -> str:
    words = word_tokenize(text.lower())
    sentence = [w for w in words if not w in STOPWORDS]
    return " ".join(sentence)


def lower(text: str) -> str:
    return text.lower()


def lemmatize_text(text: str) -> str:
    wordlist = []
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    return ' '.join(wordlist)


def clean_text(text: str) -> str:
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    text_arr = text1.split()

    return ' '.join([w for w in text_arr]).lower()


def preprocess_special_chars(comment: str) -> str:
    comment = comment.str.replace(r'(<br/>)', '', regex=True)
    comment = comment.str.replace(r'(<a).*(>).*(</a>)', '', regex=True)
    comment = comment.str.replace(r'(&amp)', '', regex=True)
    comment = comment.str.replace(r'(&gt)', '', regex=True)
    comment = comment.str.replace(r'(&lt)', '', regex=True)
    comment = comment.str.replace(r'(\xa0)', ' ', regex=True)
    return comment


def replace_emoji_with_text(text: str) -> str:
    text = replace_emoji_w_token(text)
    text = text.replace("::", " ")
    return text


def addStopwordsWithoutQuotes() -> None:
    newStopWords = set()
    for word in STOPWORDS:
        if "'" in word:
            newStopWords.add(word.replace("'", ""))

    # add custom stopwords
    STOPWORDS.add('u')
    STOPWORDS.add('b')
    STOPWORDS.add('ur')
    STOPWORDS.add('cause')
    STOPWORDS.add('gonna')
    STOPWORDS.add('gon')
    STOPWORDS.add('na')
    STOPWORDS.add('im')
    STOPWORDS.add('gon')
    STOPWORDS.add('na')
    STOPWORDS.add('cant')

    STOPWORDS.update(newStopWords)
