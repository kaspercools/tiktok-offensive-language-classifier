import pandas as pd
import emoji
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 


data_folder = 'data'

def transform_emoji_name(name: str) -> str:
    name = name.lower()
    name = name.replace(': ', '_')
    name = name.replace(' ', '_')
    name = name.replace('\'', '_')
    name = name.replace(',', '_')
    name = name.replace('-', '_')
    return ':'+name+':'

## formatting
all_emoji_emoticons = {**EMOTICONS_EMO,**UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS,**{k: v['en'].lower() for k, v in emoji.EMOJI_DATA.items()}}
all_emoji_emoticons = {k:transform_emoji_name(v.lower().replace("'","").replace(":","").strip()) for k,v in all_emoji_emoticons.items()}


def build_internet_slang_dictionary() -> dict:
    df = pd.read_csv(data_folder+'/internet_slangs.csv')
    df.head()
    df = df.drop(columns=['id'])
    slang_dictionary = df.to_dict('records')

    return slang_dictionary

def get_emoji_tokens() -> list:
    return all_emoji_emoticons.values()

def get_genz_slang() -> list:
    df = pd.read_csv(data_folder+'/genz_slang.csv')
    df['keyword']= df['keyword'].apply(lambda c: c.lower())
    df.head()
    return df.keyword.to_list()


def replace_emoji_w_token(comment:str)-> str :
    for k,v in all_emoji_emoticons.items():
        comment = comment.replace(k,v)
    return comment

def remove_Stopwords(text:str)->str:
    words = word_tokenize( text.lower() ) 
    sentence = [w for w in words if not w in STOPWORDS]
    return " ".join(sentence)
    
def lower(text:str)->str:
    return text.lower()

def lemmatize_text(text:str)->str:
    wordlist=[]
    lemmatizer = WordNetLemmatizer() 
    sentences=sent_tokenize(text)
    for sentence in sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    return ' '.join(wordlist) 

def clean_text(text:str)-> str: 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr]) 
    
    return text2.lower()

def preprocess_special_chars(comment:str)->str:
    comment = comment.str.replace(r'(<br/>)', '', regex=False)
    comment = comment.str.replace(r'(<a).*(>).*(</a>)', '', regex=False)
    comment = comment.str.replace(r'(&amp)', '', regex=False)
    comment = comment.str.replace(r'(&gt)', '', regex=False)
    comment = comment.str.replace(r'(&lt)', '', regex=False)
    comment = comment.str.replace(r'(\xa0)', ' ', regex=False) 
    return comment

def replace_emoji_with_text(text:str)-> str:
    text = replace_emoji_w_token(text)
    text = text.replace("::", " ")
    return text

STOPWORDS = set(stopwords.words('english'))
def addStopwordsWithoutQuotes():
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
