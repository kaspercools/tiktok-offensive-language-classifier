import pandas as pd

def build_internet_slang_dictionary() -> dict:
    df = pd.read_csv('../data/internet_slangs.csv')
    df.head()
    df = df.drop(columns=['id'])
    slang_dictionary = df.to_dict('records')

    return slang_dictionary

def get_emoji_tokens() -> list:
    # emoji source: https://www.kaggle.com/datasets/eliasdabbas/emoji-data-descriptions-codepoints?select=emoji_df.csv
    df = pd.read_csv('../data/emoji_df.csv')
    df.head()
    df['token'] = df['name'].apply(transform_emoji_name)
    return df.token.to_list()

def get_genz_slang() -> list:
    df = pd.read_csv('../data/genz_slang.csv')
    df.head()
    return df.keyword.to_list()

def get_emoji_token_dic()-> dict:
    df = pd.read_csv('../data/emoji_df.csv')
    df.head()
    df['token'] = df['name'].apply(transform_emoji_name)
    return pd.Series(df.emoji.values,index=df.token).to_dict()



def replace_emoji_w_token(comment, token_dict)-> str :
    for token, emoji in token_dict.items():
        comment = comment.replace(emoji, token)
    return comment


def transform_emoji_name(name) -> str:
    name = name.lower()
    name = name.replace(': ', '_')
    name = name.replace(' ', '_')
    name = name.replace('\'', '_')
    name = name.replace(',', '_')
    return ':'+name+':'