from text_preprocessing import *


# Text cleaner for sentiment analysis

def sentiment_cleaner(path, column, header='infer'):
    df = import_data(path, header)
    df = lowercase(df, column)
    df = remove_links(df, 'cleantext')
    df = remove_html(df, 'cleantext')
    df = convert_emoji(df, 'cleantext')
    df = remove_punctuation(df, 'cleantext')
    df = remove_username(df, 'cleantext')
    df = remove_tags(df, 'cleantext')
    df = convert_and(df, 'cleantext')
    df = remove_single_characters(df, 'cleantext')
    df = handle_contractions(df, 'cleantext')
    df = remove_numbers(df, 'cleantext')
    df = lemmatize_words(df, 'cleantext')
    df = remove_space(df, 'cleantext')
    return df
