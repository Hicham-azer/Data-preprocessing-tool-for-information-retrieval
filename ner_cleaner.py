from text_preprocessing import *


def ner_cleaner(path, column, header='infer'):
    df = import_data(path, header)
    df = lowercase(df, column)
    df = remove_links(df, 'cleantext')
    df = remove_html(df, 'cleantext')
    df = remove_emoji(df, 'cleantext')
    df = remove_stopwords(df, 'cleantext')
    df = convert_and(df, 'cleantext')
    df = remove_single_characters(df, 'cleantext')
    df = handle_contractions(df, 'cleantext')
    df = remove_space(df, 'cleantext')
    df = sentence_tokenizer(df, 'cleantext')
    df = list_the_words(df, 'cleantext')
    return df


