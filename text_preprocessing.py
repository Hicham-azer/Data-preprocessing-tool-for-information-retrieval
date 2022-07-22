import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import contractions
from textblob import TextBlob
import emoji

nltk.download('stopwords')


# Importing the data

def import_data(filename, header):
    df = pd.read_csv(filename, header=header)
    return df


# Punctuation Removal

def remove_punctuation(df, column):
    df['cleantext'] = df[column].apply(
        lambda s: ' '.join(re.sub(r"[!”\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~\"]", " ", s).split()))
    return df


# lowercase

def lowercase(df, column):
    df['cleantext'] = df[column].str.lower()
    return df


# contractions

def handle_contractions(df, column):
    df['cleantext'] = df[column].apply(lambda x: contractions.fix(x))
    return df


# Stopwords


def remove_stopwords(df, column):
    def remove(string):
        words = string.split()
        stop = set(stopwords.words('english'))
        clean_words = [word for word in words if word not in stop]
        clean_string = " ".join(clean_words)
        return clean_string

    df['cleantext'] = df[column].apply(lambda s: remove(s))
    return df


# Numbers removal

def remove_numbers(df, column):
    df['cleantext'] = df[column].apply(lambda s: ''.join(re.sub('\d+', '', s)))
    return df


# Links Removal

def remove_links(df, column):
    df["cleantext"] = df[column].apply(lambda s: ' '.join(re.sub('http://\S+|https://\S+|www\.\S+', " ", s).split()))
    return df


# Html tags removal

def remove_html(df, column):
    df["cleantext"] = df[column].apply(lambda s: ''.join(re.sub('<[^>]*>', '', s)))
    return df


# username removal

def remove_username(df, column):
    df['cleantext'] = df[column].apply(lambda x: ''.join(re.sub(r'@\w*', '', x)))
    return df


# tags removal

def remove_tags(df, column):
    df["cleantext"] = df[column].apply(lambda x: ''.join(re.sub('#\w+', '', x)))
    return df


# White space removal

def remove_space(df, column):
    df['cleantext'] = df[column].apply(lambda x: x.strip())
    return df


# correct words ( complexity !! )

def correct_words(df, column):
    df['cleantext'] = df[column].apply(lambda x: TextBlob(x).correct())
    return df


# convert emoji

def convert_emoji(df, column):
    df['cleantext'] = df[column].apply(lambda x: emoji.demojize(x, delimiters=('', ' ')))
    return df


# Remove emoji

def remove_emoji(df, column):
    df['cleantext'] = df[column].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
    return df


# Convert and

def convert_and(df, column):
    df['cleantext'] = df[column].apply(lambda x: ''.join(re.sub('&', 'and', x)))
    return df


# Lemmatizer

def lemmatize_words(df, column):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

    df['cleantext'] = df[column].apply(lemmatize_text)
    df['cleantext'] = df['cleantext'].apply(lambda x: ' '.join(x))
    return df


# sentence tokenizer

def sentence_tokenizer(df, column):
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    df['cleantext'] = df[column].apply(lambda x: re.split(pattern, x))
    return df


# remove_single_characters

def remove_single_characters(df, column):
    df['cleantext'] = df[column].apply(lambda x: ''.join(re.sub('( |^)\w( |$)', ' ', x)))
    df['cleantext'] = df['cleantext'].apply(lambda x: ''.join(re.sub('( |^)\w( |$)', ' ', x)))
    return df


# Convert the row to a list of words regrouped in a list that represents the sentence. It should be executed after
# sentence_tokenizer

def list_the_words(df, column):
    i = 0
    for row in df[column]:
        new_row = []
        for sentence in row:
            sentence = nltk.word_tokenize(sentence)
            new_row.append(list(sentence))
        df.at[i, 'cleantext'] = new_row
        i += 1
    return df
