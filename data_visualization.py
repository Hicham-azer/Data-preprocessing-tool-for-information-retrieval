import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Generating the word cloud

def word_cloud(df, column):
    text = " ".join(review for review in df[column])
    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()