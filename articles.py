import nltk
#nltk.download('punkt')
import numpy as np

# nltk stemmer for words
PORTER = nltk.stem.porter.PorterStemmer()


def clean(text):
    """ preproccess a list of words for nlp """

    if not isinstance(text, str):
        return []
    else:
        words = nltk.word_tokenize(text)  # split into words
        words = [word.lower() for word in words]  # lowercase
        words = [word for word in words if word.isalpha()]  # remove punctuation
        words = [PORTER.stem(word) for word in words]  # stemming
        return words  # list of words in sentence


def to_article(df_row):
    """ converts a row in a df to an article object """
    a = article(headline=df_row['main'],
                abstract=df_row['abstract'],
                byline=df_row['person'],
                pub_date=df_row['pub_date'])
    return a


class article:

    def __init__(self, headline, abstract, byline, pub_date):
        self.headline = headline
        self.abstract = abstract
        self.byline = byline
        self.pub_date = pub_date

    def clean_headline(self):
        return clean(self.headline)

    def clean_abstract(self):
        return clean(self.abstract)

    def headline_embeddings(self, hands, method):
        words = self.clean_headline()
        embeddings = np.array([hands.get_vector(w, strict=False) for w in words])
        if method == 'avg':
            return np.mean(embeddings, axis=0)
        else:
            pass

    def abstract_embeddings(self, hands, method):
        words = self.clean_abstract()
        embeddings = np.array([hands.get_vector(w, strict=False) for w in words])
        if method == 'avg':
            return np.mean(embeddings, axis=0)
        else:
            pass


if __name__ == '__main__':

    from importlib import reload
    import glove_helper

    reload(glove_helper)
    ndim = 50
    hands = glove_helper.Hands(ndim=ndim)

    a = article('hi my name is', '', '', '')
    print(a.headline_embeddings(hands, method='avg'))
