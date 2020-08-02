import nltk
#nltk.download('punkt')
import numpy as np
from transformers import BertTokenizer
import tensorflow as tf

# to compute weighted average
from sif import get_weighted_average

# nltk stemmer for words
PORTER = nltk.stem.porter.PorterStemmer()

# BERT tokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')


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
                pub_date=df_row['pub_date'],
                keywords=df_row['keywords'])
    return a


def join_if_none(strings, sep=' '):
    """ Join lists of strings with possible NoneType """
    clean_strings = [s for s in strings if s]
    return sep.join(clean_strings)


class article:

    def __init__(self, headline, abstract, byline, pub_date, keywords):
        self.headline = headline
        self.abstract = abstract
        self.byline = byline
        self.pub_date = pub_date
        self.keywords = keywords

    def clean_headline(self):
        return clean(self.headline)

    def clean_abstract(self):
        return clean(self.abstract)

    def embeddings(self, hands):
        words = self.clean_headline()
        return np.array([hands.get_vector(w, strict=False) for w in words])

    def headline_embeddings(self, hands, method, weights=None):
        embeddings = self.embeddings(hands)
        # words = self.clean_headline()
        # embeddings = np.array([hands.get_vector(w, strict=False) for w in words])
        if method == 'avg':
            return np.mean(embeddings, axis=0)
        if method == 'weighted_avg':
            return get_weighted_average(embeddings, self.clean_headline(), weights)
        else:
            pass

    def abstract_embeddings(self, hands, method, weights=None):
        words = self.clean_abstract()
        embeddings = np.array([hands.get_vector(w, strict=False) for w in words])
        if method == 'avg':
            return np.mean(embeddings, axis=0)
        if method == 'weighted_avg':
            return get_weighted_average(embeddings, self.clean_abstract(), weights)
        else:
            pass

    def first_author(self):
        byline = eval(self.byline)
        if byline:
            author = join_if_none([byline[0]['firstname'],
                                   byline[0]['middlename'],
                                   byline[0]['lastname']])
            if author != '':
                return author
            else:
                return np.nan
        else:
            return np.nan

    def subjects(self):
        kwds = eval(self.keywords)
        subjects = []
        if kwds:
            for kw in kwds:
                if kw['name'] == 'subject':
                    subjects.append(kw['value'])
        return subjects

    def bert_headline_embedding(self, model):
        ids = tf.constant(TOKENIZER.encode(self.headline))[None, :]
        return model(ids)


if __name__ == '__main__':

    from importlib import reload
    import glove_helper

    reload(glove_helper)
    ndim = 50
    hands = glove_helper.Hands(ndim=ndim)

    a = article('hi my name is', '', '', '')
    #print(a.headline_embeddings(hands, method='avg'))

    from sif import get_word_weights
    weights = get_word_weights()
    x = a.headline_embeddings(hands, method='weighted_avg', weights=weights)
    print(x.shape)


