import numpy as np
from sklearn.decomposition import TruncatedSVD
import nltk

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


def get_word_weights(a=1e-3):

    file_name = r'data/glove/enwiki_vocab_min200.txt'
    word_weights = {}
    with open(file_name) as f:
        lines = f.readlines()

    N = 0
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            i = i.split()
            if len(i) == 2:
                word_weights[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)

    new_weights = {}
    for key, value in word_weights.items():
        new_weights[clean(key)[0]] = a / (a + value / N)

    return new_weights


def get_weighted_average(embeddings, words, weights):
    """ Compute the weighted average vectors """
    wgts = []
    for w in words:
        try:
            wgts.append(weights[w])
        except KeyError:
            wgts.append(0)
    wgts = np.array(wgts)
    return np.dot(embeddings.T, wgts) / np.count_nonzero(wgts)


def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


if __name__ == '__main__':

    from importlib import reload
    import glove_helper
    from articles import article

    reload(glove_helper)
    ndim = 50
    hands = glove_helper.Hands(ndim=ndim)

    a = article('hi my name is', '', '', '')
    embeddings = a.embeddings(hands)
    weights = get_word_weights()

    print(get_weighted_average(embeddings, a.clean_headline(), weights))

    print(a.embeddings(hands))




    # print(get_word_weights())
    # params = params()
    # #params.rmpc = 1
    #
    # print(params().LW)