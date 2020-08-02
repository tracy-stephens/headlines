from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics.pairwise import cosine_similarity
from s_dbw import S_Dbw, SD
import numpy as np


def cluster_center(cluster):
    center = np.mean(cluster, axis=0)
    sim = [cosine_similarity([p, center])[0,1] for p in cluster]
    return sim.index(min(sim))


class model:

    def __init__(self, data, eps, min_samples):
        self.data = np.array(data)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = None

    def train(self):
        db = DBSCAN(eps=self.eps,
                    min_samples=self.min_samples,
                    metric='cosine')
        self.dbscan = db.fit(self.data)

    def labels(self):
        return self.dbscan.labels_

    def num_clusters(self):
        return len(set(self.labels())) - 1

    def clusters(self):
        return [self.data[self.labels() == n] for n in range(self.num_clusters())]

    def cluster_sizes(self):
        return [i.shape[0] for i in self.clusters()[:]]

    def cluster_centers(self):
        pass

    def silhouette_score(self):
        if self.num_clusters() > 0:
            return silhouette_score(self.data, self.labels(), metric='cosine')
        else:
            return np.nan

    def calinski_harabasz_score(self):
        if self.num_clusters() > 0:
            return calinski_harabasz_score(self.data, self.labels())
        else:
            return np.nan

    def s_dbw(self):
        if self.num_clusters() > 0:
            return S_Dbw(self.data, self.labels())
        else:
            return np.nan

    def sd(self):
        if self.num_clusters() > 0:
            return SD(self.data, self.labels())
        else:
            return np.nan

    def homogeneity_score(self, true_labels):
        labels = self.labels()
        pred_labels = [l for l in labels if l > -1]
        true_labels_2 = [true_labels[i] for i in range(len(labels)) if labels[i] > -1]
        return homogeneity_score(true_labels_2, pred_labels)

    def completeness_score(self, true_labels):
        labels = self.labels()
        pred_labels = [l for l in labels if l > -1]
        true_labels_2 = [true_labels[i] for i in range(len(labels)) if labels[i] > -1]
        return completeness_score(true_labels_2, pred_labels)

    def v_measure_score(self, true_labels):
        labels = self.labels()
        pred_labels = [l for l in labels if l > -1]
        true_labels_2 = [true_labels[i] for i in range(len(labels)) if labels[i] > -1]
        return v_measure_score(true_labels_2, pred_labels)


if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    from importlib import reload
    import glove_helper
    from articles import article, to_article

    reload(glove_helper)
    ndim = 50
    hands = glove_helper.Hands(ndim=ndim)

    data = pd.read_csv(r'data/europe_test.csv')
    articles = [to_article(data.loc[i]) for i in data.index]
    embeddings = [i.headline_embeddings(hands, method='avg') for i in articles]

    md = model(embeddings, eps=0.1, min_samples=2)

    print(md.labels_)


