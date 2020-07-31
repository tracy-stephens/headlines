from sklearn.cluster import DBSCAN
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

    def dunns_index(self):
        pass


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


