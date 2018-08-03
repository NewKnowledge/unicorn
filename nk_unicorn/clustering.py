from sklearn.cluster import DBSCAN, KMeans
from mlpy import MFastHCluster
import hdbscan


class HDBSCAN:

    def __init__(self, min_cluster_size=15, min_samples = 1, hdbscan_kwargs=None):
        self.hdbscan_kwargs = dict(cluster_selection_method='leaf', memory='hdbscan-cache', prediction_data=True)
        if hdbscan_kwargs:
            self.hdbscan_kwargs.update(hdbscan_kwargs)

        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, **self.hdbscan_kwargs)

    def fit_predict(self, data):
        return self.hdbscan.fit_predict(data)


class FastAgglomerative:

    def __init__(self, n_clusters = 150, cut = 100.0, agglomerative_kwargs = None):
        self.agglomerative_kwargs = dict(method = 'ward')
        if agglomerative_kwargs:
            self.agglomerative_kwargs.update(agglomerative_kwargs)
        self.cluster = MFastHCluster(**self.agglomerative_kwargs)
        self.n_clusters = n_clusters
        self.cut = cut

    def fit_predict(self, data):
        # print("starting clustering...")
        self.cluster.linkage(data)
        cluster_count = self.n_clusters
        curr_max_label = cluster_count - 1
        cut = self.cut
        while curr_max_label < cluster_count:
            cut = cut * 0.9
            labels = clustering.cut(cut)
            curr_max_label = max(labels)

        while curr_max_label > cluster_count:
            cut = cut * 1.05
            labels = clustering.cut(cut)
            curr_max_label = max(labels)

        return labels


CLUSTER_CONFIGS = {
    'kmeans': dict(alg=KMeans, kwargs=dict(n_clusters=8)),
    'dbscan': dict(alg=DBSCAN, kwargs=dict(eps=35, min_samples=1)),
    'hdbscan': dict(alg=HDBSCAN, kwargs=dict(min_cluster_size=15, min_samples = 1)),
    'agglomerative': dict(alg=FastAgglomerative, kwargs=dict(n_clusters=150, cut = 100.0)),
}
