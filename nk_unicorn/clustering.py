
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import fastcluster
import hdbscan


class HDBSCAN:

    def __init__(self, min_cluster_size=5, hdbscan_kwargs=None):
        self.hdbscan_kwargs = dict(cluster_selection_method='leaf', memory='hdbscan-cache', prediction_data=True)
        if hdbscan_kwargs:
            self.hdbscan_kwargs.update(hdbscan_kwargs)

        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, **self.hdbscan_kwargs)

    def fit_predict(self, data):
        return self.hdbscan.fit_predict(data)


class FastAgglomerative:

    def __init__(self):
        pass

        # self.cluster = MFastHCluster(method="ward")

    def fit_predict(self, data):
        pass
        # self.cluster.linkage(data)
        # return self.cluster
        #     def agglomerative_culuster(self, target_data):
        #
        #         print("starting clustering...")
        #         clustering.linkage(target_data)
        #         cluster_count = 150
        #         curr_max_label = cluster_count - 1
        #         cut = 100.0
        #         while curr_max_label < cluster_count:
        #             cut = cut * 0.9
        #             labels = clustering.cut(cut)
        #             curr_max_label = max(labels)

        #         while curr_max_label > cluster_count:
        #             cut = cut * 1.05
        #             labels = clustering.cut(cut)
        #             curr_max_label = max(labels)
        #         # labels = clustering.cut(500)
        #         return labels


CLUSTER_CONFIGS = {
    'kmeans': dict(alg=KMeans, kwargs=dict(n_clusters=8)),
    'dbscan': dict(alg=DBSCAN, kwargs=dict(eps=35, min_samples=1)),
    'hdbscan': dict(alg=HDBSCAN, kwargs=dict(min_cluster_size=5)),
    'agglomerative': dict(alg=AgglomerativeClustering, kwargs=dict(n_clusters=8)),
}
