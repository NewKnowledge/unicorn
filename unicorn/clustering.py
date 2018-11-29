from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from hdbscan import HDBSCAN

CLUSTER_CONFIGS = {
    "kmeans": dict(alg=KMeans, kwargs=dict(n_clusters=8)),
    "dbscan": dict(alg=DBSCAN, kwargs=dict(eps=2, min_samples=1)),
    "hdbscan": dict(
        alg=HDBSCAN,
        kwargs=dict(
            min_samples=1,
            min_cluster_size=15,
            cluster_selection_method="eom",
            prediction_data=False,
        ),
    ),
    "agglomerative": dict(alg=AgglomerativeClustering, kwargs=dict(n_clusters=8)),
}
