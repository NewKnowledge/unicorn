from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

from hdbscan import HDBSCAN

CLUSTER_ALGS = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "hdbscan": HDBSCAN,
    "agglomerative": AgglomerativeClustering,
}

CLUSTER_CONFIGS = {
    "kmeans": dict(n_clusters=8),
    "dbscan": dict(eps=2, min_samples=1),
    "hdbscan": dict(
        min_samples=1,
        min_cluster_size=15,
        cluster_selection_method="eom",
        prediction_data=False,
    ),
    "agglomerative": dict(n_clusters=8),
}
