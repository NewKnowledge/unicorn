""" Classes for performing image clustering """

import logging

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE

import spikesort_tsne

# from spikesorting_tsne import preprocessing_kilosort_results as preproc
# from spikesorting_tsne.tsne import t_sne as spikesort_tsne
# from spikesorting_tsne import io_with_cpp as io
# from spikesorting_tsne import spike_positioning_on_probe as pos


DIM_RED_CONFIGS = {
    "pca": dict(
        alg=PCA, kwargs=dict(n_components=8, copy=False, svd_solver="randomized")
    ),
    "rand-proj": dict(alg=GaussianRandomProjection, kwargs=dict(eps=0.1)),
    "tsne": dict(alg=TSNE, kwargs=dict(n_components=8, perplexity=30.0)),
}

CLUSTER_CONFIGS = {
    "kmeans": dict(alg=KMeans, kwargs=dict(n_clusters=8)),
    "dbscan": dict(alg=DBSCAN, kwargs=dict(eps=35, min_samples=1)),
    "agglomerative": dict(alg=AgglomerativeClustering, kwargs=dict(n_clusters=8)),
}


class Unicorn:
    def __init__(
        self,
        dim_reduc_config="pca",
        dim_reduc_kwargs={},
        cluster_config="dbscan",
        cluster_kwargs={},
    ):

        # initialize preprocessing and clustering classes
        self.standard_scaler = StandardScaler(copy=False)

        # init dimensionality reduction algorithm, overwriting default ALG_CONFIGS values with given kwargs
        dred_conf = DIM_RED_CONFIGS[dim_reduc_config]
        dred_conf["kwargs"].update(dim_reduc_kwargs)
        self.dim_reduction_alg = dred_conf["alg"](**dred_conf["kwargs"])

        clus_conf = CLUSTER_CONFIGS[cluster_config]
        clus_conf["kwargs"].update(cluster_kwargs)
        self.cluster_alg = clus_conf["alg"](**clus_conf["kwargs"])

    def scale(self, data):
        return self.standard_scaler.fit_transform(data)

    def reduce_dimension(self, data):
        logging.info(f"reducing dim of data with shape {data.shape}")
        data = self.scale(data)
        return self.dim_reduction_alg.fit_transform(data)

    def cluster(self, data, reduce_dim=True):
        data = self.reduce_dimension(data) if reduce_dim else self.scale(data)
        return self.cluster_alg.fit_predict(data)

    def get_nearest_neighbors(self, data, n_neighbors=16, reduce_dim=True):
        data = self.reduce_dimension(data) if reduce_dim else self.scale(data)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
            data
        )
        distances, inds = nbrs.kneighbors(data)

        return [
            {"distance": d, "index": (i, j)}
            for d, i, j in zip(distances, inds[:, 0], inds[:, 1])
        ]

    @staticmethod
    def get_cluster_stats(data, cluster_labels):
        return [
            {"label": label, "std": data[cluster_labels == label].std(axis=0).mean()}
            for label in set(cluster_labels)
        ]
