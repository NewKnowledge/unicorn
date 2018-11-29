""" Classes for performing image clustering """

from nk_logger import get_logger
from sklearn.preprocessing import StandardScaler

from .clustering import CLUSTER_CONFIGS
from .dim_reduction import DIM_REDUC_CONFIGS

logger = get_logger(__name__)


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

        self.dim_reduction_alg = self.init_dim_reduc_alg(
            dim_reduc_config, dim_reduc_kwargs
        )

        self.cluster_alg = self.init_cluster_alg(cluster_config, cluster_kwargs)

    def init_dim_reduc_alg(self, dim_reduc_config, dim_reduc_kwargs):
        # init dimensionality reduction algorithm, overwriting default config values with given kwargs
        if isinstance(dim_reduc_config, str):
            dred_conf = DIM_REDUC_CONFIGS[dim_reduc_config]
        elif isinstance(dim_reduc_config, dict):
            if "alg" not in dim_reduc_config:
                raise Exception("custom dimension reduction config must include 'alg'")
            dred_conf = dim_reduc_config
            if "kwargs" not in dred_conf:
                dred_conf["kwargs"] = {}
        else:
            raise Exception("dim_reduc_config must either be a string or a dict")
        dred_conf["kwargs"].update(dim_reduc_kwargs)
        return dred_conf["alg"](**dred_conf["kwargs"])

    def init_cluster_alg(self, cluster_config, cluster_kwargs):
        # init clustering algorithm, overwriting default config values with given kwargs
        if isinstance(cluster_config, str):
            clus_conf = CLUSTER_CONFIGS[cluster_config]
        elif isinstance(cluster_config, dict):
            if "alg" not in cluster_config:
                raise Exception("custom dimension reduction config must include 'alg'")
            clus_conf = cluster_config
            if "kwargs" not in clus_conf:
                clus_conf["kwargs"] = {}
        else:
            raise Exception("cluster_config must either be a string or a dict")
        clus_conf["kwargs"].update(cluster_kwargs)
        return clus_conf["alg"](**clus_conf["kwargs"])

    def scale(self, data):
        return self.standard_scaler.fit_transform(data)

    def reduce_dimension(self, data):
        logger.info(f"reducing dim of data with shape {data.shape}")
        data = self.scale(data)
        return self.dim_reduction_alg.fit_transform(data)

    def cluster(self, data, reduce_dim=True):
        data = self.reduce_dimension(data) if reduce_dim else self.scale(data)
        return self.cluster_alg.fit_predict(data)

    @staticmethod
    def get_cluster_stats(data, cluster_labels):
        return [
            {"label": label, "std": data[cluster_labels == label].std(axis=0).mean()}
            for label in set(cluster_labels)
        ]

    # def get_nearest_neighbors(self, data, n_neighbors=16, reduce_dim=True):
    #     data = self.reduce_dimension(data) if reduce_dim else self.scale(data)
    #     nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(
    #         data
    #     )
    #     distances, inds = nbrs.kneighbors(data)

    #     return [
    #         {"distance": d, "index": (i, j)}
    #         for d, i, j in zip(distances, inds[:, 0], inds[:, 1])
    #     ]
