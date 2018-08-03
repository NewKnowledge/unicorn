''' Classes for performing image clustering '''

import logging

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dim_reduction import DIM_REDUC_CONFIGS
from clustering import CLUSTER_CONFIGS


class Unicorn:

    def __init__(self,
                 dim_reduc_config='spikesort-tsne',
                 dim_reduc_kwargs={},
                 cluster_config='hdbscan',
                 cluster_kwargs={},
                 ):

        # initialize preprocessing and clustering classes
        self.standard_scaler = StandardScaler(copy=False)

        # init dimensionality reduction algorithm, overwriting default ALG_CONFIGS values with given kwargs
        dred_conf = DIM_REDUC_CONFIGS[dim_reduc_config]
        dred_conf['kwargs'].update(dim_reduc_kwargs)
        self.dim_reduction_alg = dred_conf['alg'](**dred_conf['kwargs'])

        clus_conf = CLUSTER_CONFIGS[cluster_config]
        clus_conf['kwargs'].update(cluster_kwargs)
        self.cluster_alg = clus_conf['alg'](**clus_conf['kwargs'])

    def scale(self, data):
        return self.standard_scaler.fit_transform(data)

    def reduce_dimension(self, data):
        logging.info(f'reducing dim of data with shape {data.shape}')
        data = self.scale(data)
        return self.dim_reduction_alg.fit_transform(data)

    def cluster(self, data, reduce_dim=True):
        data = self.reduce_dimension(data) if reduce_dim else self.scale(data)
        return self.cluster_alg.fit_predict(data)

    def get_nearest_neighbors(self, data, n_neighbors=16, reduce_dim=True):
        data = self.reduce_dimension(data) if reduce_dim else self.scale(data)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data)
        distances, inds = nbrs.kneighbors(data)

        return [{'distance': d, 'index': (i, j)}
                for d, i, j
                in zip(distances, inds[:, 0], inds[:, 1])]

    @staticmethod
    def get_cluster_stats(data, cluster_labels):
        return [{
            'label': label,
            'std': data[cluster_labels == label].std(axis=0).mean(),
        } for label in set(cluster_labels)]
