''' Classes for performing image clustering '''

import logging

import numpy as np
from keras.applications import inception_v3
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE

from .image_utils import image_array_from_path, image_array_from_url


ALG_CONFIGS = {
    # dimensionality reduction
    'pca': dict(alg=PCA, kwargs=dict(n_components=8, copy=False, svd_solver='randomized')),
    'rand-proj': dict(alg=GaussianRandomProjection, kwargs=dict(eps=0.1)),
    'tsne': dict(alg=TSNE, kwargs=dict(n_components=8, perplexity=30.0)),
    # clustering
    'kmeans': dict(alg=KMeans, kwargs=dict(n_clusters=8)),
    'dbscan': dict(alg=DBSCAN, kwargs=dict(eps=35, min_samples=1)),
}
# TODO sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity=’euclidean’, memory=None, connectivity=None, compute_full_tree=’auto’, linkage=’ward’, pooling_func=<function mean>)


class Unicorn:

    def __init__(self,
                 dim_reduc_config='pca',
                 cluster_config='dbscan',
                 ):

        # initialize preprocessing and clustering classes
        self.standard_scaler = StandardScaler(copy=False)
        dred_conf = ALG_CONFIGS[dim_reduc_config]
        clus_conf = ALG_CONFIGS[cluster_config]
        self.dim_reduction_alg = dred_conf['alg'](**dred_conf['kwargs'])
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


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, target_size=(299, 299)):
        # TODO allow for other keras imagenet models
        self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        self.target_size = target_size
        self.preprocess = inception_v3.preprocess_input
        self.decode = inception_v3.decode_predictions

    # TODO cache url/path -> feature funcs?
    def get_features_from_paths(self, image_paths, n_channels=None):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
        return self.get_features(images_array, n_channels=n_channels)

    def get_features_from_urls(self, image_urls, n_channels=None):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to
        successfully downloaded images along with the urls that were successful.
        '''
        logging.info(f'getting {len(image_urls)} images from urls')
        images_array = [image_array_from_url(url, target_size=self.target_size) for url in image_urls]
        # filter out unsuccessful image urls which output None in the list of
        url_to_image = {url: img for url, img in zip(image_urls, images_array) if img is not None}
        images_array = np.array(list(url_to_image.values()))

        logging.info(f'getting features from image arrays')
        features = self.get_features(images_array, n_channels=n_channels)
        return features, list(url_to_image.keys())

    def get_features(self, images_array, n_channels=None):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        logging.info(f'preprocessing {images_array.shape[0]} images')
        images_array = self.preprocess(images_array)
        logging.info(f'computing image features')
        image_features = self.model.predict(images_array)
        if n_channels:
            logging.info(f'truncated to first {n_channels} channels')
            # if n_channels is specified, only keep that number of channels
            image_features = image_features[:, :, :, :n_channels]

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))
