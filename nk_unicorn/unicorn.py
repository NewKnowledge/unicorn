''' Classes for performing image clustering '''

import sys

import numpy as np
from keras.applications import inception_v3
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from logger import logger

from .image_utils import image_array_from_path, image_array_from_url


class ImagenetModel:

    ''' A class for featurizing images using pre-trained neural nets '''

    def __init__(self, target_size=(299, 299)):
        # TODO allow for other keras imagenet models
        self.model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
        self.target_size = target_size
        self.preprocess = inception_v3.preprocess_input
        self.decode = inception_v3.decode_predictions

    def get_features_from_paths(self, image_paths, n_channels=None):
        ''' takes a list of image filepaths and returns the features resulting from applying the imagenet model to those images '''
        images_array = np.array([image_array_from_path(fpath, target_size=self.target_size) for fpath in image_paths])
        return self.get_features(images_array, n_channels=n_channels)

    def get_features_from_urls(self, image_urls, n_channels=None):
        ''' takes a list of image urls and returns the features resulting from applying the imagenet model to 
        successfully downloaded images along with the urls that were successful.
        '''
        logger.info(f'getting {len(image_urls)} images from urls')
        images_array = [image_array_from_url(url, target_size=self.target_size) for url in image_urls]
        # filter out unsuccessful image urls which output None in the list of
        url_to_image = {url: img for url, img in zip(image_urls, images_array) if img is not None}
        images_array = np.array(list(url_to_image.values()))

        logger.info(f'getting features from image arrays')
        features = self.get_features(images_array, n_channels=n_channels)
        return features, list(url_to_image.keys())

    def get_features(self, images_array, n_channels=None):
        ''' takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array '''
        # NOTE we want to do preprocessing and predicting in batches whenever possible
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        logger.info(f'preprocessing images')
        images_array = self.preprocess(images_array)
        logger.info(f'computing image features')
        image_features = self.model.predict(images_array)
        if n_channels:
            logger.info(f'truncated to first {n_channels} channels')
            # if n_channels is specified, only keep that number of channels
            image_features = image_features[:, :, :, :n_channels]

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))


class Unicorn:

    def __init__(self, n_components=8):
        self.n_components = n_components

        # TODO options for other dim reduction (random projections, tSNE)
        self.dim_reduction_alg = PCA(n_components=self.n_components, copy=False, svd_solver='randomized')

        # TODO options for other clustering algs (kmeans, agglomerative)
        self.cluster_alg = DBSCAN(eps=35, min_samples=1)
        self.standard_scaler = StandardScaler(copy=False)

    def scale(self, data):
        return self.standard_scaler.fit_transform(data)

    def reduce_dimension(self, data):
        data = self.scale(data)
        return self.dim_reduction_alg.fit_transform(data)

    def cluster(self, data):
        data = self.reduce_dimension(data)
        return self.cluster_alg.fit_predict(data)

    # TODO incorporate below:

    # def calc_distance(self, data):
    #     ''' Calculate pairwise feature distance between images in a dataframe
    #         where rows are images and columns are features
    #     '''
    #     from scipy.spatial.distance import squareform, pdist

    #     pwise_dist_df = pd.DataFrame(
    #         squareform(
    #             pdist(feature_data)
    #         ),
    #         columns=feature_data.index,
    #         index=feature_data.index
    #     )

    #     return pwise_dist_df

    # def run_kmeans(self, feature_data, target_data):

    #     model = KMeans(n_clusters=self.n_clusters)
    #     model.fit(target_data)

    #     output_data = pd.concat(
    #         {'label': pd.Series(feature_data.index),
    #          'cluster_label': pd.Series(model.labels_)},
    #         axis=1
    #     )

    #     return output_data

    # def run_dbscan(self, feature_data, target_data):

    #     dbscn = DBSCAN(eps=35, min_samples=1).fit(target_data)

    #     output_data = pd.concat(
    #         {'label': pd.Series(feature_data.index),
    #          'cluster_label': pd.Series(dbscn.labels_)},
    #         axis=1
    #     )

    #     return output_data

    # def run_knn(self, target_data):

    #     nbrs = NearestNeighbors(
    #         n_neighbors=2,
    #         algorithm='ball_tree'
    #     ).fit(target_data)
    #     distances, indices = nbrs.kneighbors(target_data)

    #     output_data = pd.concat(
    #         {'indices_0': pd.Series(target_data.index[indices[:, 0]]),
    #          'indices_1': pd.Series(target_data.index[indices[:, 1]]),
    #          'distances': pd.Series(distances[:, 1])},
    #         axis=1
    #     ).sort_values('distances')

    #     return output_data

    # # kmeans on pairwise distance
    # pwise_dist_df = self.calc_distance(feature_data)
    # result = self.run_kmeans(feature_data, pwise_dist_df)

    # knn on image features
    # result = self.run_knn(feature_data)

    # return result, processed_feature_data
