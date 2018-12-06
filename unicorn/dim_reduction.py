from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from umap import UMAP

DIM_REDUC_CONFIGS = {
    "pca": dict(n_components=8, copy=False, svd_solver="randomized"),
    "rand-proj": dict(eps=0.1),
    "tsne": dict(n_components=8, perplexity=30.0),
    "umap": dict(n_neighbors=15, min_dist=0.1, metric="euclidean"),
}

DIM_REDUC_ALGS = {
    "pca": PCA,
    "rand-proj": GaussianRandomProjection,
    "tsne": TSNE,
    "umap": UMAP,
}
