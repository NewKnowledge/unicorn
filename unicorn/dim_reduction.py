from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection

DIM_REDUC_CONFIGS = {
    "pca": dict(
        alg=PCA, kwargs=dict(n_components=8, copy=False, svd_solver="randomized")
    ),
    "rand-proj": dict(alg=GaussianRandomProjection, kwargs=dict(eps=0.1)),
    "tsne": dict(alg=TSNE, kwargs=dict(n_components=8, perplexity=30.0)),
}
