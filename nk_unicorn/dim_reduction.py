from spikesorting_tsne import tsne as TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA


class SpikesortTSNE:
    def __init__(self,
                 n_components=3,
                 tsne_kwargs=None,):
        self.n_components = n_components

        # set default params for tsne alg
        self.tsne_kwargs = dict(
            theta = 0.3
            eta = 400.0
            num_dims = 3
            perplexity = 25
            iterations = 1500
            random_seed = 1
            verbose = 2
            files_dir='tsne',
        )
        # overwrite defaults with provided values
        if tsne_kwargs:
            self.tsne_kwargs.update(tsne_kwargs)

    def fit_transform(self, data):
        return TSNE.t_sne(samples=data,
                         num_dims=self.n_components,
                         **self.tsne_kwargs)


DIM_REDUC_CONFIGS = {
    'pca': dict(alg=PCA, kwargs=dict(n_components=8, copy=False, svd_solver='randomized')),
    'rand-proj': dict(alg=GaussianRandomProjection, kwargs=dict(eps=0.1)),
    'spikesort-tsne': dict(alg=SpikesortTSNE, kwargs=dict(n_components=3,)),
}
