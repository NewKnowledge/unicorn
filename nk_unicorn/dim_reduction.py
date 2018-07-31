class SpikesortTSNE:
    def __init__(self,
                 n_components=2,
                 tsne_kwargs=None,
                 ):
        self.n_components = n_components

        # set default params for tsne alg
        self.tsne_kwargs = dict(
            iterations=1000,
            theta=0.4,
            eta=200.0,
            perplexity=100,
            random_seed=1,
            verbose=2,
            files_dir='tsne',
        )
        # overwrite defaults with provided values
        if tsne_kwargs:
            self.tsne_kwargs.update(tsne_kwargs)

    def fit_transform(self, data):
        return spikesort_tsne.tsne.t_sne(samples=data,
                                         num_dims=self.n_components,
                                         **self.tsne_kwargs)
