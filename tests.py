from unicorn import Unicorn
from logging import Formatter
from nk_logger import get_logger, config_logger
import numpy as np

config_logger(prefix="unicorn", root_log_level="ERROR", formatter=Formatter())
logger = get_logger(__name__)

dim = 100
n_samples = 1000
data = np.random.randn(n_samples, dim)


def test_basic():
    unicorn = Unicorn()
    res = unicorn.cluster(data)
    assert isinstance(res, np.ndarray)
    assert res.shape == (n_samples,)
    assert isinstance(res[0], (np.int32, np.int64))


def test_algs():
    unicorn = Unicorn(dim_reduc_alg="umap", cluster_alg="hdbscan")
    res = unicorn.cluster(data)
    assert isinstance(res, np.ndarray)
    assert res.shape == (n_samples,)
    assert isinstance(res[0], (np.int32, np.int64))


# TODO add tests for algs other than defaults


if __name__ == "__main__":
    test_basic()
