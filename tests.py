from unicorn import Unicorn
from logging import Formatter
from nk_logger import get_logger, config_logger
import numpy as np

config_logger(prefix="unicorn", root_log_level="ERROR", formatter=Formatter())
logger = get_logger(__name__)


def test_unicorn(n_samples=1000, dim=100):
    unicorn = Unicorn()
    data = np.random.randn(n_samples, dim)
    res = unicorn.cluster(data)
    assert isinstance(res, np.ndarray)
    assert res.shape == (n_samples,)


if __name__ == "__main__":
    test_unicorn()
