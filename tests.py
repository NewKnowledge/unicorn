import logging
from nk_unicorn import Unicorn
from nk_imagenet import ImagenetModel

# TODO write tests


def test_unicorn():
    model = ImagenetModel()
    logging.info('creating unicorn')
    unicorn = Unicorn()
    logging.info('success creating unicorn')
    logging.warning('tests are not implemented')
    assert True
