import logging

from nk_unicorn import ImagenetModel, Unicorn


def test_nothing():
    logging.warning('tests are not implemented, this is a no-op placeholder')
    assert True


# TODO write tests
# def test_unicorn():
#     unicorn = Unicorn()
#     from glob import glob
#     image_paths = glob('images/*.jpg')
#     print('image paths to cluster:', image_paths)
#     image_net = ImagenetModel()
#     dat = image_net.get_features_from_paths(image_paths)
#     print('shape of array data to cluster:', dat.shape)
#     clusters = unicorn.cluster(dat)
#     print('clusters:', clusters)
