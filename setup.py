from setuptools import setup


setup(
    name="nk_unicorn",
    version="1.1.0",
    description="UNsupervised Image Clustering via Object Recognition Networks",
    packages=["nk_unicorn"],
    include_package_data=True,
    install_requires=[
        "cachetools >= 2.1.0",
        "Keras >= 2.1.6",
        "numpy >= 1.13.3",
        "pandas >= 0.22.0",
        "Pillow >= 5.1.0",
        "pytest >= 3.6.2",
        "requests >= 2.18.4",
        "scikit-learn >= 0.19.1",
        "tensorflow >= 1.8.0",
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/imagenet.git@master#egg=nk_imagenet"
        "git+https://github.com/NewKnowledge/nk-logger.git@master#egg=nk_logger"
    ],
)
