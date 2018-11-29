from setuptools import setup


setup(
    name="unicorn",
    version="1.0.0",
    description="UNsupervised Image Clustering via Object Recognition Networks",
    packages=["unicorn"],
    include_package_data=True,
    install_requires=["pytest", "scikit-learn", "hdbscan"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/nk-logger.git@master#egg=nk_logger"
    ],
)
