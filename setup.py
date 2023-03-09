import setuptools

setuptools.setup(
    name="gknet",
    version="1.0.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "progress",
        "tqdm",
        "matplotlib",
        "easydict",
        "scipy",
        "shapely",
        "numba",
    ],
    extras_require={"nms": ["cython"]},
)
