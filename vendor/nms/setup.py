from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension(
        "nms", ["nms.pyx"], extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
    )
]

setup(name="nms", ext_modules=cythonize(extensions), include_dirs=[numpy.get_include()])
