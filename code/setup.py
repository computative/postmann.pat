from distutils.core import setup
from Cython.Build import cythonize
from os import remove as rm

rm("/home/marius/Dokumenter/fys4411/postmann.pat/code/blocking.so")
rm("/home/marius/Dokumenter/fys4411/postmann.pat/code/blocking.c")

setup(
    ext_modules = cythonize("blocking.pyx", annotate=True),
    extra_compile_args=["-O3"],
)
