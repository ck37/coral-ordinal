import setuptools
from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(_here, 'coral_ordinal', 'version.py')) as f:
    exec(f.read(), version)
    
    
with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'coral-ordinal',
    url = 'https://github.com/ck37/coral-ordinal',
    author = 'Chris Kennedy, Stephen Matthews, Georg M. Goerg',
    author_email = 'chrisken@gmail.com',
    packages = setuptools.find_packages(),
    install_requires = ['numpy', 'tensorflow>=2.2'],
    version = version['__version__'],
    long_description_content_type = "text/markdown",
    license = 'MIT',
    description = 'Tensorflow Keras implementation of CORAL ordinal regression output layer, loss, activation, and metrics',
    long_description = long_description,
    python_requires = '>=3.6',
)
