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
    name = 'coral_ordinal',
    url = 'https://github.com/ck37/coral_ordinal',
    author = 'Chris Kennedy',
    author_email = 'chrisken@gmail.com',
    # Needed to actually package something
    packages = ['coral_ordinal'],
    # Needed for dependencies
    install_requires = ['numpy', 'pandas', 'tensorflow', 'scipy'],
    version = version['__version__'],
    long_description_content_type = "text/markdown",
    # The license can be anything you like
    license = 'MIT',
    description = 'TF.Keras implementation of CORAL ordinal classification output layer',
    long_description = long_description
)
