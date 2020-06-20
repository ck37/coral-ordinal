from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name = 'CORAL Ordinal',
    url = 'https://github.com/ck37/coral_ordinal',
    author = 'Chris Kennedy',
    author_email = 'chrisken@gmail.com',
    # Needed to actually package something
    packages = ['coral_ordinal'],
    # Needed for dependencies
    install_requires = ['numpy', 'pandas', 'tensorflow', 'scipy'],
    # *strongly* suggested for sharing
    version = '0.1',
    # The license can be anything you like
    license = 'MIT',
    description = 'TF.Keras implementation of CORAL ordinal classification output layer',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
