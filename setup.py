#!/usr/bin/env python
from setuptools import setup, find_packages

# pip install -e .

setup(
    name="dwd",
    version="0.1.0",
    author='Risa Ueno',
    author_email='risno@bas.ac.uk',
    packages=find_packages(exclude=['*test']),
)
