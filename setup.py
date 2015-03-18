#!/usr/bin/env python

"""Installation script for FFOMP"""

from setuptools import setup

setup(
    name='FFOMP',
    version='0.1',
    description='Flexible fitting of model parameters',
    long_description=open('README.rst', 'r').read(),
    author='Jinmo Zhao and Gustavo E Scuseria',
    author_email='tschijnmotschau@gmail.com',
    url='tschijnmo.github.io/FFOMP',
    license='MIT',
    packages=['FFOMP', ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        ],
    install_requires=[
        'numpy>=1.9.0',
        'scipy>=0.15.0',
        'sympy>=0.7.6',
        'cclib>=1.3',
        'PyYAML>=3.11',
        ],
    )
