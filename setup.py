#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    name='tnt',
    description="TNT for pytorch model training",
    version='0.1.0',
    python_requires='>=3.5',
    packages=find_packages(),
    #dependency_links=["https://mirrors.tencent.com/pypi/simple/"],
    entry_points={
        'console_scripts': [
            'tnt_runner=tnt.runner:main',
            'tnt_converter=tnt.converter:main',
            'tnt_comb_converter=tnt.comb_converter:main',
            'tnt_general_converter=tnt.general_converter:main',
        ],
    },
)
