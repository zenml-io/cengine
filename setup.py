#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is used to create the package we'll publish to PyPI.

.. currentmodule:: setup.py
.. moduleauthor:: maiot GmbH <support@maiot.io>
"""

import importlib.util
import os
from pathlib import Path
from setuptools import setup, find_packages
from codecs import open  # Use a consistent encoding.
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get the base version from the library.  (We'll find it in the `version.py`
# file in the src directory, but we'll bypass actually loading up the library.)
vspec = importlib.util.spec_from_file_location(
    "version",
    str(Path(__file__).resolve().parent /
        'ce_standards' / "version.py")
)
vmod = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(vmod)
version = getattr(vmod, '__version__')

# If the environment has a build number set...
if os.getenv('buildnum') is not None:
    # ...append it to the version.
    version = "{version}.{buildnum}".format(
        version=version,
        buildnum=os.getenv('buildnum')
    )

setup(
    name='cengine',
    description="This is the maiot Core Engine.",
    long_description=long_description,
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=version,
    install_requires=[
        # Include dependencies here
        'click>=7.0,<8',
        'nbformat>=5.0.4',
        'panel==0.8.3',
        'plotly==4.0.0',
        'tabulate==0.8.7',
        'certifi>=14.05.14',
        'six>=1.10',
        'python_dateutil>=2.5.3',
        'py==1.8.1',
        'urllib3>=1.15.1',
        'apache_beam[gcp]==2.17',
        'pyarrow==0.15.1',
        'pandas==0.24.2',
        'tensorflow==2.1.0',
        'tensorflow_serving_api==2.1.0',
        'tensorflow_model_analysis==0.21.4',
        'tfx-bsl==0.21.4',
        'absl-py==0.8.1',
        'avro_python3==1.8.1'
    ],
    entry_points="""
    [console_scripts]
    cengine=ce_cli.cli:cli
    """,
    python_requires=">=3.5.*",
    license='Proprietary',  # noqa
    author='maiot GmbH',
    author_email='support@maiot.io',
    url='https://docs.maiot.io/',
    keywords=[
        "deep", "learning", "production", "machine", "pipeline"
    ],
    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for.
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',

        # Pick your license.  (It should match "license" above.)

        '''License :: Apache License 2.0''',  # noqa
        # noqa
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True
)
