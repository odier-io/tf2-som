#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import json

from setuptools import setup

########################################################################################################################

if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'README.md'), 'r') as f1:

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf_som', 'metadata.json'), 'r') as f2:

            readme = f1.read()
            metadata = json.load(f2)

            setup(
                name = 'tf2-som',
                version = metadata['version'],
                author = ', '.join(metadata['author_names']),
                author_email = ', '.join(metadata['author_emails']),
                description = 'Tensorflow 2 implementation of the Self Organizing Maps (SOM)',
                long_description = readme,
                long_description_content_type = 'text/markdown',
                keywords = ['som', 'self organizing map', 'machine learning'],
                url = 'https://odier-io.github.io/tf2-som/',
                license = 'CeCILL-C',
                packages = ['tf_som'],
                data_files = [('tf_som', ['tf_som/metadata.json'])],
                include_package_data = True,
                package_data = {'': ['*.md', '*.txt'], 'demo': ['colors.csv', 'demo.ipynb']},
                install_requires = ['h5py', 'tqdm', 'numpy', 'tensorflow>=2'],
                extras_require = {
                    'pandas': ['pandas'],
                    'astropy': ['astropy'],
                    'matplotlib': ['matplotlib'],
                }
            )

########################################################################################################################
