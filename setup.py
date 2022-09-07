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
                url = 'https://www.github.com/ami-team/tf_som/',
                license = 'CeCILL-C',
                packages = ['tf_som'],
                package_data = {'': ['*.md', '*.txt'], 'demo': ['colors.csv', 'demo.ipynb']},
                install_requires = ['h5py', 'tqdm', 'numpy', 'tensorflow'],
                platforms = 'any',
            )

########################################################################################################################
