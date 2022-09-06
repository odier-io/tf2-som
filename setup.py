#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import json

from setuptools import setup

########################################################################################################################

if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tf_som', 'metadata.json'), 'r') as f:

        metadata = json.load(f)

        setup(
            name = 'tf_som',
            version = metadata['version'],
            author = ', '.join(metadata['author_names']),
            author_email = ', '.join(metadata['author_emails']),
            description = 'Tensorflow implementation of the Self Organizing Maps (SOM)',
            url = 'https://www.github.com/ami-team/tf_som/',
            license = 'CeCILL-C',
            packages = ['tf_som'],
            package_data = {'': ['*.md', '*.txt'], 'demo': ['colors.csv', 'demo.ipynb']},
            install_requires = ['h5py', 'tqdm', 'numpy', 'tensorflow'],
            platforms = 'any'
        )

########################################################################################################################
