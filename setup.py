#!/usr/bin/env python
# Install script for mouselab pupil tracker
# Joao Couto - November 2016

import os
from setuptools import setup

longdescription = '''Mouse pupil tracker GUI and tools.'''

setup(
    name = 'neutracker',
    version = '0.2.0',
    author = 'Cagatay Aydin',
    author_email = 'cagjony@gmail.com',
    description = (' Mouse Pupil Tracker'),
    long_description = longdescription,
    license = 'GPL',
    packages = ['neutracker'],
    install_requires=[
          'tifffile'
      ],
    entry_points = {
        'console_scripts': [
            'neutracker-gui = neutracker.gui:main',
            ]
        }
    )
