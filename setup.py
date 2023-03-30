"""
Setup-Script for loanpy
"""

from setuptools import setup, find_packages
from pathlib import Path

setup(
  name='loanpy',
  description='a linguistic toolkit for predicting loanword adaptation \
and sound change',
  long_description=open("README.rst").read(),
  author='Viktor Martinović',
  author_email='viktor.martinovic@hotmail.com',
  version='3.0',
  packages=find_packages(),
  include_package_data=True,
  extras_require={
  "test": ["pytest==7.1.2", "coverage==7.2.2"],
  "dev": ["wheel", "twine", "sphinx"]
  },
  keywords=['borrowing detection',
            'computational linguistics',
            'loanword adaptation',
            'sound change',
            'etymology',
            'contact linguistics'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    "Topic :: Text Processing :: Linguistic",
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3.9'
  ],

  url='https://github.com/martino-vic/loanpy',
  download_url='https://github.com/martino-vic/loanpy/archive/v.2.0-beta.tar.gz',
  license='MIT',  # https://help.github.com/articles/licensing-a-repository
  platforms=["unix", "linux", "windows"],
)
