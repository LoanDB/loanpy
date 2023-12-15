"""
Setup-Script for loanpy
"""

from setuptools import setup, find_packages
from pathlib import Path

setup(
  name='loanpy',
  description='a linguistic toolkit for detecting old loanwords by predicting, \
evaluating and applying changes in horizontal and vertical lexical transfers',
  long_description=open("README.rst").read(),
  author='Viktor Martinović',
  author_email='viktor.martinovic@hotmail.com',
  version='3.0.4',
  packages=find_packages(),
  data_files=[("loanpy", ["loanpy/ipa_all.csv"])],
  include_package_data=True,
  extras_require={
  "test": ["pytest>=7.1.2", "coverage>=7.2.2"],
  "dev": ["wheel", "twine", "sphinx"]
  },
  keywords=['linguistics', 'loanwords', 'language-contact',
            'sound-change-applier'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    "Topic :: Text Processing :: Linguistic",
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",

  ],
  url='https://github.com/martino-vic/loanpy',
  download_url='https://github.com/LoanpyDataHub/loanpy/archive/3.0.0.tar.gz',
  license='MIT',
  platforms=["Windows", "macOS", "Linux"],
  python_requires=">=3.7",
  project_urls={
  "documentation": "https://loanpy.readthedocs.io/en/latest/home.html",
  "citation": "https://zenodo.org/record/7893906",
  "continuous integration": "https://dl.circleci.com/status-badge/redirect/gh/LoanpyDataHub/loanpy/tree/main",
  "test coverage": "https://coveralls.io/github/LoanpyDataHub/loanpy",
  "database": "https://github.com/LoanpyDataHub"
  }
)
