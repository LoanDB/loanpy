"""
Setup-Script for loanpy
"""

from setuptools import setup, find_packages
from pathlib import Path

setup(
  name='loanpy',
  description='a linguistic toolkit for predicting loanword adaptation \
and historical reconstructions',
  long_description=open("README.rst").read(),
  author='Viktor Martinović',
  author_email='viktor.martinovic@hotmail.com',
  version='2.0.3',
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  include_package_data=True,
  install_requires=[
    "appdirs==1.4.4",
    "colorlog==6.6.0",
    "csvw==2.0.0",
    "cycler==0.11.0",
    "editdistance==0.6.0",
    "fonttools==4.33.3",
    "gensim==4.2.0",
    "ipatok==0.4.1",
    "isodate==0.6.1",
    "kiwisolver==1.4.2",
    "latexcodec==2.0.1",
    "matplotlib==3.5.2",
    "munkres==1.1.4",
    "networkx==2.8.3",
    "numpy==1.22.4",
    "packaging==21.3",
    "pandas==1.4.2",
    "panphon==0.20.0",
    "Pillow==9.1.1",
    "pybtex==0.24.0",
    "pyparsing==3.0.9",
    "python-dateutil==2.8.2",
    "pytz==2022.1",
    "PyYAML==6.0",
    "regex==2022.6.2",
    "rfc3986==1.5.0",
    "scipy==1.8.1",
    "six==1.16.0",
    "smart-open==6.0.0",
    "tabulate==0.8.9",
    "unicodecsv==0.14.1",
    "uritemplate==4.1.1",
          ],
  extras_require={
  "test": ["pytest==7.1.2"], "dev": ["wheel", "twine", "sphinx"]
  },
  keywords=['borrowing detection',
            'computational linguistics',
            'loanword adaptation',
            'historical reconstruction',
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
  license='afl-3.0',  # https://help.github.com/articles/licensing-a-repository
  platforms=["unix", "linux", "windows"],
)
