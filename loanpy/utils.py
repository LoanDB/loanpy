# -*- coding: utf-8 -*-
"""
Module focusing on functions to support generating and preprocessing
loanpy-compatible input data.

This module contains functions for
optimal year cutoffs, manipulating IPA data, and
processing cognate sets. It provides helper functions for reading and
processing linguistic datasets and performing various operations such as
filtering and validation.
"""
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Union

logging.basicConfig(  # set up logger (instead of print)
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

def find_optimal_year_cutoff(tsv: List[List[str]], origins: Iterable) -> int:
    """
    Determine the optimal year cutoff for a given dataset and origins.

    This function reads TSV content from a given dataset and origins,
    calculates the accumulated count of words with the specified origin until
    each given year, and finds the optimal year cutoff using the euclidean
    distance to the upper left corner in a coordinate system where the
    relative increase of years is on the x-axis and the relative increase
    in the cumulative number of words is on the y-axis.

    :param tsv: A table where the first row is the header
    :type tsv: list of list of strings

    :param origins: A set of origins to be considered for counting words.
    :type origins: a set of strings

    :return: The optimal year cutoff for the dataset and origins.
    :rtype: int

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=CBMOG3bJfNCb&line=3&uniqifier=1>`_

    .. code-block:: python

        >>> from loanpy.utils import find_optimal_year_cutoff
        >>> tsv = [
        ...     ['form', 'sense', 'Year', 'Etymology', 'Loan'],
        ...     ['gulyás', 'goulash, Hungarian stew', '1800', 'unknown', ''],
        ...     ['Tisza', 'a major river in Hungary', '1230', 'uncertain', ''],
        ...     ['Pest', 'part of Budapest, the capital', '1241', 'Slavic', 'True'],
        ...     ['paprika', 'ground red pepper, spice', '1598', 'Slavic', 'True']
        ... ]
        >>> find_optimal_year_cutoff(tsv, "Slavic")
        1241
    """
    # Step 1: Read the TSV content from a string
    data = []
    h = {i: tsv[0].index(i) for i in tsv[0]}
    for row in tsv[1:]:
        if len(row) > 1 and row[h["Year"]]:  # Check if the year value is not empty
            row_dict = dict(zip(tsv[0], row))
            data.append(row_dict)

    # Step 2: Extract years and create a set of possible integers
    possible_years = sorted({int(row["Year"]) for row in data})

    # Step 3: Count words with the specified origin until each given year
    year_count_list = []
    accumulated_count = 0
    for year in possible_years:
        count = sum(1 for row in data if int(row["Year"]) <= \
                    year and row["Etymology"] in origins)
        year_count_list.append((year, count))

    # Step 4: Convert the dictionary to a list of tuples
    year_count_list.sort()

    # Step 4.1: Turn x and y axis into relative values
    start_year = year_count_list[0][0]
    end_count= year_count_list[-1]
    #print(year_count_list)
    relative_year_count_list = [(i[0]-start_year, i[1]) for i in year_count_list]
    relative_year_count_list = [
        (i[0] / end_count[0], i[1] / end_count[1]) for i in year_count_list
        ]
    #print(relative_year_count_list, year_count_list)
    # Step 5: Find optimal cut-off point using distance to upper left corner
    max_count = max(count for _, count in relative_year_count_list)
    optimal_year = None
    min_distance = float("inf")
    for year, count in relative_year_count_list:
        distance = (year ** 2 + (max_count - count) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            optimal_year_count = (year, count)
    return year_count_list[relative_year_count_list.index(optimal_year_count)][0]

def cvgaps(str1: str, str2: str) -> List[str]:
    """
    Replace gaps in the first input string based on the second input string.

    This function takes two aligned strings, replaces "-" in the first string
    with either "C" (consonant) or "V" (vowel) depending on the corresponding
    character in the second string, and returns the new strings as a list.

    :param str1: The first aligned input string.
    :type str1: str

    :param str2: The second aligned input string.
    :type str2: str

    :return: A list containing the modified first string and the unchanged
             second string.
    :rtype: list of strings

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=kp2qNEflhtn4&line=1&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import cvgaps
        >>> cvgaps("b l -", "b l a")
        ['b l V', 'b l a']
        >>> cvgaps("b - a", "b l a")
        ['b C a', 'b l a']

    """
    ipa_all = read_ipa_all()
    vow = [rw[0] for rw in ipa_all if rw[ipa_all[0].index("cons")]=="-1"]
    new = []
    cnt = 0
    for i, j in zip(str1.split(" "), str2.split(" ")):
        if i == "-":
            if j in vow:
                new.append("V")
            else:
                new.append("C")
        else:
            new.append(i)

    return [" ".join(new), str2]

def prefilter(data: List[List[str]], srclg: str, tgtlg: str) -> List[List[str]]:
    """
    Filter dataset to keep only cognate sets where both source and target languages
    occur.

    This function filters the input dataset to retain only the cognate sets where
    both source and target languages are present. The filtered dataset is then
    sorted based on cognate set ID and language ID.

    :param data: A list of lists containing language data. Columns
                 ``Language_ID`` and ``Cognacy`` must be provided.
    :type data: list of list of strings

    :param srclg: The source language ID to be considered.
    :type srclg: str

    :param tgtlg: The target language ID to be considered.
    :type tgtlg: str

    :return: A filtered and sorted list of lists containing cognate sets with both
             source and target languages present.
    :rtype: list of list of strings

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=UqfRHYY_hxvt&line=16&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import prefilter
        >>> data = [
        ...     ['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '2', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '3', 'x'],
        ...     ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '4', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '4', 'x'],
        ...     ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '5', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '5', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
        ...     ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']
        ... ]
        >>> prefilter(data, "de", "en")
        [['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x'],
        ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
        ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']]
    """

    cogids = []
    headers = data.pop(0)
    lgidx, cogidx = headers.index("Language_ID"), headers.index("Cognacy")
    # take only rows with src/tgtlg
    data = [row for row in data if row[lgidx] in {srclg, tgtlg}]
    # get list of cogids and count how often each one occurs
    cogids = Counter([row[cogidx] for row in data])
    # take only cognate sets that have 2 entries
    cogids = [i for i in cogids if cogids[i] == 2]  # allowedlist
    data = [row for row in data if row[cogidx] in cogids]

    def sorting_key(row):
        col2_order = {srclg: 0, tgtlg: 1}
        return int(row[cogidx]), col2_order.get(row[lgidx], 2)

    data = sorted(data, key=sorting_key)

    assert is_valid_language_sequence(data, srclg, tgtlg)
    data.insert(0, headers)
    return data

def is_valid_language_sequence(
        data: List[List[str]], source_lang: str, target_lang: str,
        idx_col_lg_id=2
        ) -> bool:

    """
    Validate if the data has a valid alternating sequence of source and target
    language.

    The data is expected to have language IDs in the third column (index 2).
    The sequence should be: source_lang, target_lang, source_lang,
    target_lang, ...

    :param data: A list of lists containing language data. No header.
    :type data: list

    :param source_lang: The expected source language ID.
    :type source_lang: str

    :param target_lang: The expected target language ID.
    :type target_lang: str

    :return: True if the sequence is valid, False otherwise.
    :rtype: bool

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=SbRE_h2wfvjx&line=9&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import is_valid_language_sequence
        >>> data = [
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']
        ... ]
        >>> is_valid_language_sequence(data, "de", "en")
        True
        >>> from loanpy.utils import is_valid_language_sequence
        >>> data = [
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ...     ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
        ...     ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
        ...     ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']
        ... ]
        >>> is_valid_language_sequence(data, "de", "en")
        2023-04-25 23:04:07,532 - INFO - Problem in row 5
        False
    """
    if len(data) % 2 != 0:
        logging.info("Odd number of rows, source/target language is missing.")
        return False
    for idx, row in enumerate(data):
        expected_lang = source_lang if idx % 2 == 0 else target_lang
        if row[idx_col_lg_id] != expected_lang:
            logging.info(f"Problem in row {idx}")
            return False
    return True


def is_same_length_alignments(data: List[List[str]]) -> bool:
    """
    Check if alignments within a cognate set have the same length.

    This function iterates over the input data and asserts that the alignments
    within each cognate set have the same length. Alignments are expected to be in
    column 4 (index 3).

    :param data: A list of lists containing language data. No header.
    :type data: list of list of strings

    :return: True if all alignments within each cognate set have the same length,
             False otherwise.
    :rtype: bool

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=5DQ3fmb0jmcZ&line=1&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import is_same_length_alignments
        >>> is_same_length_alignments([[0, 1, 2, "a - c", 4, 5], [0, 1, 2, "d e f", 4, 5]])
        True
        >>> is_same_length_alignments([[0, 1, 2, "a b c", 4, 5], [0, 1, 2, "d e", 4, 5]])
        2023-04-25 23:08:05,042 - INFO - 0
        ['a', '-', 'c']
        ['d', 'e']
        False
    """

    rownr = 0
    for i in range(0, len(data)-1, 2):
        first = data[i][3].split(" ")
        second = data[i+1][3].split(" ")
        try:
            assert len(first) == len(second)
        except AssertionError:
            logging.info(f"{rownr}\n{first}\n{second}")
            return False
    return True

def read_ipa_all() -> List[List[str]]:
    """
    This function reads the ``ipa_all.csv`` table located in the same
    directory as the loanpy-modules and returns it as a list of lists.

    :return: A list of lists containing IPA data read from ``ipa_all.csv``.
    :rtype: list of list of strings

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=59YkE8Krj86V&line=3&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import read_ipa_all
        >>> ipa_all = read_ipa_all()
        >>> type(ipa_all)
        list
        >>> len(ipa_all)
        6492
        >>> ipa_all[:2]
        [['ipa', 'syl', 'son', 'cons', 'cont', 'delrel', 'lat', 'nas',
        'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr', 'lab', 'hi', 'lo',
        'back', 'round', 'velaric', 'tense', 'long', 'hitone', 'hireg'],
        ['˩', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
        '0', '0', '0', '0', '0', '0', '0', '0', '0', '-1', '-1']]

    """
    module_path = Path(__file__).parent.absolute()
    data_path = module_path / 'ipa_all.csv'
    with data_path.open("r", encoding="utf-8") as f:
        return list(csv.reader(f))

def modify_ipa_all(
        input_file: Union[str, Path], output_file: Union[str, Path]
        ) -> None:
    """
    Original file is ``ipa_all.csv`` from folder ``data`` in `panphon 0.20.0
    <https://pypi.org/project/panphon/0.20.0/>`_
    and was copied with the permission of its author.
    The ``ipa_all.csv`` table of loanpy was created with this function.
    Following modifications are undertaken:

    #. All ``+`` signs are replaced by ``1``, all ``-`` signs by ``-1``
    #. Two phonemes are appended to the column ``ipa``,
       namely "C", and "V", meaning "any consonant", and "any vowel".
    #. Any phoneme containing "j", "w", or "ʔ" is redefined as a consonant.

    :param input_file: The path to the file ``ipa_all.csv``.
    :type input_file: A string or a path-like object

    :param output_file: The name and path of the new csv-file that is to be
                        written.
    :type output_file: A string or a path-like object

    :return: Write new file
    :rtype: None
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = list(csv.reader(infile))
        header = data[0]

        rows = []
        with open(output_file, 'w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, lineterminator='\n')
            writer.writerow(header)
            for row in data[1:]:
                # Replace "+" with 1 and "-" with -1
                row = [1 if x == '+' else -1 if x == '-' else x for x in row]

                # Check for "j" or "w" and set "cons" value to 1
                if any(i in row[0] for i in ['j', 'w', 'ʔ', 'ɹ']):
                    row[header.index('cons')] = 1

                # Ensure all rows have the same length
                assert len(row) == len(header), "Rows must have same length"
                writer.writerow(row)

            # Add C and V for any consonant or any vowel
            writer.writerow(['C', 0, 0, 1] + [0] * (len(header) - 4))
            writer.writerow(['V', 0, 0, -1] + [0] * (len(header) - 4))

def prod(iterable: Iterable[Union[int, float]]) -> Union[int, float]:
    """
    Calculate the product of all elements in an iterable.

    This function takes an iterable (e.g., list, tuple) as input and computes
    the product of all its elements. This function had to be hard-coded
    because ``from math import prod`` caused incompatibility issues with some
    python versions on certain platforms.

    :param iterable: The input iterable containing numbers.
    :type iterable: Iterable[int] or Iterable[float]
    :return: The product of all elements in the input iterable.
    :rtype: int or float

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=OeII4PN6lQzO&line=2&uniqifier=1>`__

    .. code-block:: python

       >>> from loanpy.utils import prod
       >>> prod([1, 2, 3])  # one times two times three
       6
    """
    result = 1
    for item in iterable:
        result *= item
    return result

class IPA():
    """
    Class built on loanpy's modified version of panphon's ``ipa_all.csv``
    table to handle certain tasks that require IPA-data.

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=v2LicUQFlqjh&line=3&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import IPA
        >>> ipa = IPA()
        >>> type(ipa.vowels)
        list
        >>> len(ipa.vowels)
        1464
        >>> ipa.vowels[0]
        'ʋ̥'
    """
    def __init__(self) -> None:
        """
        Read the ipa-file and define a list of vowels
        """
        ipa = read_ipa_all()
        considx = ipa[0].index("cons")
        self.vowels = [row[0] for row in ipa if row[considx] == "-1"]

    def get_cv(self, ipastr: str) -> str:
        """
        This method takes an IPA string as input and
        returns either "V" if the string is a vowel or "C" if it is a
        consonant.

        :param ipastr: An IPA string representing a phonetic character.
        :type ipastr: str
        :return: A string "V" if the input IPA string is a vowel, or "C" if it
                 is a consonant.
        :rtype: str

        `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=0byaFeL4luN3&line=1&uniqifier=1>`__

        .. code-block:: python

            >>> from loanpy.utils import IPA
            >>> ipa = IPA()
            >>> ipa.get_cv("p")
            'C'
            >>> ipa.get_cv("u")
            'V'
        """
        return "V" if ipastr in self.vowels else "C"

    def get_prosody(self, ipastr: str) -> str:
        """
        Generate a prosodic string from an IPA string.

        This function takes an IPA string as input and generates a prosodic
        string by classifying each phoneme as a vowel (V) or consonant (C).

        :param ipastr: The tokenised input IPA string. Phonemes must be
                       separated by space or dot.
        :type ipastr: str

        :return: The generated prosodic string.
        :rtype: str

        `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=jKHMmjYVmGk-&line=1&uniqifier=1>`__

        .. code-block:: python

            >>> from loanpy.utils import IPA
            >>> ipa = IPA()
            >>> ipa.get_prosody("l o l")
            'CVC'
            >>> ipa.get_prosody("r o f.l")
            'CVCC'
        """
        return "".join([self.get_cv(ph) for ph in re.split("[ |.]", ipastr)])

    def get_clusters(self, segments: Iterable[str]) -> str:
        """
        Takes a list of phonemes and segments them into consonant and vowel
        clusters, like so: "abcdeaofgh" -> ["a", "b.c.d", "e.a.o", "f.g.h"]

        :param segments: A word, ideally as a list of IPA symbols
        :type segments: iterable

        :return: Same word but with consonants and vowels clustered together
        :rtype: str

        `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=cUwxzw9_mSvQ&line=1&uniqifier=1>`__

        .. code-block:: python

            >>> from loanpy.utils import IPA
            >>> ipa = IPA()
            >>> ipa.get_clusters(["r", "a", "u", "f", "l"])
            'r a.u f.l'
        """
        out = [segments[0]]
        prev_cv = self.get_cv(segments[0])

        for i in range(1, len(segments)):
            this_cv = self.get_cv(segments[i])

            if prev_cv == this_cv:
                out[-1] += "." + segments[i]
            else:
                out.append(segments[i])

            prev_cv = this_cv

        return " ".join(out)

def scjson2tsv(jsonin: Union[str, Path], outtsv: Union[str, Path],
               outtsv_phonotactics: Union[str, Path]
               ) -> None:
    """
    Turn a computer-readable sound correspondence json-file into a
    human readbale tab separated value file (tsv).

    #. read json
    #. put information into columns
    #. write file

    :param jsonin: The name of the json-file containing the sound
                   correspondences to be converted
    :type jsonin: str or path-like object

    :param outtsv: The name of the output file containing the sound
                   correspondences. Should end in ".tsv".
    :type outtsv: str or path-like object

    :param outtsv_phonotactics: The name of the output file containing the
                   phonotactic (=prosodic) correspondences. Should end in
                   ".tsv".
    :type outtsv: str or path-like object

    :return: Write two tsv-files to the specified two output paths
    :rtype: None

    `Run in Google Colab >> <https://colab.research.google.com/drive/1JlHKfdff_yjCO8yvxiKV9xoRAiEPgarM#scrollTo=HZpkFtY_mewV&line=9&uniqifier=1>`__

    .. code-block:: python

        >>> from loanpy.utils import scjson2tsv
        >>> scjson2tsv("sc.json", "sc.tsv", "sc_p.tsv")
        >>> with open("sc.tsv", "r") as f:
        ...     print(f.read())
        sc	src	tgt	freq	CogID
        a o a	o	1	512
        a e	a	e	2	3, 4
        >>> with open("sc_p.tsv", "r") as f:
        ...     print(f.read())
        sc	src	tgt	freq	CogID
        CV CV	CV	CV	1	7
    """
    # read json
    with open(jsonin, "r") as f:
        scdict = json.load(f)
    with open(outtsv, "w+") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sc", "src", "tgt", "freq", "CogID"])
        for sc in scdict[1]:
            writer.writerow([sc] + sc.split(" ") +
                            [scdict[1][sc], ", ".join([str(i) for i in scdict[2][sc]])])

    with open(outtsv_phonotactics, "w+") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sc", "src", "tgt", "freq", "CogID"])
        for sc in scdict[4]:
            writer.writerow([sc] + sc.split(" ") +
                            [scdict[4][sc], ", ".join([str(i) for i in scdict[5][sc]])])
