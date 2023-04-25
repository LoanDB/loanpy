# -*- coding: utf-8 -*-
"""
Module for analyzing and processing linguistic data.

This module contains functions for analysing linguistic data, particularly
focusing on finding optimal year cutoffs, manipulating IPA data, and
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
    each given year, and finds the optimal year cutoff using the distance to
    the upper left corner of the accumulated count.

    :param tsv: A table where the first row is the header
    :type tsv: list of list of strings

    :param origins: A set of origins to be considered for counting words.
    :type origins: a set of strings

    :return: The optimal year cutoff for the dataset and origins.
    :rtype: int

    .. example::

        >>> from loanpy.utils import find_optimal_year_cutoff
        >>> tsv = [
        >>>         ['form', 'sense', 'Year', 'Etymology', 'Loan'],
        >>>         ['gulyás', 'goulash, Hungarian stew', '1800', 'unknown', ''],
        >>>         ['Tisza', 'a major river in Hungary', '1230', 'uncertain', ''],
        >>>         ['Pest', 'part of Budapest, the capital', '1241', 'Slavic', 'True'],
        >>>         ['paprika', 'ground red pepper, spice', '1598', 'Slavic', 'True']
        >>>       ]
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
    Filter dataset to keep only cogsets where both source and target languages
    occur.

    This function filters the input dataset to retain only the cogsets where
    both source and target languages are present. The filtered dataset is then
    sorted based on cogset ID and language order.

    :param data: A list of lists containing language data.
    :type data: list of list of strings

    :param srclg: The source language ID to be considered.
    :type srclg: str

    :param tgtlg: The target language ID to be considered.
    :type tgtlg: str

    :return: A filtered and sorted list of lists containing cogsets with both
             source and target languages present.
    :rtype: list of list of strings
    """

    cogids = []
    headers = data.pop(0)
    lgidx, cogidx = headers.index("Language_ID"), headers.index("Cognacy")
    # take only rows with src/tgtlg
    data = [row for row in data if row[lgidx] in {srclg, tgtlg}]
    # get list of cogids and count how often each one occurs
    cogids = Counter([row[cogidx] for row in data])
    # take only cogsets that have 2 entries
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
        data: List[List[str]], source_lang: str, target_lang: str
        ) -> bool:

    """
    Validate if the data has a valid alternating sequence of source and target
    languages.

    The data is expected to have language IDs in the second column (index 1).
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
    """
    if len(data) % 2 != 0:
        logging.info("Odd number of rows, source/target language is missing.")
        return False
    for idx, row in enumerate(data):
        expected_lang = source_lang if idx % 2 == 0 else target_lang
        if row[2] != expected_lang:
            logging.info(f"Problem in row {idx}")
            return False
    return True


def is_same_length_alignments(data: List[List[str]]) -> bool:
    """
    Check if alignments within a cogset have the same length.

    This function iterates over the input data and asserts that the alignments
    within each cogset have the same length. Alignments are expected to be in
    column 3 (index 2).

    :param data: A list of lists containing language data. No header.
    :type data: list of list of strings

    :return: True if all alignments within each cogset have the same length,
             False otherwise.
    :rtype: bool

    :raises AssertionError: If the length of the alignments within a cogset
                            does not match.
    """

    rownr = 0
    for i in range(0, len(data)-1, 2):
        first = data[i][3].split(" ")
        second = data[i+1][3].split(" ")
        try:
            assert len(first) == len(second)
        except AssertionError:
            logging.info(rownr, "\n", first, "\n", second)
            return False
    return True

def read_ipa_all() -> List[List[str]]:
    """
    This function reads the ``ipa_all.csv`` file located in the same
    directory as the module and returns the IPA data as a list of lists.

    :return: A list of lists containing IPA data read from ``ipa_all.csv``.
    :rtype: list of list of strings
    """
    module_path = Path(__file__).parent.absolute()
    data_path = module_path / 'ipa_all.csv'
    with data_path.open("r", encoding="utf-8") as f:
        return [row.split(",") for row in f.read().strip().split("\n")]

def modify_ipa_all(
        input_file: Union[str, Path], output_file: Union[str, Path]
        ) -> None:
    """
    Original file is ``ipa_all.csv`` from folder ``data`` in panphon 0.20.0
    and was copied with the permission of the author.
    The ``ipa_all.csv`` table of loanpy was created with this function.
    Following modifications are undertaken:

    #. All ``+`` signs are replaced by ``1``, all ``-`` signs by ``-1``
    #. Two phonemes are appended to the column ``ipa``,
       namely "C", and "V", meaning "any consonant", and "any vowel".
    #. Any phoneme containing "j" or "w" is redefined as a consonant

    :param input_file: The path to the file ``ipa_all.csv``.
    :type input_file: A string or a path-like object

    :param output_file: The name and path of the new csv-file that is to be
                        written.
    :type output_file: A string or a path-like object
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        header = infile.readline().strip().split(',')

        rows = []
        for line in infile:
            row = line.strip().split(',')
            # Replace "+" with 1 and "-" with -1
            row = [1 if x == '+' else -1 if x == '-' else x for x in row]

            # Check for "j" or "w" and set "cons" value to 1
            if 'j' in row[0] or 'w' in row[0]:
                row[header.index('cons')] = 1

            rows.append(row)

        # Ensure all rows have the same length
        row_length = len(header)
        assert all(len(row) == row_length for row in rows), \
        "All rows must have the same length"

        # Add C and V for any consonant or any vowel
        rows.append(['C', 0, 0, 1] + [0] * (len(header) - 4))
        rows.append(['V', 0, 0, -1] + [0] * (len(header) - 4))

    # Write the modified data to a new CSV file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(','.join(header))
        for row in rows:
            outfile.write('\n' + ','.join(str(x) for x in row))

def prod(iterable: Iterable[Union[int, float]]) -> Union[int, float]:
    """
    Calculate the product of all elements in an iterable.

    This function takes an iterable (e.g., list, tuple) as input and computes
    the product of all its elements.

    :param iterable: The input iterable containing numbers.
    :type iterable: Iterable[int] or Iterable[float]
    :return: The product of all elements in the input iterable.
    :rtype: int or float
    """
    result = 1
    for item in iterable:
        result *= item
    return result

class IPA():
    """
    Class built on loanpy's modified version of panphon's ``ipa_all.csv``
    table to handle certain tasks that require IPA-data.
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
        This method takes an IPA string (phonetic notation) as input and
        returns either "V" if the string is a vowel or "C" if it is a
        consonant, based on the set of vowels defined within the class.

        :param ipastr: An IPA string representing a phonetic character.
        :type ipastr: str
        :return: A string "V" if the input IPA string is a vowel, or "C" if it
                 is a consonant.
        :rtype: str
        """
        return "V" if ipastr in self.vowels else "C"

    def get_prosody(self, ipastr: str) -> str:
        """
        Generate a prosodic string from an IPA string.

        This function takes an IPA string as input and generates a prosody
        string by classifying each phoneme as a vowel (V) or consonant (C).

        :param ipastr: The tokenised input IPA string. Phonemes must be
                       separated by space or dot.
        :type ipastr: str

        :return: The generated prosodic string.
        :rtype: str
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
