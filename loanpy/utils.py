# -*- coding: utf-8 -*-
"""
Module for analyzing and processing linguistic data.

This module contains functions for analysing linguistic data, particularly
focusing on finding optimal year cutoffs, manipulating IPA data, and
processing cognate sets. It provides helper functions for reading and
processing linguistic datasets and performing various operations such as
filtering and validation.
"""
from collections import Counter
from pathlib import Path
import re

def find_optimal_year_cutoff(tsv, origins):
    """
    Determine the optimal year cutoff for a given dataset and origins.

    This function reads TSV content from a given dataset and origins,
    calculates the accumulated count of words with the specified origin until
    each given year, and finds the optimal year cutoff using the distance to
    the upper left corner of the accumulated count.

    :param tsv: A table where the first row is the header and the
    :type tsv: list of list of strings

    :param origins: A set of origins to be considered for counting words.
    :type origins: a set of strings

    :return: The optimal year cutoff for the dataset and origins.
    :rtype: int
    """
    # Step 1: Read the TSV content from a string
    data = []
    for row in tsv[1:]:
        if len(row) > 1 and row[2]:  # Check if the year value is not empty
            row_dict = dict(zip(tsv[0], row))
            data.append(row_dict)

    # Step 2: Extract years and create a set of possible integers
    possible_years = sorted({int(row["year"]) for row in data})

    # Step 3: Count words with the specified origin until each given year
    year_count_list = []
    accumulated_count = 0
    for year in possible_years:
        count = sum(1 for row in data if int(row["year"]) <= \
                    year and row["origin"] in origins)
        year_count_list.append((year, count))

    # Step 4: Convert the dictionary to a list of tuples
    year_count_list.sort()

    # Step 5: Find optimal cut-off point using distance to upper left corner
    max_count = max(count for _, count in year_count_list)
    optimal_year = None
    min_distance = float("inf")
    for year, count in year_count_list:
        distance = (year ** 2 + (max_count - count) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            optimal_year = year

    return optimal_year

def cvgaps(str1, str2):
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

def prefilter(data, srclg, tgtlg):
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
    #print(data)

    # take only rows with src/tgtlg
    data = [row for row in data if row[2] in {srclg, tgtlg}]
    #print("here1:", data)
    # get list of cogids and count how often each one occurs
    cogids = Counter([row[9] for row in data])
    #print("here2:", cogids)
    # take only cogsets that have 2 entries
    cogids = [i for i in cogids if cogids[i] == 2]  # allowedlist
    data = [row for row in data if row[9] in cogids]

    def sorting_key(row):
        col2_order = {srclg: 0, tgtlg: 1}
        return int(row[9]), col2_order.get(row[2], 2)

    data = sorted(data, key=sorting_key)
    #print(data)
    assert is_valid_language_sequence(data, srclg, tgtlg)
    return data

def is_valid_language_sequence(data, source_lang, target_lang):
    """
    Validate if the data has a valid alternating sequence of source and target
    languages.

    The data is expected to have language IDs in the second column (index 1).
    The sequence should be: source_lang, target_lang, source_lang,
    target_lang, ...

    :param data: A list of lists containing language data.
    :type data: list

    :param source_lang: The expected source language ID.
    :type source_lang: str

    :param target_lang: The expected target language ID.
    :type target_lang: str

    :return: True if the sequence is valid, False otherwise.
    :rtype: bool
    """
    if len(data) % 2 != 0:
        print("Odd number of rows, source/target language is missing.")
        return False
    for idx, row in enumerate(data):
        expected_lang = source_lang if idx % 2 == 0 else target_lang
        if row[2] != expected_lang:
            print(f"Problem in row {idx}")
            return False
    return True


def is_same_length_alignments(data):
    """
    Check if alignments within a cogset have the same length.

    This function iterates over the input data and asserts that the alignments
    within each cogset have the same length. Alignments are expected to be in
    column 3 (index 2).

    :param data: A list of lists containing language data.
    :type data: list of list of strings

    :return: True if all alignments within each cogset have the same length,
             False otherwise.
    :rtype: bool

    :raises AssertionError: If the length of the alignments within a cogset
                            does not match.
    """
    itertable = iter(data)
    rownr = 0
    while True:
        try:
            first = next(itertable)[3].split(" ")
            second = next(itertable)[3].split(" ")
            rownr += 2
            try:
                assert len(first) == len(second)
            except AssertionError:
                print(rownr, "\n", first, "\n", second)
                return False
        except StopIteration:
            break
    return True

def read_ipa_all():
    """
    Read IPA data from the 'ipa_all.csv' file.

    This function reads the 'ipa_all.csv' file located in the same
    directory as the module and returns the IPA data as a list of lists.

    :return: A list of lists containing IPA data read from 'ipa_all.csv'.
    :rtype: list of list of strings
    """
    module_path = Path(__file__).parent.absolute()
    data_path = module_path / 'ipa_all.csv'
    with data_path.open("r", encoding="utf-8") as f:
        return [row.split(",") for row in f.read().strip().split("\n")]

def get_prosody(ipastr):
    """
    Generate a prosodic string from an IPA string.

    This function takes an IPA string as input and generates a prosody string
    by classifying each phoneme as a vowel (V) or consonant (C).

    :param ipastr: The tokenised input IPA string. Phonemes must be separated
                   by space or dot.
    :type ipastr: str

    :return: The generated prosodic string.
    :rtype: str
    """
    ipa_all = read_ipa_all()
    vowels = [rw[0] for rw in ipa_all if rw[ipa_all[0].index("cons")]=="-1"]
    ipastr = re.split("[ |.]", ipastr)
    return "".join(["V" if phoneme in vowels else "C" for phoneme in ipastr])

def modify_ipa_all(input_file, output_file):
    """
    Original file is from folder "data" in panphon 0.20.0
    and was copied with the permission of its author.
    The ipa_all.csv table of loanpy was created with this function.
    Following modifications are undertaken:
    1) All "+" signs are replaced by 1, all "-" signs by -1
    2) Two phonemes are appended to the column "ipa",
    namely "C", and "V": "any consonant", and "any vowel".
    3) Any phoneme containing "j" or "w" is redefined as a consonant
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

def prod(iterable):
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
