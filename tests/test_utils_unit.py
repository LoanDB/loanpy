# -*- coding: utf-8 -*-
import pytest
#from pytest_mock import MockerFixture
from unittest.mock import patch

from loanpy.utils import (IPA, find_optimal_year_cutoff, cvgaps, prefilter,
is_valid_language_sequence, is_same_length_alignments, read_ipa_all,
modify_ipa_all, prod)

# Sample input data
tsv_string = """form\tsense\tyear\torigin\tLoan
a¹\teine Interjektion\t1833\t\t
á\t〈eine Interjektion〉\t1372\tLang\t
aba ×\tFlausch, Fries, Flanell, Besatz am Rock\t1556\tLang\tTrue
abajdoc\tgemischt, , Mischkorn, schmutziges Getreide, Fraß, Gekoch\
\t1320\tLang\tTrue
"""
tsv = [row.split("\t") for row in tsv_string.split("\n")]
origins = ["Lang"]

def test_find_optimal_year_cutoff_sample():
    assert find_optimal_year_cutoff(tsv, origins) == 1320

def test_find_optimal_year_cutoff_empty_input():
    with pytest.raises(ValueError):
        assert find_optimal_year_cutoff([], origins) is None

def test_find_optimal_year_cutoff_no_origins():
    assert find_optimal_year_cutoff(tsv, []) == 1320

def test_find_optimal_year_cutoff_no_matching_origins():
    non_matching_origins = ("Non-Matching-Origin",)
    assert find_optimal_year_cutoff(tsv, non_matching_origins) == 1320

def test_find_optimal_year_cutoff_single_entry():
    single_entry_tsv = "form\tsense\tyear\torigin\tLoan\na\texample\t1900\
\tProto-Finno-Ugric\tTrue"
    tsv = [row.split("\t") for row in single_entry_tsv.split("\n")]
    origins = ["Proto-Finno-Ugric"]
    assert find_optimal_year_cutoff(tsv, origins) == 1900

def test_cvgaps():
    assert cvgaps("b l a", "b l a") == ["b l a", "b l a"]
    assert cvgaps("b l -", "b l a") == ["b l V", "b l a"]
    assert cvgaps("b l a", "b l -") == ["b l a", "b l -"]
    assert cvgaps("b - a", "b l a") == ["b C a", "b l a"]
    assert cvgaps("b l a", "b - a") == ["b l a", "b - a"]

@patch("loanpy.utils.is_valid_language_sequence")
def test_prefilter1(is_valid_language_sequence_mock):
    is_valid_language_sequence_mock.return_value = True
    data = [
    ['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '2', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '3', 'x'],
    ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '4', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '4', 'x'],
    ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '5', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '5', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
    ['x', 'x', 'nl', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']
    ]


    expected1 = [
    ['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']
    ]

    expected2 = [
    ['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '1', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '6', 'x']
    ]

    assert prefilter(data, "de", "en") == expected1
    data.insert(0,
    ['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x']
    )  # reinsert header bc pop in prefilter modifies in-place
    assert prefilter(data, "en", "de") == expected2

@patch("loanpy.utils.is_valid_language_sequence")
def test_prefilter2(is_valid_language_sequence_mock):
    is_valid_language_sequence_mock.return_value = False
    data = [
    ['x', 'x', 'Language_ID', 'x', 'x', 'x', 'x', 'x', 'x', 'Cognacy', 'x'],
    ['x', 'x', 'de', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x'],
    ['x', 'x', 'en', 'x', 'x', 'x', 'x', 'x', 'x', '0', 'x']
    ]
    with pytest.raises(AssertionError):
        assert prefilter(data, "en", "de") is None

def test_is_valid_language_sequence():
    # Test case 1: Valid sequence
    data1 = [
        ["row1", "x", "en", "data1"],
        ["row2", "x", "fr", "data2"],
        ["row3", "x", "en", "data3"],
        ["row4", "x", "fr", "data4"],
    ]
    assert is_valid_language_sequence(data1, "en", "fr") == True

    # Test case 2: Invalid sequence (languages swapped)
    data2 = [
        ["row1", "fr", "data1"],
        ["row2", "en", "data2"],
        ["row3", "fr", "data3"],
        ["row4", "en", "data4"],
    ]
    assert is_valid_language_sequence(data2, "en", "fr") == False

    # Test case 3: Invalid sequence (missing target language row)
    data3 = [
        ["row1", "en", "data1"],
        ["row2", "fr", "data2"],
        ["row3", "en", "data3"],
    ]
    assert is_valid_language_sequence(data3, "en", "fr") == False

    # Test case 4: Valid sequence with different languages
    data4 = [
        ["row1", "x", "es", "data1"],
        ["row2", "x", "de", "data2"],
        ["row3", "x", "es", "data3"],
        ["row4", "x", "de", "data4"],
    ]
    assert is_valid_language_sequence(data4, "es", "de") == True

    # Test case 5: Invalid sequence with different languages
    data5 = [
        ["row1", "de", "data1"],
        ["row2", "es", "data2"],
        ["row3", "de", "data3"],
        ["row4", "es", "data4"],
    ]
    assert is_valid_language_sequence(data5, "es", "de") == False

    # Test case 6: Empty data
    data6 = []
    assert is_valid_language_sequence(data6, "en", "fr") == True

def test_is_same_length_alignments():
    data = ""
    assert is_same_length_alignments(data)
    data = [[0, 1, 2, "a b c", 4, 5], [0, 1, 2, "d e f", 4, 5]]
    assert is_same_length_alignments(data)
    data = [[0, 1, 2, "a b c", 4, 5], [0, 1, 2, "d e", 4, 5]]
    assert not is_same_length_alignments(data)
    data = [[0, 1, 2, "b c", 4, 5], [0, 1, 2, "d e f", 4, 5]]
    assert not is_same_length_alignments(data)
    data = [[0, 1, 2, "a b c", 4, 5], [0, 1, 2, "d e f", 4, 5],
           [0, 1, 2, "g h", 4, 5], [0, 1, 2, "i j", 4, 5]
           ]
    assert is_same_length_alignments(data)
    data = [[0, 1, 2, "a b c", 4, 5], [0, 1, 2, "d e f", 4, 5],
           [0, 1, 2, "g h", 4, 5], [0, 1, 2, "i", 4, 5]
           ]
    assert not is_same_length_alignments(data)

def test_read_ipa_all():
    result = read_ipa_all()
    assert isinstance(result, list)
    assert len(result) == 6492

@patch("loanpy.utils.read_ipa_all")
def test_init_ipa(read_ipa_all_mock):
    read_ipa_all_mock.return_value = [
        ["ipa", "bla", "cons"],
        ["a", "vowel", "-1"],
        ["b", "consonant", "1"],
        ["c", "consonant", "1"],
        ["d", "consonant", "1"],
        ["e", "vowel", "-1"]
        ]
    ipa = IPA()
    assert ipa.vowels == ["a", "e"]
    read_ipa_all_mock.assert_called_with()


class IPAmonkey():
    def __init__(self):
        self.vowels = ["a", "e"]
        self.get_cv_returns = iter("VCCCVCCCVCCVVVCC")
    def get_cv(self, arg):
        return next(self.get_cv_returns)

def test_get_cv():
    ipamonkey = IPAmonkey()
    assert IPA.get_cv(ipamonkey, "a") == "V"
    assert IPA.get_cv(ipamonkey, "a") == "V"
    assert IPA.get_cv(ipamonkey, "b") == "C"
    assert IPA.get_cv(ipamonkey, "c") == "C"
    assert IPA.get_cv(ipamonkey, "d") == "C"
    assert IPA.get_cv(ipamonkey, "e") == "V"
    assert IPA.get_cv(ipamonkey, "") == "C"
    assert IPA.get_cv(ipamonkey, "9") == "C"
    assert IPA.get_cv(ipamonkey, "&") == "C"

def test_get_prosody():

    ipamonkey = IPAmonkey()

    assert IPA.get_prosody(ipamonkey, "a") == "V"
    assert IPA.get_prosody(ipamonkey, "b") == "C"
    assert IPA.get_prosody(ipamonkey, "c") == "C"
    assert IPA.get_prosody(ipamonkey, "d") == "C"
    assert IPA.get_prosody(ipamonkey, "e") == "V"
    assert IPA.get_prosody(ipamonkey, "") == "C"
    assert IPA.get_prosody(ipamonkey, "9") == "C"
    assert IPA.get_prosody(ipamonkey, "&") == "C"

    assert IPA.get_prosody(ipamonkey, "a b") == "VC"
    assert IPA.get_prosody(ipamonkey, "d e") == "CV"
    assert IPA.get_prosody(ipamonkey, "e.a b.d") == "VVCC"

def test_get_clusters():
    ipamonkey = IPAmonkey()
    assert IPA.get_clusters(ipamonkey, "abcd") == "a b.c.d"
    assert IPA.get_clusters(ipamonkey, "efgh") == "e f.g.h"
    assert IPA.get_clusters(ipamonkey, "ijji") == "i j.j i"
    assert IPA.get_clusters(ipamonkey, "aejd") == "a.e j.d"
    
def test_modify_ipa_all(tmp_path):
    # Create temporary files for sound correspondence dictionary and inventories
    sc_path = tmp_path / "ipa_all.csv"
    out_path = tmp_path / "ipa_all2.csv"
    with open(sc_path, "w+", encoding="utf-8") as file:
        file.write('ipa,bla,back,cons,bla\n\
a,-,0,-,+\n\
b,-,+,+,-\n\
j,0,+,0,-\n\
wᵏ,0,+,0,-')
    #sc_path.write_text()

    expected = 'ipa,bla,back,cons,bla\n\
a,-1,0,-1,1\n\
b,-1,1,1,-1\n\
j,0,1,1,-1\n\
wᵏ,0,1,1,-1\n\
C,0,0,1,0\n\
V,0,0,-1,0'

    modify_ipa_all(sc_path, out_path)
    with open(out_path, 'r', encoding='utf-8') as file:
        assert file.read() == expected

def test_prod_empty_list():
    assert prod([]) == 1

def test_prod_single_element():
    assert prod([5]) == 5

def test_prod_two_elements():
    assert prod([2, 3]) == 6

def test_prod_multiple_elements():
    assert prod([1, 2, 3, 4]) == 24

def test_prod_negative_numbers():
    assert prod([-1, 2, -3, 4]) == 24
    assert prod([-1, 2, 3, 4]) == -24

def test_prod_zero():
    assert prod([0, 1, 2, 3]) == 0
