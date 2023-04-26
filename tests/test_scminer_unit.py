# -*- coding: utf-8 -*-
import pytest
import json
from loanpy.scminer import get_correspondences, uralign, get_heur, get_prosodic_inventory
from pathlib import Path
import shutil

from unittest.mock import patch

def test_get_correspondences_basic1():
    input_table = [    ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
    ['0', '1', 'LG1', 'a b', 'VC'],
    ['1', '1', 'LG2', 'c d', 'CC']
]

    expected_output = [
    {'a': ['c'], 'b': ['d']},
    {'a c': 1, 'b d': 1},
    {'a c': [1], 'b d': [1]},
    {'VC': ['CC']},
    {'VC CC': 1},
    {'VC CC': [1]}
    ]
    assert get_correspondences(input_table) == expected_output

def test_get_correspondences_basic2():
    input_table = [    ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
    ['0', '1', 'LG1', 'a b', 'VC'],
    ['1', '1', 'LG2', 'c d', 'CC'],
    ['2', '2', 'LG1', 'a b', 'VC'],
    ['3', '2', 'LG2', 'c d', 'CC']
]

    expected_output = [
    {'a': ['c'], 'b': ['d']},
    {'a c': 2, 'b d': 2},
    {'a c': [1, 2], 'b d': [1, 2]},
    {'VC': ['CC']},
    {'VC CC': 2},
    {'VC CC': [1, 2]}
    ]
    assert get_correspondences(input_table) == expected_output

def test_get_correspondences_basic3():
    input_table = [    ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
    ['0', '1', 'LG1', 'a b', 'VC'],
    ['1', '1', 'LG2', 'c d', 'CC'],
    ['2', '2', 'LG1', 'a b', 'VC'],
    ['3', '2', 'LG2', 'c d', 'CC'],
    ['4', '3', 'LG1', 'a b', 'VC'],
    ['5', '3', 'LG2', 'e f', 'VC']
]

    expected_output = [
    {'a': ['c', 'e'], 'b': ['d', 'f']},
    {'a c': 2, 'b d': 2, 'a e': 1, 'b f': 1},
    {'a c': [1, 2], 'b d': [1, 2], 'a e': [3], 'b f': [3]},
    {'VC': ['CC', 'VC']},
    {'VC CC': 2, 'VC VC': 1},
    {'VC CC': [1, 2], 'VC VC': [3]}
    ]
    assert get_correspondences(input_table) == expected_output

def test_get_correspondences_with_heur():
    input_table = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
  ['0', '1', 'WOT', 'a j a n', 'VCVC'],
  ['1', '1', 'EAH', 'a j a n', 'VCVC']
]

    heur = {"a": ["e"], "j": ["w"], "n": ["m"]}
    expected_output = [
    {'a': ['a', 'e'], 'j': ['j', 'w'], 'n': ['n', 'm']},
    {'a a': 2, 'j j': 1, 'n n': 1},
    {'a a': [1], 'j j': [1], 'n n': [1]},
    {'VCVC': ['VCVC']},
    {'VCVC VCVC': 1}, {'VCVC VCVC': [1]}
    ]
    assert get_correspondences(input_table, heur) == expected_output

@pytest.mark.parametrize(
    "input_table, heur, expected_output",
    [
        (
            [
                ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
                ['0', '1', 'WOT', 'a j a n', 'VCVC'],
                ['1', '1', 'EAH', 'a j a n', 'VCVC']
            ],
            {"a": ["e"], "j": ["w"], "n": ["m"]},
            [                {'a': ['a', 'e'], 'j': ['j', 'w'], 'n': ['n', 'm']},
                {'a a': 2, 'j j': 1, 'n n': 1},
                {'a a': [1], 'j j': [1], 'n n': [1]},
                {'VCVC': ['VCVC']},
                {'VCVC VCVC': 1},
                {'VCVC VCVC': [1]}
            ]
        ),
        (
            [                ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
                ['0', '1', 'H', '#aː t͡ʃ#', 'VC'],
                ['1', '1', 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV']
            ],
            {},
            [                {'#aː': ['a.ɣ.a'], 't͡ʃ#': ['t͡ʃ']},
                {'#aː a.ɣ.a': 1, 't͡ʃ# t͡ʃ': 1},
                {'#aː a.ɣ.a': [1], 't͡ʃ# t͡ʃ': [1]},
                {'VC': ['VCVCV']},
                {'VC VCVCV': 1},
                {'VC VCVCV': [1]}
            ]
        )
    ]
)

def test_get_correspondences_parametrized(input_table, heur, expected_output):
    assert get_correspondences(input_table, heur) == expected_output

def test_uralign_same_length():
    """
    Test the uralign function when left and right strings have the same length.
    """
    left = "a b c d e"
    right = "f g h i j"
    result = uralign(left, right)
    expected = "#a b c d e# -#\nf g h i j -"
    assert result == expected

def test_uralign_left_shorter():
    """
    Test the uralign function when left string is shorter than right string.
    """
    left = "a b c"
    right = "d e f g h i"
    result = uralign(left, right)
    expected = "#a b c# -#\nd e f ghi"
    assert result == expected

def test_uralign_right_shorter():
    """
    Test the uralign function when right string is shorter than left string.
    """
    left = "a b c d e f"
    right = "g h i"
    result = uralign(left, right)
    expected = "#a b c + def#\ng h i"
    assert result == expected

@patch("loanpy.scminer.read_ipa_all")
def test_get_heur(read_ipa_all_mock):
    """
    Test function for the get_heur function.

    This function tests the get_heur function with a valid language ID,
    and a missing data file. It creates a temporary directory named
    "cldf" in the current working directory, creates a file called
    ".transcription-report.json" in this temporary directory,
    writes some test content to the file, and makes some
    assertions about the returned dictionary from the get_heur function.
    """
    # Test with a valid language code

    read_ipa_all_mock.return_value = [
['ipa', 'syl', 'son', 'cons', 'cont',
'delrel', 'lat', 'nas', 'strid', 'voi', 'sg', 'cg', 'ant', 'cor', 'distr',
'lab', 'hi', 'lo', 'back', 'round', 'velaric', 'tense', 'long', 'hitone',
'hireg'],
['a', '1', '1', '-1', '1', '-1', '-1', '-1', '0', '1', '-1', '-1', '0', '-1',
 '0', '-1', '-1', '1', '-1', '-1', '-1', '1', '-1', '0', '0'],
['b', '-1', '-1', '1', '-1', '-1', '-1', '-1', '0', '1', '-1', '-1', '1', '-1',
 '0', '1', '-1', '-1', '-1', '-1', '-1', '0', '-1', '0', '0']
 ]
    # set up
    tmp_dir = Path.cwd() / "cldf"
    if tmp_dir.exists():  # pragma: no cover
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    file_path = tmp_dir / ".transcription-report.json"

    with open(file_path, "w+", encoding='utf-8') as f:
        f.write(json.dumps({
        "by_language": {"eng": {"segments": {"a": 1, "b": 2}}}}))

    result = get_heur("eng")

    assert result == {'a': ['a', 'b'], 'b': ['b', 'a']}
    assert result["a"] == ["a", "b"]
    assert result["b"] == ["b", "a"]

    #tear down
    shutil.rmtree(tmp_dir)

    read_ipa_all_mock.assert_called_with()

    # Test with missing data file
    with pytest.raises(FileNotFoundError):
        result = get_heur("eng")

import pytest

@pytest.fixture
def data():
    return [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
  [0, 1, 'H', '#aː t͡ʃ# -#', 'VC'],
  [1, 1, 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV'],
  [2, 2, 'H', '#aː ɟ uː#', 'VCV'],
  [3, 2, 'EAH', 'a l.d a.ɣ', 'VCCVC'],
  [4, 3, 'H', '#ɒ j n', 'VCC'],
  [5, 3, 'EAH', 'a j.a n', 'VCVC']
]


def test_extract_cvcv_and_phonemes(data):
    assert set(get_prosodic_inventory(data)) == {"VCVCV", "VCCVC", "VCVC"}
