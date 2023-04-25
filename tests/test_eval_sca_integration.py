# -*- coding: utf-8 -*-
import csv
import json
from pathlib import Path

from loanpy.eval_sca import eval_one, eval_all

TESTFILESDIR = Path(__file__).parent / "test_files"

def test_evaluate_all_returns_expected_output():

    eded = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
                  [0, 1, 'WOT', 'a ɣ a t͡ʃː ɯ', 'VCVCV'],
                  [1, 1, 'EAH', 'a ɣ a t͡ʃ i', 'VCVCV'],
                  [2, 2, 'WOT', 'a l d a ɣ', 'VCCVC'],
                  [3, 2, 'EAH', 'a l d a ɣ', 'VCCVC']
                ]
    heur = {"a": ["a", "e", "i", "o", "u"], "ɣ": ["g", "k", "h", "j", "w"],
            "t͡ʃː": ["t", "s", "ʃ", "t͡s", "z"], "ɯ": ["o", "u", "i", "a", "e"],
            "i": ["i", "e", "a", "u", "o"], "l": ["r", "j", "w", "h", "x"],
            "d": ["d", "t", "v", "w", "k"]
            }

    fp_vs_tp = eval_all(
        intable=eded, heur=heur, adapt=True, guess_list=[1, 2, 5, 1000]
        )
    assert fp_vs_tp == [(0.0, 0.0), (0.0, 0.0), (0.01, 0.0), (1.0, 0.0)]

def test_eval_one_adapt():
    """
    With repaired phonotactics and without
    """

    intable = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
                  [0, 1, 'WOT', 'a ɣ a t͡ʃː ɯ', 'VCVCV'],
                  [1, 1, 'EAH', 'a ɣ a t͡ʃ i', 'VCVCV'],
                  [2, 2, 'WOT', 'a l d a ɣ', 'VCCVC'],
                  [3, 2, 'EAH', 'a l d a ɣ', 'VCCVC']
                ]
    heur = {"a": ["a", "e", "i", "o", "u"], "ɣ": ["g", "k", "h", "j", "w"],
            "t͡ʃː": ["t", "s", "ʃ", "t͡s", "z"], "ɯ": ["o", "u", "i", "a", "e"],
            "i": ["i", "e", "a", "u", "o"], "l": ["r", "j", "w", "h", "x"],
            "d": ["d", "t", "v", "w", "k"]
            }

    # assert results
    assert eval_one(intable, heur, True, 1, ()) == 0.0
    assert eval_one(intable, heur, True, 2, ()) == 0.0
    assert eval_one(intable, heur, True, 5, ()) == 0.0
    assert eval_one(intable, heur, True, 1000, ()) == 0.0


    with open(TESTFILESDIR / "WOT2EAHedicted.tsv", "r", encoding='utf-8') as f:
        eded = [row.split("\t") for row in f.read().strip().split("\n")]
        # always make sure the slice is an uneven number!
    with open(TESTFILESDIR / "heur.json", "r", encoding='utf-8') as f:
        heur = json.load(f)


    #adapt. repair phonotactics = false, different sizes and places of slices
    assert eval_one([eded[0]] + eded[-20:], heur, True, 1, False) == 0.6
    assert eval_one([eded[0]] + eded[-20:], heur, True, 2, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 10, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 100, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 1000, False) == 0.9

    assert eval_one(eded[:51], heur, True, 1, False) == 0.68
    assert eval_one(eded[:51], heur, True, 2, False) == 0.68
    assert eval_one(eded[:51], heur, True, 10, False) == 0.8
    assert eval_one(eded[:51], heur, True, 100, False) == 0.84
    assert eval_one(eded[:51], heur, True, 1000, False) == 0.84

    assert eval_one(eded, heur, True, 1, False) == 0.65
    assert eval_one(eded, heur, True, 2, False) == 0.76
    assert eval_one(eded, heur, True, 10, False) == 0.86
    assert eval_one(eded, heur, True, 100, False) == 0.92
    assert eval_one(eded, heur, True, 1000, False) == 0.95

    #adapt. repair phonotactics = false, different sizes and places of slices
    assert eval_one(eded[:21], heur, True, 1, True) == 0.2
    assert eval_one(eded[:21], heur, True, 2, True) == 0.2
    assert eval_one(eded[:21], heur, True, 10, True) == 0.3
    assert eval_one(eded[:21], heur, True, 100, True) == 0.3
    assert eval_one(eded[:21], heur, True, 1000, True) == 0.3

    assert eval_one([eded[0]] + eded[-20:], heur, True, 1, True) == 0.4
    assert eval_one([eded[0]] + eded[-20:], heur, True, 2, True) == 0.4
    assert eval_one([eded[0]] + eded[-20:], heur, True, 10, True) == 0.4
    assert eval_one([eded[0]] + eded[-20:], heur, True, 100, True) == 0.4
    assert eval_one([eded[0]] + eded[-20:], heur, True, 1000, True) == 0.6

    assert eval_one(eded[:51], heur, True, 1, True) == 0.6
    assert eval_one(eded[:51], heur, True, 2, True) == 0.6
    assert eval_one(eded[:51], heur, True, 10, True) == 0.72
    assert eval_one(eded[:51], heur, True, 100, True) == 0.76
    assert eval_one(eded[:51], heur, True, 1000, True) == 0.76

    assert eval_one(eded, heur, True, 1, True) == 0.64
    assert eval_one(eded, heur, True, 2, True) == 0.75
    assert eval_one(eded, heur, True, 10, True) == 0.84
    assert eval_one(eded, heur, True, 100, True) == 0.9
    assert eval_one(eded, heur, True, 1000, True) == 0.92

    with open(TESTFILESDIR / "H2EAHedicted.tsv", "r", encoding='utf-8') as f:
        eded = list(csv.reader(f, delimiter="\t"))
        heur=None
    #recstr. repair phonotactics = false, different sizes and places of slices
    assert eval_one(eded[:21], heur, False, 1) == 0.1
    assert eval_one(eded[:21], heur, False, 2) == 0.1
    assert eval_one(eded[:21], heur, False, 10) == 0.2
    assert eval_one(eded[:21], heur, False, 100) == 0.2
    assert eval_one(eded[:21], heur, False, 1000) == 0.2

    assert eval_one(eded, heur, False, 1) == 0.19
    assert eval_one(eded, heur, False, 2) == 0.28
    assert eval_one(eded, heur, False, 10) == 0.42
    assert eval_one(eded, heur, False, 100) == 0.56
    assert eval_one(eded, heur, False, 1000) == 0.59
