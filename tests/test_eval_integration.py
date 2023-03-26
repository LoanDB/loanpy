import json
from pathlib import Path

from loanpy.eval import eval_one

TESTFILESDIR = Path(__file__).parent.parent / "test_files"

def test_eval_one_adapt():
    """
    With repaired phonotactics and without
    """

    edicted = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
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
    assert eval_one(edicted, heur, True, 1, ()) == 0.0
    assert eval_one(edicted, heur, True, 2, ()) == 0.0
    assert eval_one(edicted, heur, True, 5, ()) == 0.0
    assert eval_one(edicted, heur, True, 1000, ()) == 0.0


    with open(TESTFILESDIR / "WOT2EAHedicted.tsv", "r") as f:
        eded = [row.split("\t") for row in f.read().strip().split("\n")]
        # always make sure the slice is an uneven number!
    with open(TESTFILESDIR / "heur.json", "r") as f:
        heur = json.load(f)


    #adapt. repair phonotactics = false, different sizes and places of slices
    assert eval_one([eded[0]] + eded[-20:], heur, True, 1, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 2, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 10, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 100, False) == 0.7
    assert eval_one([eded[0]] + eded[-20:], heur, True, 1000, False) == 0.9

    assert eval_one(eded[:51], heur, True, 1, False) == 0.44
    assert eval_one(eded[:51], heur, True, 2, False) == 0.52
    assert eval_one(eded[:51], heur, True, 10, False) == 0.8
    assert eval_one(eded[:51], heur, True, 100, False) == 0.84
    assert eval_one(eded[:51], heur, True, 1000, False) == 0.84

    assert eval_one(eded, heur, True, 1, False) == 0.02
    assert eval_one(eded, heur, True, 2, False) == 0.08
    assert eval_one(eded, heur, True, 10, False) == 0.31
    assert eval_one(eded, heur, True, 100, False) == 0.78
    assert eval_one(eded, heur, True, 1000, False) == 0.94

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

    assert eval_one(eded[:51], heur, True, 1, True) == 0.4
    assert eval_one(eded[:51], heur, True, 2, True) == 0.48
    assert eval_one(eded[:51], heur, True, 10, True) == 0.72
    assert eval_one(eded[:51], heur, True, 100, True) == 0.76
    assert eval_one(eded[:51], heur, True, 1000, True) == 0.76

    assert eval_one(eded, heur, True, 1, True) == 0.02
    assert eval_one(eded, heur, True, 2, True) == 0.08
    assert eval_one(eded, heur, True, 10, True) == 0.21
    assert eval_one(eded, heur, True, 100, True) == 0.34
    assert eval_one(eded, heur, True, 1000, True) == 0.4

with open(TESTFILESDIR / "H2EAHedicted.tsv", "r") as f:
    eded = [row.split("\t") for row in f.read().strip().split("\n")]
    heur=None
    #recstr. repair phonotactics = false, different sizes and places of slices

    assert eval_one(eded[:21], heur, False, 1) == 0.1
    assert eval_one(eded[:21], heur, False, 2) == 0.1
    assert eval_one(eded[:21], heur, False, 10) == 0.2
    assert eval_one(eded[:21], heur, False, 100) == 0.2
    assert eval_one(eded[:21], heur, False, 1000) == 0.2

    assert eval_one(eded, heur, False, 1) == 0.03
    assert eval_one(eded, heur, False, 2) == 0.07
    assert eval_one(eded, heur, False, 10) == 0.26
    assert eval_one(eded, heur, False, 100) == 0.51
    assert eval_one(eded, heur, False, 1000) == 0.58
