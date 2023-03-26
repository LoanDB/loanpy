import json
from pathlib import Path

from loanpy.eval import eval_one

TESTFILESDIR = Path(__file__).parent.parent / "test_files"

def test_eval_one_adapt():
    """
    No heuristics
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

    with open(TESTFILESDIR / "edicted.tsv", "r") as f:
        eded = [row.split("\t") for row in f.read().strip().split("\n")[:51]]
    with open(TESTFILESDIR / "heur.json", "r") as f:
        heur = json.load(f)

    assert eval_one(eded, heur, True, 1) == 0.0
