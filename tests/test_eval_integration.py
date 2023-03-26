from loanpy.eval import eval_one

def test_eval_one_adapt():
    """
    No heuristics
    """
    #edicted = Path(__file__).parent.parent / "test_files" / "edicted.tsv"
    #edicted = [row.split("\t") for row in edicted.strip().split("\n")]
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
    result = eval_one(edicted, heur, True, 1, ())
    assert result == 0.0
