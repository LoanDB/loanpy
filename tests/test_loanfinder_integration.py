# -*- coding: utf-8 -*-
from loanpy.loanfinder import phonetic_matches, semantic_matches

def test_phonetic_matches(tmpdir):
    donor = [
        ['0', 'Donorese-0_hot1-1', 'e g e g', 'VCVC', 'hot', ['igig', 'agag']],
        ['1', 'Donorese-1_dog1-1', 'e g g e', 'VCCV', 'dog', ['iggi', 'agga']]
            ]
    recipient = [
        ['0', 'Recipientese-0', 'i k k i', 'cold', '^(i|u)(g)(g)(i|u)$'],
        ['1', 'Recipientese-1', 'i i k k', 'cat', '^(i|u)(i|u)(g)(g)$']
                ]

    outpath = tmpdir.join("test_phon_match.tsv")
    phonetic_matches(donor, recipient, outpath)
    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tloanID\tadrcID\tdf\tform\
\tpredicted\tmeaning\n0\t0\t0\trecipient\ti k k i\t^(i|u)(g)(g)(i|u)$\tcold\n1\
\t0\t1\tdonor\te g g e\tiggi\tdog\n'

def test_semantic_matches():
    # Test case 3: Multiple row input
    phmtsv = [
        ["ID", "loanID", "adrcID", "df", "form", "predicted", "meaning"],
        ["0", "0", "0", "recipient", "i k k i", "^(i|u)(g)(g)(i|u)$", "cold"],
        ["1", "0", "1", "donor", "e g g e", "iggi", "dog"],
    ]
    assert semantic_matches(phmtsv, lambda x, y: [3, "x", "y"]) == \
'ID\tloanID\tadrcID\tdf\tform\tpredicted\tmeaning\tsemsim\tclosest_sem\
\n1\t0\t1\tdonor\te g g e\tiggi\tdog\t3\ty\
\n0\t0\t0\trecipient\ti k k i\t^(i|u)(g)(g)(i|u)$\tcold\t3\tx'
