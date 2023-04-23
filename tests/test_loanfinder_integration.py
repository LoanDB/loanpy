# -*- coding: utf-8 -*-
from loanpy.loanfinder import phonetic_matches, semantic_matches

def test_phonetic_matches(tmpdir):
    donor = [
        ['a0', 'f0', 'igig'],
        ['a1', 'f1', 'iggi']
            ]
    recipient = [
        ['0', 'Recipientese-0', '^(i|u)(g)(g)(i|u)$'],
        ['1', 'Recipientese-1', '^(i|u)(i|u)(g)(g)$']
                ]
    outpath = tmpdir.join("test_phon_match.tsv")
    phonetic_matches(recipient, donor, outpath)
    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tID_rc\tID_ad\n0\tRecipientese-0\tf1\n'

def test_phonetic_matches2(tmpdir):
    """
    test if max_ad is working.
    specify how many adaptations to take into account maximum
    per word. E.g. adapt.csv may have 1000 adaptations per word,
    but read only 100, or 1.
    """
    donor = [
        ['a0', 'f0', 'igig'],
        ['a1', 'f1', 'iggi'],
        ['a2', 'f1', 'uggu'],
            ]
    recipient = [
        ['0', 'Recipientese-0', '^(i|u)(g)(g)(i|u)$'],
        ['1', 'Recipientese-1', '^(i|u)(i|u)(g)(g)$']
                ]
    outpath = tmpdir.join("test_phon_match.tsv")
    phonetic_matches(recipient, donor, outpath, 1)
    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tID_rc\tID_ad\n0\tRecipientese-0\tf1\n'

def test_phonetic_matches3(tmpdir):
    """
    Same input dataframe as previously, but now we say max_ad=2,
    so it should loop through *all* adaptations.
    """
    donor = [
        ['a0', 'f0', 'igig'],
        ['a1', 'f1', 'iggi'],
        ['a2', 'f1', 'agga'],
            ]
    recipient = [
        ['0', 'Recipientese-0', '^(i|u)(g)(g)(i|u)$'],
        ['1', 'Recipientese-1', '^(i|u)(i|u)(g)(g)$']
                ]
    outpath = tmpdir.join("test_phon_match.tsv")
    phonetic_matches(recipient, donor, outpath, 2)
    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tID_rc\tID_ad\n0\tRecipientese-0\tf1\n'

def test_semantic_matches():
    pass  # there was nothing mocked in unit test.
