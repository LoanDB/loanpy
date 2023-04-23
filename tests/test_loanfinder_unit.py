# -*- coding: utf-8 -*-
import pytest
from loanpy.loanfinder import phonetic_matches, semantic_matches
from unittest.mock import patch, call

@patch("loanpy.loanfinder.re.match", side_effect = [0, 1, 0, 0])
def test_phonetic_matches(re_match_mock, tmpdir):
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

    # 7  calls bc after matching with iggi it doesn't continue to agga
    assert re_match_mock.call_args_list == [
        call('^(i|u)(g)(g)(i|u)$', 'igig'),
        call('^(i|u)(g)(g)(i|u)$', 'iggi'),
        call('^(i|u)(i|u)(g)(g)$', 'igig'),
        call('^(i|u)(i|u)(g)(g)$', 'iggi')
    ]

@patch("loanpy.loanfinder.re.match", side_effect = [0, 1, 0, 0])
def test_phonetic_matches2(re_match_mock, tmpdir):
    """
    test if max_ad is working.
    We're specifying how many adaptations to take into account maximum
    per word. E.g. adapt.csv may have 1000 adaptations per word,
    but we want to read only 100, or 1.
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

    # 7  calls bc after matching with iggi it doesn't continue to agga
    assert re_match_mock.call_args_list == [
        call('^(i|u)(g)(g)(i|u)$', 'igig'),
        call('^(i|u)(g)(g)(i|u)$', 'iggi'),
        call('^(i|u)(i|u)(g)(g)$', 'igig'),
        call('^(i|u)(i|u)(g)(g)$', 'iggi')
    ]

@patch("loanpy.loanfinder.re.match", side_effect = [0, 1, 0, 0, 0, 0])
def test_phonetic_matches3(re_match_mock, tmpdir):
    """
    Same input dataframe as previously, but now we say max_ad=2,
    so it should loop through *all* adaptations.
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
    phonetic_matches(recipient, donor, outpath, 2)
    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tID_rc\tID_ad\n0\tRecipientese-0\tf1\n'

    # 7  calls bc after matching with iggi it doesn't continue to agga
    assert re_match_mock.call_args_list == [
        call('^(i|u)(g)(g)(i|u)$', 'igig'),
        call('^(i|u)(g)(g)(i|u)$', 'iggi'),
        call('^(i|u)(i|u)(g)(g)$', 'igig'),
        call('^(i|u)(i|u)(g)(g)$', 'iggi'),
        call('^(i|u)(i|u)(g)(g)$', 'uggu')
    ]

def test_semantic_matches(tmpdir):
    # basic test
    phmtsv = [
        ["ID", "ID_rc", "ID_ad", "lg1", "lg2"],
        ["0", "20-bla", "f53", "lg1", "lg2"],
        ["1", "87-bli", "f7", "l1", "lg2"],
    ]
    outpath = tmpdir.join("test_sem_match.tsv")
    semantic_matches(phmtsv, lambda x, y: 3, outpath)

    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tID_rc\tID_ad\tsemsim\n0\t20-bla\tf53\t3\n1\t87-bli\tf7\t3\n'

    # test with higher threshold
    semantic_matches(phmtsv, lambda x, y: 3, outpath, thresh=5)

    with open(outpath, "r") as f:
        result = f.read()
    assert result == 'ID\tID_rc\tID_ad\tsemsim\n'
