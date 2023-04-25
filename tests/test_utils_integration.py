# -*- coding: utf-8 -*-
from loanpy.utils import IPA, prefilter

def test_prefilter1():
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

def test_ipa_init():
    ipa = IPA()
    assert len(ipa.__dict__) == 1
    assert isinstance(ipa.__dict__["vowels"], list)
    assert len(ipa.__dict__["vowels"]) == 1464
    assert all(i in ipa.vowels for i in "aeiou")
    assert not any(i in ipa.vowels for i in "jklmw")

def test_get_cv():
    ipa = IPA()
    assert all(ipa.get_cv(i)=="C" for i in "thbnwjprl")
    assert all(ipa.get_cv(i)=="V" for i in "aeiouy")

def test_get_prosody():
    ipa = IPA()
    assert ipa.get_prosody("l o l") == "CVC"
    assert ipa.get_prosody("r o f.l") == "CVCC"
    assert ipa.get_prosody("b.l a") == "CCV"
    assert ipa.get_prosody("l i l a l u") == "CVCVCV"
    assert ipa.get_prosody("b.l i.i b.l a.a") == "CCVVCCVV"
    assert ipa.get_prosody("a.h.a") == "VCV"
    assert ipa.get_prosody("a.h.a a.e") == "VCVVV"
    assert ipa.get_prosody("j") == "C"

def test_get_clusters():
    ipa = IPA()
    assert ipa.get_clusters("lol") == "l o l"
    assert ipa.get_clusters("rofl") == "r o f.l"
    assert ipa.get_clusters("bla") == "b.l a"
    assert ipa.get_clusters("lilalu") == "l i l a l u"
    assert ipa.get_clusters("bliiblaa") == "b.l i.i b.l a.a"
    assert ipa.get_clusters("aha") == "a h a"
    assert ipa.get_clusters("ahaae") == "a h a.a.e"
    assert ipa.get_clusters("j") == "j"
