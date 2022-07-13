"""integration tests for loanpy.adrc.py (2.0 BETA) with pytest 7.1.1"""

from collections import Counter, OrderedDict
from inspect import ismethod
from os import remove
from pathlib import Path

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from loanpy.adrc import Adrc, read_scdictlist

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"
PATH2SC_TEST = Path(__file__).parent / "input_files" / "sc_ad_3cogs.txt"
PATH2SC_HANDMADE = Path(__file__).parent / "input_files" / "sc_ad_handmade.txt"
PATH2SC_HANDMADE2 = Path(
    __file__).parent / "input_files" / "sc_ad_handmade2.txt"
PATH2SC_HANDMADE3 = Path(
    __file__).parent / "input_files" / "sc_rc_handmade.txt"


def test_read_scdictlist():
    """test if list of sound correspondence dictionaries is read correctly"""

    # set up: create a mock list of dicts and write it to file
    dict0 = {"dict0": "szia"}
    dict1 = {"dict1": "cső"}
    out = [dict0, dict1, dict0, dict1]
    PATH2SC_TEST = Path(__file__).parent / "test_read_scdictlist.txt"
    with open(PATH2SC_TEST, "w", encoding="utf-8") as f:
        f.write(str(out))

    # assert mock dict list is read in correctly
    assert read_scdictlist(PATH2SC_TEST) == out

    # tear down
    remove(PATH2SC_TEST)
    del dict0, dict1, out, PATH2SC_TEST


def test_move_sc():
    pass  # unit == integration test (no patches)


def test_init():
    """test if the Adrc-class is initiated properly"""

    # check if initiation without args works fine
    adrc_inst = Adrc()
    assert len(adrc_inst.__dict__) == 13

    # 5 attributes initiated in Adrc, rest inherited
    assert adrc_inst.scdict is None
    assert adrc_inst.sedict is None
    assert adrc_inst.edict is None
    assert adrc_inst.scdict_phonotactics is None
    assert adrc_inst.workflow == OrderedDict()

    # 4 attributes inherited from Qfy
    assert adrc_inst.adapting is True
    assert adrc_inst.connector == "<"
    assert adrc_inst.scdictbase == {}
    assert adrc_inst.vfb is None

    # 6 attributes inherited from Etym via Qfy
    assert adrc_inst.dfety is None
    assert adrc_inst.dfrest is None
    assert adrc_inst.inventories == {}
    ismethod(adrc_inst.distance_measure)

    # assert initiation runs correctly with non-empty params as well

    # set up fake sounndchange.txt file
    d0, d1, d2, d3 = [{'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'],
                       'n': ['n'], 't͡ʃː': ['t͡ʃ'], 'γ': ['γ'], 'ɯ': ['i']},
                      {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1, 'l<l': 1,
                       'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'γ<γ': 2},
                      {'a<a': [1, 2, 3], 'd<d': [2], 'i<ɯ': [1], 'j<j': [3],
                       'l<l': [2], 'n<n': [3], 't͡ʃ<t͡ʃː': [1], 'γ<γ': [1, 2]},
                      {'VCCVC': ['VCCVC'], 'VCVC': ['VCVC'],
                       'VCVCV': ['VCVCV']}]

    adrc_inst = Adrc(
        scdictlist=PATH2SC_TEST,
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH",
        adapting=False,
        most_frequent_phonotactics=2)

    assert len(adrc_inst.__dict__) == 13

    # assert initiation went correctly
    assert adrc_inst.scdict == d0
    assert adrc_inst.sedict == d1
    assert adrc_inst.edict == d2
    assert adrc_inst.scdict_phonotactics == d3
    assert adrc_inst.workflow == OrderedDict()

    # 4 attributes inherited from Qfy
    assert adrc_inst.adapting is False
    assert adrc_inst.connector == "<*"
    assert adrc_inst.scdictbase == {}
    assert adrc_inst.vfb is None

    # 6 attributes inherited from Etym via Qfy
    assert_frame_equal(
        adrc_inst.dfety, DataFrame(
            {"Segments_tgt": ["a γ a t͡ʃ i", "a l d a γ", "a j a n"],
             "Segments_src": ["a γ a t͡ʃː ɯ", "a l d a γ", "a j a n"],
             "CV_Segments_tgt": ["a γ a t͡ʃ i", "a l.d a γ", "a j a n"],
             "CV_Segments_src": ["a γ a t͡ʃː ɯ", "a l.d a γ", "a j a n"],
             "ProsodicStructure_tgt": ["VCVCV", "VCCVC", "VCVC"],
             "ProsodicStructure_src": ["VCVCV", "VCCVC", "VCVC"],
             "Cognacy": [1, 2, 3]}))
    assert adrc_inst.inventories["Segments"] == Counter({'a': 6, 'γ': 2,
    't͡ʃ': 1, 'i': 1, 'l': 1, 'd': 1, 'j': 1, 'n': 1})
    assert adrc_inst.inventories["CV_Segments"] == Counter({'a': 6, 'γ': 2,
    't͡ʃ': 1, 'i': 1, 'l.d': 1, 'j': 1, 'n': 1})

    assert adrc_inst.inventories["ProsodicStructure"] == Counter(
    {'VCVCV': 1, 'VCCVC': 1, 'VCVC': 1})

    ismethod(adrc_inst.distance_measure)
    assert_frame_equal(adrc_inst.dfrest,
        DataFrame({"Segments_tgt": [], "CV_Segments_tgt": [],
                   "ProsodicStructure_tgt": []}))

    # don't remove yet,
    # remove("test_soundchanges.txt")


def test_get_diff():
    """test if the difference is calculated correctly
    between the first two sound of a sound correspondence list"""

    # create instance
    adrc_inst = Adrc(
        scdictlist=PATH2SC_TEST,
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH")

    # assert
    assert adrc_inst.get_diff(
        sclistlist=[["d", "x", "$"], ["a", "x", "$"],
                    ["d", "x", "$"], ["a", "x", "$"]],
        ipa=["d", "a", "d", "a"]) == [1, 6, 1, 6]

    assert adrc_inst.get_diff(
        sclistlist=[["d", "x", "$"], ["a", "$"], ["d", "x", "$"], ["a", "$"]],
        ipa=["d", "a", "d", "a"]) == [1, float("inf"), 1, float("inf")]

    assert adrc_inst.get_diff(  # test if second exception works
                                sclistlist=[["x", "x", "$"], ["a", "x", "$"],
                                            ["x", "x", "$"], ["a", "x", "$"]],
                                ipa=["k", "a", "k", "a"]) == [9999999,
                                                              6, 9999999, 6]

    assert adrc_inst.get_diff(
        sclistlist=[["x", "x", "$"], ["x", "x", "$"],
                    ["x", "x", "$"], ["x", "x", "$"]],
        ipa=["k", "i", "k", "i"]) == [9999999] * 4

    del adrc_inst


def test_read_sc():
    """test if sound correspondences are read correctly"""

    # set up mock class, plug in mock scdict, mock tokenise, mock math.prod
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE)

    # assert
    assert adrc_inst.read_sc(
        ipa="dade", howmany=1) == [["d"], ["a"], ["d"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=2) == [["d", "tʰ"], ["a"], ["d"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=3) == [["d", "tʰ"], ["a", "e"], ["d"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=4) == [["d", "tʰ"], ["a", "e"], ["d"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=5) == [["d", "tʰ"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=6) == [["d", "tʰ"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=7) == [["d", "tʰ"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=12) == [["d", "tʰ", "t"], ["a", "e"],
                                    ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=18) == [["d", "tʰ", "t"], ["a", "e", "i"],
                                    ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=24) == [["d", "tʰ", "t", "tː"], ["a", "e", "i"],
                                    ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=36) == [["d", "tʰ", "t", "tː"], ["a", "e", "i"],
                                    ["d", "tʰ", "t"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=48) == [["d", "tʰ", "t", "tː"], ["a", "e", "i"],
                                    ["d", "tʰ", "t", "tː"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=64) == [["d", "tʰ", "t", "tː"],
                                    ["a", "e", "i", "o"],
                                    ["d", "tʰ", "t", "tː"], ["y"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=128) == [["d", "tʰ", "t", "tː"],
                                     ["a", "e", "i", "o"],
                                     ["d", "tʰ", "t", "tː"], ["y", "u"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=160) == [["d", "tʰ", "t", "tː"],
                                     ["a", "e", "i", "o", "u"],
                                     ["d", "tʰ", "t", "tː"], ["y", "u"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=240) == [["d", "tʰ", "t", "tː"],
                                     ["a", "e", "i", "o", "u"],
                                     ["d", "tʰ", "t", "tː"], ["y", "u", "e"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=99999999) == [["d", "tʰ", "t", "tː"],
                                          ["a", "e", "i", "o", "u"],
                                          ["d", "tʰ", "t", "tː"],
                                          ["y", "u", "e"]]
    assert adrc_inst.read_sc(
        ipa="dade", howmany=float("inf")) == [["d", "tʰ", "t", "tː"],
                                              ["a", "e", "i", "o", "u"],
                                              ["d", "tʰ", "t", "tː"],
                                              ["y", "u", "e"]]

    # tear down
    del adrc_inst


def test_reconstruct():
    """test if reconstructions based on sound correspondences work"""

    # test first break: some sounds are not in scdict

    # set up adrc instance
    adrc_inst = Adrc(
        adapting=False, scdictlist=Path(
            __file__).parent / "input_files" / "sc_rc_3cogs.txt")

    # assert reconstruct works when sound changes are missing from data
    assert adrc_inst.reconstruct(
        ipastr="k i k i") == "#k, i, k, i# not old"

    # assert it's actually clusterising by default
    assert adrc_inst.reconstruct(
        ipastr="k.r i.e k.r i.e") == "#k.r, i.e, k.r, i.e# not old"

    # try r can be old!
    assert adrc_inst.reconstruct(
        ipastr="k r i e k r i e") == "#k, i, e, k, i, e# not old"

    # test 2nd break: phonotactics_filter and vowelharmony_filter are False
    assert adrc_inst.reconstruct(
        ipastr="aː r uː") == "^(a) (n) (a) (a t͡ʃ i)$"

    # test same break again but param <howmany> is greater than 1 now
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2) == "^(a) (n) (a) (a t͡ʃ i|γ)$"

    # overwrite adrc_inst, now with forms_csv, to read phonotactic_inventory.
    adrc_inst = Adrc(
        adapting=False, forms_csv=PATH2FORMS,
        source_language="H", target_language="EAH",
        scdictlist=Path(__file__).parent / "input_files" / "sc_rc_3cogs.txt")

    # test with phonotactics_filter=True (a filter)
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2,
        phonotactics_filter=True) == "^a n a γ$"

    # assert reconstruct works with phonotactics_filter=True & result is empty
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=1,
        phonotactics_filter=True) == "wrong phonotactics"

    # same as previous one but with clusterised=True
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=1,
        phonotactics_filter=True) == "wrong phonotactics"

    # assert vowelharmony_filter works
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2, phonotactics_filter=False,
        vowelharmony_filter=True) == "^a n a a t͡ʃ i$|^a n a γ$"

    #filters out ^anuat͡ʃi$ out, since it has back and front vow
    assert adrc_inst.reconstruct(
        ipastr="aː r u", howmany=2,  # short u!
        phonotactics_filter=False, vowelharmony_filter=True) == "^a n u γ$"
    # test vowelharmony_filter=True, result is empty
    assert adrc_inst.reconstruct(
        ipastr="aː r u", howmany=1, phonotactics_filter=False,
        vowelharmony_filter=True) == "wrong vowel harmony"

    # vtest sort_by_nse=9999999 assert reconstr works and sorts the result by nse
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=9999999) == "^a n a γ$|^a n a a t͡ʃ i$"

    # test if sort_by_nse=1 works
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=1) == "^a n a γ$|^a n a a t͡ʃ i$"

    # test if sort_by_nse=2 works
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=2) == "^a n a γ$|^a n a a t͡ʃ i$"

    # test if sort_by_nse=9999999 works
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False,
        sort_by_nse=9999999) == "^a n a γ$|^a n a a t͡ʃ i$"

    # test if sort_by_nse=0 works
    assert adrc_inst.reconstruct(
        ipastr="aː r uː", howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=0) == '^(a) (n) (a) (a t͡ʃ i|γ)$'
    # since last 3 params are all False or 0, combinatorics is not triggered.

    del adrc_inst


def test_repair_phonotactics():
    """test if phonotactic structures are adapted correctly"""

    # create instance
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE,  # CVC and CVCCV: hard-coded in this file!
        forms_csv=PATH2FORMS,  # to extract VCVCV, VCCVC, VCVC for heuristics
        source_language="WOT", target_language="EAH")

    # assert repair_phonotactics works with max_repaired_phonotactics=1
    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=0) == [['k', 'i', 'k', 'i']]
    # but also with max_repaired_phonotactics=2
    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i",
        max_repaired_phonotactics=2) == [['k', 'i', 'k'],
                                         ['k', 'i', 'C', 'k', 'i']]

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=3,
        # only 2 strucs available
        show_workflow=True) == [['k', 'i', 'k'], ['k', 'i', 'C', 'k', 'i']]
    assert adrc_inst.workflow == OrderedDict([(
        'donor_phonotactics', 'CVCV'),
        ('predicted_phonotactics', "['CVC', 'CVCCV']")])

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=2,
        # C can be inserted before or after k
        max_paths2repaired_phonotactics=2) == [['k', 'i', 'k'],
                                               ['k', 'i', 'C', 'k', 'i'],
                                               ['k', 'i', 'k', 'C', 'i']]

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=3,
        show_workflow=True) == [['k', 'i', 'k'], ['k', 'i', 'C', 'k', 'i'],
                                ['k', 'i', 'k', 'C', 'i']]
    assert adrc_inst.workflow == OrderedDict([(
        'donor_phonotactics', 'CVCV'),
        ('predicted_phonotactics', "['CVC', 'CVCCV']")])

    # can't squeeze out more from this example, this was the max.
    assert adrc_inst.repair_phonotactics(ipastr="k i k i",
                                        max_repaired_phonotactics=9999999,
                                        # C can be inserted before or after k
                                        max_paths2repaired_phonotactics=9999999
                                        ) == [['k', 'i', 'k'],
                                              ['k', 'i', 'C', 'k', 'i'],
                                              ['k', 'i', 'k', 'C', 'i']]

    # test with different input strings now
    # prosodic_string: j=C except when between two Cs then it's V
    # so "alkjpqf" is"VCCVCCC" but "ja" would be "CV"
    # same story with "r, m, n, w" btw.
    assert adrc_inst.repair_phonotactics(
        ipastr="a l k j p q f", max_repaired_phonotactics=5, show_workflow=True) == [
        [
            'a', 'l', 'k', 'j', 'f'], [
                'a', 'l', 'j', 'f'], [
                    'a', 'l', 'V', 'k', 'j']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VCCVCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    # same input str, higher max_rep.ph., add max_p2rep.
    assert adrc_inst.repair_phonotactics(ipastr="a l k j p q f",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10,
                                        show_workflow=True) == [
                                             ['a', 'l', 'k', 'j', 'f'],
                                             ['a', 'l', 'k', 'j', 'q'],
                                             ['a', 'l', 'k', 'j', 'p'],
                                             ['a', 'k', 'j', 'f'],
                                             ['a', 'k', 'j', 'q'],
                                             ['a', 'k', 'j', 'p'],
                                             ['a', 'l', 'j', 'f'],
                                             ['a', 'l', 'j', 'q'],
                                             ['a', 'l', 'j', 'p'],
                                             ['a', 'k', 'j', 'f', 'V'],
                                             ['a', 'k', 'j', 'q', 'V'],
                                             ['a', 'k', 'j', 'p', 'V'],
                                             ['a', 'k', 'j', 'p', 'V'],
                                             ['a', 'k', 'j', 'p', 'V'],
                                             ['a', 'l', 'j', 'f', 'V'],
                                             ['a', 'l', 'j', 'q', 'V'],
                                             ['a', 'l', 'j', 'p', 'V'],
                                             ['a', 'l', 'j', 'p', 'V'],
                                             ['a', 'l', 'j', 'p', 'V']
                                        ]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VCCVCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    # almost same input str, just "j" replaced by "t" so it's always "C"
    assert adrc_inst.repair_phonotactics(
        ipastr="a l k t p q f", max_repaired_phonotactics=5, show_workflow=True) == [
        [
            'a', 'l', 'k', 'V', 't'], [
                'a', 'l', 'V', 'f'], [
                    'a', 'l', 'V', 'f', 'V']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VCCCCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    assert adrc_inst.repair_phonotactics(ipastr="a l k t p q f",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10,
                                        show_workflow=True) == [
        ['a', 'p', 'q', 'V', 'f'], ['a', 't', 'q', 'V', 'f'],
        ['a', 't', 'p', 'V', 'f'], ['a', 't', 'p', 'V', 'q'],
        ['a', 'k', 'q', 'V', 'f'], ['a', 'k', 'p', 'V', 'f'],
        ['a', 'k', 'p', 'V', 'q'], ['a', 'k', 't', 'V', 'f'],
        ['a', 'k', 't', 'V', 'q'], ['a', 'k', 't', 'V', 'q'],
        ['a', 'q', 'V', 'f'], ['a', 'p', 'V', 'f'], ['a', 'p', 'V', 'q'],
        ['a', 't', 'V', 'f'], ['a', 't', 'V', 'q'], ['a', 't', 'V', 'q'],
        ['a', 't', 'V', 'f'], ['a', 't', 'V', 'p'], ['a', 'k', 'V', 'f'],
        ['a', 'k', 'V', 'q'], ['a', 'q', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'q', 'V'], ['a', 't', 'V', 'f', 'V'],
        ['a', 't', 'V', 'q', 'V'], ['a', 't', 'V', 'q', 'V'],
        ['a', 't', 'V', 'f', 'V'], ['a', 't', 'V', 'p', 'V'],
        ['a', 't', 'V', 'p', 'V'], ['a', 't', 'V', 'p', 'V']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VCCCCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    assert adrc_inst.repair_phonotactics(ipastr="a a a",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10,
                                        show_workflow=True) == [
        ['a', 'C', 'a', 'C', 'a'], ['a', 'C', 'a', 'C'], ['a', 'C', 'a', 'C'],
        ['a', 'C', 'a', 'C'], ['a', 'C', 'C', 'a', 'C'],
        ['a', 'C', 'C', 'a', 'C'],
        ['a', 'C', 'a', 'C'], ['a', 'C', 'C', 'a', 'C'],
        ['a', 'C', 'C', 'a', 'C']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VVV'),
         ('predicted_phonotactics', "['VCVCV', 'VCVC', 'VCCVC']")])

    assert adrc_inst.repair_phonotactics(ipastr="z r r r",
                                        max_repaired_phonotactics=12,
                                        max_paths2repaired_phonotactics=2,
                                        show_workflow=True) == [
        ['V', 'r', 'r', 'V', 'r'], ['V', 'z', 'r', 'V', 'r'],
        ['V', 'r', 'V', 'r'],
        ['V', 'r', 'V', 'r'], ['V', 'r', 'V', 'r', 'V'],
        ['V', 'r', 'V', 'r', 'V']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'CCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    # test struc missing from dict and rank_closest instead, test show_workflow
    # pretend scdict_phonotactics is empty:
    adrc_inst.scdict_phonotactics = {}
    assert adrc_inst.repair_phonotactics(ipastr="k i k i",
                                        max_repaired_phonotactics=2,
                                        show_workflow=True) == [['V', 'k', 'i',
                                                                 'k', 'i'],
                                                                ['i', 'k',
                                                                 'i', 'C']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'CVCV'),
         ('predicted_phonotactics', "['VCVCV', 'VCVC']")])

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=3) == [
        ['V', 'k', 'i', 'k', 'i'], ['i', 'k', 'i', 'C'],
        ['V', 'k', 'k', 'i', 'C']]
    # first i gets deleted: kiki-kki-Vkki-VkkiC to get VCCVC

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=4) == [
        ['V', 'k', 'i', 'k', 'i'], ['i', 'k', 'i', 'C'],
        ['V', 'k', 'k', 'i', 'C']]

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k']]

    # no change
    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=3) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k']]

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=3,
        max_paths2repaired_phonotactics=2) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k'], [
                        'i', 'C', 'k', 'i', 'C'], [
                            'i', 'k', 'C', 'i', 'C']]

    assert adrc_inst.repair_phonotactics(
        ipastr="k i k i", max_repaired_phonotactics=999,
        max_paths2repaired_phonotactics=999) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k'], [
                        'i', 'C', 'k', 'i', 'C'], [
                            'i', 'k', 'C', 'i', 'C'], [
                                'V', 'C', 'k', 'i', 'k'], [
                                    'V', 'k', 'k', 'i', 'C'], [
                                        'V', 'k', 'C', 'i', 'k']]

    # test with different input strings now
    assert adrc_inst.repair_phonotactics(
        ipastr="a l k j p q f", max_repaired_phonotactics=5) == [
        ['a', 'l', 'k', 'j', 'f'], ['a', 'l', 'j', 'f'],
        ['a', 'l', 'V', 'k', 'j']]

    # almost same as before, just "j" replaced with "t" so it's always "C"
    assert adrc_inst.repair_phonotactics(
        ipastr="a l k t p q f", max_repaired_phonotactics=5) == [
        ['a', 'l', 'k', 'V', 't'], ['a', 'l', 'V', 'f'],
        ['a', 'l', 'V', 'f', 'V']]

    assert adrc_inst.repair_phonotactics(ipastr="a l k j p q f",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10
                                        ) == [
                                         ['a', 'l', 'k', 'j', 'f'],
                                         ['a', 'l', 'k', 'j', 'q'],
                                         ['a', 'l', 'k', 'j', 'p'],
                                         ['a', 'k', 'j', 'f'],
                                         ['a', 'k', 'j', 'q'],
                                         ['a', 'k', 'j', 'p'],
                                         ['a', 'l', 'j', 'f'],
                                         ['a', 'l', 'j', 'q'],
                                         ['a', 'l', 'j', 'p'],
                                         ['a', 'k', 'j', 'f', 'V'],
                                         ['a', 'k', 'j', 'q', 'V'],
                                         ['a', 'k', 'j', 'p', 'V'],
                                         ['a', 'k', 'j', 'p', 'V'],
                                         ['a', 'k', 'j', 'p', 'V'],
                                         ['a', 'l', 'j', 'f', 'V'],
                                         ['a', 'l', 'j', 'q', 'V'],
                                         ['a', 'l', 'j', 'p', 'V'],
                                         ['a', 'l', 'j', 'p', 'V'],
                                         ['a', 'l', 'j', 'p', 'V']]


    # almost same as before, just "j" replaced with "t" so it's always "C"
    assert adrc_inst.repair_phonotactics(ipastr="a l k t p q f",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10
                                        ) == [
        ['a', 'p', 'q', 'V', 'f'], ['a', 't', 'q', 'V', 'f'],
        ['a', 't', 'p', 'V', 'f'], ['a', 't', 'p', 'V', 'q'],
        ['a', 'k', 'q', 'V', 'f'], ['a', 'k', 'p', 'V', 'f'],
        ['a', 'k', 'p', 'V', 'q'], ['a', 'k', 't', 'V', 'f'],
        ['a', 'k', 't', 'V', 'q'], ['a', 'k', 't', 'V', 'q'],
        ['a', 'q', 'V', 'f'], ['a', 'p', 'V', 'f'], ['a', 'p', 'V', 'q'],
        ['a', 't', 'V', 'f'], ['a', 't', 'V', 'q'], ['a', 't', 'V', 'q'],
        ['a', 't', 'V', 'f'], ['a', 't', 'V', 'p'], ['a', 'k', 'V', 'f'],
        ['a', 'k', 'V', 'q'], ['a', 'q', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'q', 'V'], ['a', 't', 'V', 'f', 'V'],
        ['a', 't', 'V', 'q', 'V'], ['a', 't', 'V', 'q', 'V'],
        ['a', 't', 'V', 'f', 'V'], ['a', 't', 'V', 'p', 'V'],
        ['a', 't', 'V', 'p', 'V'], ['a', 't', 'V', 'p', 'V']]

    assert adrc_inst.repair_phonotactics(
        ipastr="a a a", max_repaired_phonotactics=10,
        max_paths2repaired_phonotactics=10) == [
        [
            'a', 'C', 'a', 'C', 'a'], [
                'a', 'C', 'a', 'C'], [
                    'a', 'C', 'a', 'C'], [
                        'a', 'C', 'a', 'C'], [
                            'a', 'C', 'C', 'a', 'C'], [
                                'a', 'C', 'C', 'a', 'C'], [
                                    'a', 'C', 'a', 'C'], [
                                        'a', 'C', 'C', 'a', 'C'], [
                                            'a', 'C', 'C', 'a', 'C']]

    assert adrc_inst.repair_phonotactics(
        ipastr="z r r r", max_repaired_phonotactics=12,
        max_paths2repaired_phonotactics=2) == [
        ['V', 'r', 'r', 'V', 'r'], ['V', 'z', 'r', 'V', 'r'],
        ['V', 'r', 'V', 'r'],
        ['V', 'r', 'V', 'r'], ['V', 'r', 'V', 'r', 'V'],
        ['V', 'r', 'V', 'r', 'V']]

    del adrc_inst


def test_adapt():
    """test if words are adapted correctly with sound correspondence data"""

    # basic settings
    # create instance
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE,
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH")

    # assert adapt is working
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=5,
        max_repaired_phonotactics=0) == "d a d y, d a tʰ y, d e d y, d e tʰ y, tʰ a d y"

    # change max_repaired_phonotactics to 2 from 1.
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=6,
        max_repaired_phonotactics=2
    ) == "d a d, d e d, tʰ a d, tʰ e d, d a j d y, d e j d y"

    # change max_paths2repaired_phonotactics to 2 from 1.
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2
    ) == "d a d, tʰ a d, d a j d y, tʰ a j d y, d a d j y, tʰ a d j y"

    # assert nothing changes if weights stay same relative to each other
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        deletion_cost=1,
        insertion_cost=0.49
    ) == "d a d, tʰ a d, d a j d y, tʰ a j d y, d a d j y, tʰ a d j y"

    # assert nothing changes if weights stay same relative to each other
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        deletion_cost=2,
        insertion_cost=0.98
    ) == "d a d, tʰ a d, d a j d y, tʰ a j d y, d a d j y, tʰ a d j y"

    # o is a back vowel and will be replaced by "F"
    # which in turn turns to æ
    assert adrc_inst.adapt(
        ipastr="d e d e d o",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True
    # dyd, tʰyd is twice because since paths2repaired_phonotactics=2
    # and "ded" can be reached by deleting "edo" from "dededo" OR "de.o"
    ) == "d y d y d æ, tʰ y d y d æ, d y d, tʰ y d, d y d, tʰ y d"

    # let's assume 'CVCCV' was an allowed structure
    adrc_inst.inventories["ProsodicStructure"].update(Counter(['CVCVCV']))
    # apply filter where unallowed structures are filtered out
    assert adrc_inst.adapt(
        ipastr="d e d e d o",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True
        ) == "d y d y d æ, tʰ y d y d æ"

    # test with cluster_filter=True
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE2,  # different scdictlist
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH")
    # add this so phonotactics_filter won't be empty
    adrc_inst.inventories["ProsodicStructure"].update(Counter(['CVCCV']))
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True) == "t͡ʃ a l d a"

    # let more things go through filter cluster_filter:
    adrc_inst.inventories["CV_Segments"].update(Counter("d"))
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True) == "d a l d a, t͡ʃ a l d a"

    # sort result by nse (likelihood of reconstruction)
    #adrc_inst.inventories["CV_Segments"].add("d")
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True,
        sort_by_nse=9999999) == "t͡ʃ a l d a, d a l d a"

    # test show_workflow - run adapt first, then check workflow
    assert adrc_inst.adapt(
        ipastr="d a d e",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True,
        sort_by_nse=9999999,
        show_workflow=True) == "t͡ʃ a l d a, d a l d a"

    assert adrc_inst.workflow == OrderedDict(
        [
            ('donor_phonotactics',
             'CVCV'),
            ('predicted_phonotactics',
             "['CVC', 'CVCCV']"),
            ('adapted_phonotactics',
             "[['d', 'a', 'd'], \
['d', 'a', 'C', 'd', 'e'], ['d', 'a', 'd', 'C', 'e']]"),
            ('adapted_vowelharmony',
             "[['d', 'a', 'd'], ['d', 'a', 'C', 'd', 'e'], \
['d', 'a', 'd', 'C', 'e']]"),
            ('before_combinatorics',
             "[[['d', 't͡ʃ'], ['a', 'e'], ['d', 't͡ʃ']], \
[['d', 't͡ʃ'], ['a', 'e'], ['l'], ['d', 't͡ʃ'], ['e', 'a']], \
[['d', 't͡ʃ'], ['a', 'e'], ['d', 't͡ʃ'], ['l'], ['e', 'a']]]")])

    # tear down
    del adrc_inst


def test_get_nse():
    """test if normalised sum of examples incl. workflow is calculated well"""
    # assert with mode=="adapt" (default)
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE,
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH")

    # assert with show_workflow=True
    assert adrc_inst.get_nse("dade", "dady") == (
        33.25, 133, "[1, 6, 1, 125]", "['d<d', 'a<a', 'd<d', 'e<y']")

    # assert with mode=="reconstruct"
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE3,
        forms_csv=PATH2FORMS,
        source_language="H", target_language="EAH",
        adapting=False)

    assert adrc_inst.get_nse("ɟ ɒ l o ɡ", "j ɑ l.k ɑ") == (
        6.67, 40, "[10, 9, 8, 7, 6, 0]",
        "['#-<*-', '#ɟ<*j', 'ɒ<*ɑ', 'l<*l.k', 'o<*ɑ', 'ɡ#<*-']")

    # tear down
    del adrc_inst
