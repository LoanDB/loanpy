"""integration tests for loanpy.adrc.py (2.0 BETA) with pytest 7.1.1"""

from collections import OrderedDict
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
    with open(PATH2SC_TEST, "w") as f:
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
    assert len(adrc_inst.__dict__) == 17

    # 5 attributes initiated in Adrc, rest inherited
    assert adrc_inst.scdict is None
    assert adrc_inst.sedict is None
    assert adrc_inst.edict is None
    assert adrc_inst.scdict_phonotactics is None
    assert adrc_inst.workflow == OrderedDict()

    # 4 attributes inherited from Qfy
    assert adrc_inst.mode == "adapt"
    assert adrc_inst.connector == "<"
    assert adrc_inst.scdictbase == {}
    assert adrc_inst.vfb is None

    # 8 attributes inherited from Etym via Qfy
    assert isinstance(adrc_inst.phon2cv, dict)
    assert len(adrc_inst.phon2cv) == 6358
    assert isinstance(adrc_inst.vow2fb, dict)
    assert len(adrc_inst.vow2fb) == 1240
    assert adrc_inst.dfety is None
    assert adrc_inst.phoneme_inventory is None
    assert adrc_inst.cluster_inventory is None
    assert adrc_inst.phonotactic_inventory is None
    ismethod(adrc_inst.distance_measure)
    assert adrc_inst.forms_target_language is None

    # assert initiation runs correctly with non-empty params as well

    # set up fake sounndchange.txt file
    d0, d1, d2, d3 = [{'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'],
                       'n': ['n'], 't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']},
                      {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1, 'l<l': 1,
                       'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2},
                      {'a<a': [1, 2, 3], 'd<d': [2], 'i<ɯ': [1], 'j<j': [3],
                       'l<l': [2], 'n<n': [3], 't͡ʃ<t͡ʃː': [1], 'ɣ<ɣ': [1, 2]},
                      {'VCCVC': ['VCCVC'], 'VCVC': ['VCVC'],
                       'VCVCV': ['VCVCV']}]

    adrc_inst = Adrc(
        scdictlist=PATH2SC_TEST,
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH",
        mode="reconstruct",
        most_frequent_phonotactics=2)

    assert len(adrc_inst.__dict__) == 17

    # assert initiation went correctly
    assert adrc_inst.scdict == d0
    assert adrc_inst.sedict == d1
    assert adrc_inst.edict == d2
    assert adrc_inst.scdict_phonotactics == d3
    assert adrc_inst.workflow == OrderedDict()

    # 4 attributes inherited from Qfy
    assert adrc_inst.mode == "reconstruct"
    assert adrc_inst.connector == "<*"
    assert adrc_inst.scdictbase == {}
    assert adrc_inst.vfb is None

    # 8 attributes inherited from Etym via Qfy
    assert isinstance(adrc_inst.phon2cv, dict)
    assert len(adrc_inst.phon2cv) == 6358
    assert isinstance(adrc_inst.vow2fb, dict)
    assert len(adrc_inst.vow2fb) == 1240
    assert_frame_equal(
        adrc_inst.dfety, DataFrame(
            {"Target_Form": ["aɣat͡ʃi", "aldaɣ", "ajan"],
             "Source_Form": ["aɣat͡ʃːɯ", "aldaɣ", "ajan"],
             "Cognacy": [1, 2, 3]}))
    assert adrc_inst.phoneme_inventory == {'a', 'd', 'i', 'j',
                                           'l', 'n', 't͡ʃ', 'ɣ'}
    assert adrc_inst.cluster_inventory == {'a', 'ia', 'j', 'ld',
                                           'n', 't͡ʃ', 'ɣ'}
    assert adrc_inst.phonotactic_inventory == {'VCVCV', 'VCCVC'}
    ismethod(adrc_inst.distance_measure)
    assert adrc_inst.forms_target_language == ['aɣat͡ʃi', 'aldaɣ', 'ajan']

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
        mode="reconstruct", scdictlist=Path(
            __file__).parent / "input_files" / "sc_rc_3cogs.txt")

    # assert reconstruct works when sound changes are missing from data
    assert adrc_inst.reconstruct(
        ipastr="kiki") == "#k, i, k, i# not old"

    # assert it's actually clusterising by default
    assert adrc_inst.reconstruct(
        ipastr="kriekrie") == "#kr, ie, kr, ie# not old"

    # try clusterised=False, r can be old!
    assert adrc_inst.reconstruct(
        clusterised=False, ipastr="kriekrie") == "#k, i, e, k, i, e# not old"

    # test 2nd break: phonotactics_filter and vowelharmony_filter are False
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=False) == "^(a)(n)(a)(at͡ʃi)$"

    # test same break again but param <howmany> is greater than 1 now
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=False, howmany=2) == "^(a)(n)(a)(at͡ʃi|ɣ)$"

    # overwrite adrc_inst, now with forms_csv, to read phonotactic_inventory.
    adrc_inst = Adrc(
        mode="reconstruct", forms_csv=PATH2FORMS,
        source_language="H", target_language="EAH",
        scdictlist=Path(__file__).parent / "input_files" / "sc_rc_3cogs.txt")

    # test with phonotactics_filter=True (a filter)
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=False, howmany=2,
        phonotactics_filter=True) == "^anaɣ$"

    # assert reconstruct works with phonotactics_filter=True & result is empty
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=False, howmany=1,
        phonotactics_filter=True) == "wrong phonotactics"

    # same as previous one but with clusterised=True
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=1,
        phonotactics_filter=True) == "wrong phonotactics"

    # assert vowelharmony_filter works
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2, phonotactics_filter=False,
        vowelharmony_filter=True) == "^anaat͡ʃi$|^anaɣ$"

    # let's assume "i" was a back vowel
    adrc_inst.vow2fb["i"] = "B"
    # would would filter ^anaat͡ʃi$ out, since it has back and front vow
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2,
        phonotactics_filter=False, vowelharmony_filter=True) == "^anaɣ$"
    # test vowelharmony_filter=True, result is empty
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=1, phonotactics_filter=False,
        vowelharmony_filter=True) == "wrong vowel harmony"
    # vtear down
    adrc_inst.vow2fb["i"] = "F"

    # vtest sort_by_nse=True assert reconstr works and sorts the result by nse
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=True) == "^anaɣ$|^anaat͡ʃi$"

    # test if sort_by_nse=1 works
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=1) == "^anaɣ$|^anaat͡ʃi$"

    # test if sort_by_nse=2 works
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=2) == "^anaɣ$|^anaat͡ʃi$"

    # test if sort_by_nse=float("inf") works
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False,
        sort_by_nse=float("inf")) == "^anaɣ$|^anaat͡ʃi$"

    # test if sort_by_nse=0 works
    assert adrc_inst.reconstruct(
        ipastr="aːruː", clusterised=True, howmany=2, phonotactics_filter=False,
        vowelharmony_filter=False, sort_by_nse=0) == '^(a)(n)(a)(at͡ʃi|ɣ)$'
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
        ipastr="kiki", max_repaired_phonotactics=0) == [['k', 'i', 'k', 'i']]
    # but also with max_repaired_phonotactics=2
    assert adrc_inst.repair_phonotactics(
        ipastr="kiki",
        max_repaired_phonotactics=2) == [['k', 'i', 'k'],
                                         ['k', 'i', 'C', 'k', 'i']]

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=3,
        # only 2 strucs available
        show_workflow=True) == [['k', 'i', 'k'], ['k', 'i', 'C', 'k', 'i']]
    assert adrc_inst.workflow == OrderedDict([(
        'donor_phonotactics', 'CVCV'),
        ('predicted_phonotactics', "['CVC', 'CVCCV']")])

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=2,
        # C can be inserted before or after k
        max_paths2repaired_phonotactics=2) == [['k', 'i', 'k'],
                                               ['k', 'i', 'C', 'k', 'i'],
                                               ['k', 'i', 'k', 'C', 'i']]

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=3,
        show_workflow=True) == [['k', 'i', 'k'], ['k', 'i', 'C', 'k', 'i'],
                                ['k', 'i', 'k', 'C', 'i']]
    assert adrc_inst.workflow == OrderedDict([(
        'donor_phonotactics', 'CVCV'),
        ('predicted_phonotactics', "['CVC', 'CVCCV']")])

    # can't squeeze out more from this example, this was the max.
    assert adrc_inst.repair_phonotactics(ipastr="kiki",
                                        max_repaired_phonotactics=9999999,
                                        # C can be inserted before or after k
                                        max_paths2repaired_phonotactics=9999999
                                        ) == [['k', 'i', 'k'],
                                              ['k', 'i', 'C', 'k', 'i'],
                                              ['k', 'i', 'k', 'C', 'i']]

    # test with different input strings now
    assert adrc_inst.repair_phonotactics(
        ipastr="alkjpqf", max_repaired_phonotactics=5, show_workflow=True) == [
        [
            'a', 'l', 'k', 'V', 'j'], [
                'a', 'l', 'V', 'f'], [
                    'a', 'l', 'V', 'f', 'V']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VCCCCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    assert adrc_inst.repair_phonotactics(ipastr="alkjpqf",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10,
                                        show_workflow=True) == [
        ['a', 'p', 'q', 'V', 'f'], ['a', 'j', 'q', 'V', 'f'],
        ['a', 'j', 'p', 'V', 'f'], ['a', 'j', 'p', 'V', 'q'],
        ['a', 'k', 'q', 'V', 'f'], ['a', 'k', 'p', 'V', 'f'],
        ['a', 'k', 'p', 'V', 'q'], ['a', 'k', 'j', 'V', 'f'],
        ['a', 'k', 'j', 'V', 'q'], ['a', 'k', 'j', 'V', 'q'],
        ['a', 'q', 'V', 'f'], ['a', 'p', 'V', 'f'], ['a', 'p', 'V', 'q'],
        ['a', 'j', 'V', 'f'], ['a', 'j', 'V', 'q'], ['a', 'j', 'V', 'q'],
        ['a', 'j', 'V', 'f'], ['a', 'j', 'V', 'p'], ['a', 'k', 'V', 'f'],
        ['a', 'k', 'V', 'q'], ['a', 'q', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'q', 'V'], ['a', 'j', 'V', 'f', 'V'],
        ['a', 'j', 'V', 'q', 'V'], ['a', 'j', 'V', 'q', 'V'],
        ['a', 'j', 'V', 'f', 'V'], ['a', 'j', 'V', 'p', 'V'],
        ['a', 'j', 'V', 'p', 'V'], ['a', 'j', 'V', 'p', 'V']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'VCCCCCC'),
         ('predicted_phonotactics', "['VCCVC', 'VCVC', 'VCVCV']")])

    assert adrc_inst.repair_phonotactics(ipastr="aaa",
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

    assert adrc_inst.repair_phonotactics(ipastr="zrrr",
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
    assert adrc_inst.repair_phonotactics(ipastr="kiki",
                                        max_repaired_phonotactics=2,
                                        show_workflow=True) == [['V', 'k', 'i',
                                                                 'k', 'i'],
                                                                ['i', 'k',
                                                                 'i', 'C']]
    assert adrc_inst.workflow == OrderedDict(
        [('donor_phonotactics', 'CVCV'),
         ('predicted_phonotactics', "['VCVCV', 'VCVC']")])

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=3) == [
        ['V', 'k', 'i', 'k', 'i'], ['i', 'k', 'i', 'C'],
        ['V', 'k', 'k', 'i', 'C']]
    # first i gets deleted: kiki-kki-Vkki-VkkiC to get VCCVC

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=4) == [
        ['V', 'k', 'i', 'k', 'i'], ['i', 'k', 'i', 'C'],
        ['V', 'k', 'k', 'i', 'C']]

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k']]

    # no change
    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=3) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k']]

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=3,
        max_paths2repaired_phonotactics=2) == [
        [
            'V', 'k', 'i', 'k', 'i'], [
                'i', 'k', 'i', 'C'], [
                    'V', 'k', 'i', 'k'], [
                        'i', 'C', 'k', 'i', 'C'], [
                            'i', 'k', 'C', 'i', 'C']]

    assert adrc_inst.repair_phonotactics(
        ipastr="kiki", max_repaired_phonotactics=999,
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
        ipastr="alkjpqf", max_repaired_phonotactics=5) == [
        ['a', 'l', 'k', 'V', 'j'], ['a', 'l', 'V', 'f'],
        ['a', 'l', 'V', 'f', 'V']]

    assert adrc_inst.repair_phonotactics(ipastr="alkjpqf",
                                        max_repaired_phonotactics=10,
                                        max_paths2repaired_phonotactics=10
                                        ) == [
        ['a', 'p', 'q', 'V', 'f'], ['a', 'j', 'q', 'V', 'f'],
        ['a', 'j', 'p', 'V', 'f'], ['a', 'j', 'p', 'V', 'q'],
        ['a', 'k', 'q', 'V', 'f'], ['a', 'k', 'p', 'V', 'f'],
        ['a', 'k', 'p', 'V', 'q'], ['a', 'k', 'j', 'V', 'f'],
        ['a', 'k', 'j', 'V', 'q'], ['a', 'k', 'j', 'V', 'q'],
        ['a', 'q', 'V', 'f'], ['a', 'p', 'V', 'f'], ['a', 'p', 'V', 'q'],
        ['a', 'j', 'V', 'f'], ['a', 'j', 'V', 'q'], ['a', 'j', 'V', 'q'],
        ['a', 'j', 'V', 'f'], ['a', 'j', 'V', 'p'], ['a', 'k', 'V', 'f'],
        ['a', 'k', 'V', 'q'], ['a', 'q', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'f', 'V'],
        ['a', 'p', 'V', 'q', 'V'], ['a', 'j', 'V', 'f', 'V'],
        ['a', 'j', 'V', 'q', 'V'], ['a', 'j', 'V', 'q', 'V'],
        ['a', 'j', 'V', 'f', 'V'], ['a', 'j', 'V', 'p', 'V'],
        ['a', 'j', 'V', 'p', 'V'], ['a', 'j', 'V', 'p', 'V']]

    assert adrc_inst.repair_phonotactics(
        ipastr="aaa", max_repaired_phonotactics=10,
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
        ipastr="zrrr", max_repaired_phonotactics=12,
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
        ipastr="dade",
        howmany=5,
        max_repaired_phonotactics=0) == "dady, datʰy, dedy, detʰy, tʰady"

    # change max_repaired_phonotactics to 2 from 1.
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=6,
        max_repaired_phonotactics=2
    ) == "dad, ded, tʰad, tʰed, dajdy, dejdy"

    # change max_paths2repaired_phonotactics to 2 from 1.
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2
    ) == "dad, tʰad, dajdy, tʰajdy, dadjy, tʰadjy"

    # assert nothing changes if weights stay same relative to each other
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        deletion_cost=1,
        insertion_cost=0.49
    ) == "dad, tʰad, dajdy, tʰajdy, dadjy, tʰadjy"

    # assert nothing changes if weights stay same relative to each other
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        deletion_cost=2,
        insertion_cost=0.98
    ) == "dad, tʰad, dajdy, tʰajdy, dadjy, tʰadjy"

    # let's assume e is a back vowel and repair vowel harmony
    adrc_inst.vow2fb["e"] = "B"
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True
    ) == "dad, tʰad, dajdæ, tʰajdæ, dujdy, tʰujdy"

    # let's assume 'CVCCV' was an allowed structure
    adrc_inst.phonotactic_inventory.add('CVCCV')
    # apply filter where unallowed structures are filtered out
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=6,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True
        ) == "dajdæ, tʰajdæ, dujdy, tʰujdy, dadjæ, tʰadjæ"

    # test with cluster_filter=True
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE2,  # different scdictlist
        forms_csv=PATH2FORMS,
        source_language="WOT", target_language="EAH")
    # add this so phonotactics_filter won't be empty
    adrc_inst.phonotactic_inventory.add('CVCCV')
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True) == "t͡ʃalda"

    # let more things go through filter cluster_filter:
    adrc_inst.cluster_inventory.add("d")
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True) == "dalda, t͡ʃalda"

    # sort result by nse (likelihood of reconstruction)
    adrc_inst.cluster_inventory.add("d")
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True,
        sort_by_nse=True) == "t͡ʃalda, dalda"

    # test show_workflow - run adapt first, then check workflow
    assert adrc_inst.adapt(
        ipastr="dade",
        howmany=1000,
        max_repaired_phonotactics=2,
        max_paths2repaired_phonotactics=2,
        repair_vowelharmony=True,
        phonotactics_filter=True,
        cluster_filter=True,
        sort_by_nse=True,
        show_workflow=True) == "t͡ʃalda, dalda"

    assert adrc_inst.workflow == OrderedDict(
        [
            ('tokenised',
             "['d', 'a', 'd', 'e']"),
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
        mode="reconstruct")

    assert adrc_inst.get_nse("ɟɒloɡ", "jɑlkɑ") == (
        6.67, 40, "[10, 9, 8, 7, 6, 0]",
        "['#-<*-', '#ɟ<*j', 'ɒ<*ɑ', 'l<*lk', 'o<*ɑ', 'ɡ#<*-']")

    # tear down
    del adrc_inst
