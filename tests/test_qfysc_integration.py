"""integration test for loanpy.qfysc.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from inspect import ismethod
from os import remove
from pathlib import Path

from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from pytest import raises

from loanpy.qfysc import (Etym,
read_scdictbase, read_dst, cldf2pd)

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"
PATH2FORMS2 = Path(__file__).parent / "input_files" / "forms.csv"

def test_get_scdictbase():
    """test if heuristic sound correspondence dictionary
    is calculated correctly"""
    # test with phoneme_inventory manually plugged in
    etym = Etym()
    etym.inventories["Segments"] = ["e", "b", "c"]
    scdictbase = etym.get_scdictbase(write_to=False)
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["b", "c", "e"]  # b is obv most similar to p
    assert scdictbase["h"] == ["c", "b", "e"]
    assert scdictbase["e"] == ["e", "b", "c"]
    assert scdictbase["C"] == ["b", "c"]
    assert scdictbase["V"] == ["e"]
    assert scdictbase["F"] == ["e"]
    assert scdictbase["B"] == []
    del etym, scdictbase

    # test with invetory extracted from forms.csv
    path2scdict_integr_test = Path(__file__).parent / "integr_test_scdict.txt"
    etym = Etym(forms_csv=PATH2FORMS2, source_language=1, target_language=2)
    scdictbase = etym.get_scdictbase(write_to=path2scdict_integr_test)
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["z", "x", "y"]  # IPA z is most similar to IPA p
    assert scdictbase["h"] == ["x", "z", "y"]
    assert scdictbase["e"] == ["y", "z", "x"]
    assert scdictbase["C"] == ["x", "z"]
    assert scdictbase["V"] == ["y"]
    assert scdictbase["F"] == ["y"]
    assert scdictbase["B"] == []

    # test if written correctly
    with open(path2scdict_integr_test, "r", encoding="utf-8") as f:
        # quite large file, so best to just test like this
        scdictbase = literal_eval(f.read())

    remove(path2scdict_integr_test)
    del etym, scdictbase, path2scdict_integr_test

def test_rankclosest():
    """test if closest phonemes from inventory are ranked correctly"""

    # assert phonemes are ranked correctly
    etym = Etym()
    etym.inventories["Segments"] =  set(["a", "b", "c"])
    assert etym.rank_closest(ph="d") == "b, c, a"
    assert etym.rank_closest(ph="d", howmany=2) == "b, c"
    etym.inventories["Segments"] =  set(["r", "t", "l"])
    assert etym.rank_closest(ph="d", howmany=1) == "t"
    del etym


def test_rankclosest_phonotactics():
    """test if most similar phonotactic profiles from inventory
    are ranked up correctly"""

    # assert structures are ranked correctly
    etym = Etym(PATH2FORMS2, source_language=1, target_language=2)
    # phonotactic_inventory is only lg2 aka "xyz"
    assert etym.rank_closest_phonotactics(struc="CVCV") == "CVC"
    assert etym.rank_closest_phonotactics(
        struc="CVCV", howmany=3) == "CVC"
    del etym

def test_cldf2pd():
    """test if the CLDF format is correctly tranformed to a pandas dataframe"""

    # set up
    dfexp = DataFrame({"Segments_tgt": ["x y z"],
                       "Segments_src": ["a b c"],
                       "CV_Segments_tgt": ["x y z"],
                       "CV_Segments_src": ["a b.c"],
                       "ProsodicStructure_tgt": ["CVC"],
                       "ProsodicStructure_src": ["VCC"],
                       "Cognacy": [1],
                       })
    dfexp2 = DataFrame({"Segments_tgt": [], "CV_Segments_tgt": [], "ProsodicStructure_tgt": []})

    # assert
    assert cldf2pd(None, source_language="whatever",
                   target_language="wtvr2") == (None, None)
    out = cldf2pd(PATH2FORMS2, source_language=1,
                               target_language=2)
    assert_frame_equal(out[0], dfexp)
    print(out[1])
    assert_frame_equal(out[1], dfexp2)

    # tear down
    del dfexp


def test_read_dst():
    """test if input-string is correctly mapped to
    method of panphon.distance.Distance"""
    out = read_dst("weighted_feature_edit_distance")
    assert ismethod(out)


def test_form2list():
    pass  # unit == integration test


def test_init():
    """test if class Etym is initiated correctly"""
    # set up: initiate without args
    mocketym = Etym()

    # assert that the right number of class attributes were instanciated
    assert len(mocketym.__dict__) == 6

    # assert the other 5 attributes were read correctly
    assert mocketym.dfety is None
    assert mocketym.phoneme_inventory is None
    assert mocketym.cluster_inventory is None
    assert mocketym.phonotactic_inventory is None
    ismethod(mocketym.distance_measure)

    # tear down
    del mocketym

    # set up2: run with advanced parameters
    # input vars for init params
    mocketym = Etym(forms_csv=PATH2FORMS, source_language=1, target_language=2)

    # assert right number of attributes was initiated (7)
    assert len(mocketym.__dict__) == 6

    # (3) assert dfety was read correctly
    assert_frame_equal(mocketym.dfety, DataFrame(
        {"Target_Form": ["xyz"], "Source_Form": ["abc"], "Cognacy": [1]}))

    # assert the other 4 attributes were read correctly
    assert mocketym.phoneme_inventory == {'x', 'y', 'z'}
    assert mocketym.cluster_inventory == {'x', 'y', 'z'}
    assert mocketym.phonotactic_inventory == {"CVC"}
    ismethod(mocketym.distance_measure)

    # tear down
    del mocketym

def test_get_inventories():
    """test if phoneme/cluster/phonotactic inventories are read in well"""
    pass
    #TODO: re-write this.

def test_read_mode():
    pass  # no patches in unittest (equal integration test)


def test_read_connector():
    pass  # no patches in unittest (equal integration test)


def test_read_scdictbase():
    """test if scdictbase is generated correctly from ipa_all.csv"""

    # setup
    base = {"a": ["e", "o"], "b": ["p", "v"]}
    path = Path(__file__).parent / "test_read_scdictbase.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(base))

    # assert
    assert read_scdictbase(base) == base
    assert read_scdictbase(path) == base

    # tear down
    remove(path)
    del base, path


def test_init():
    """test if loanpy.qfy.Etym is initiated correctly"""
    qfy = Etym()

    # assert number of attributes (super() + rest)
    assert len(qfy.__dict__) == 8

    # 6 attributes inherited from Etym
    assert qfy.dfety is None
    assert qfy.inventories == {}
    ismethod(qfy.distance_measure)

    # 4 attributes initiated in Etym
    assert qfy.mode == "adapt"
    assert qfy.connector == "<"
    assert qfy.scdictbase == {}
    assert qfy.vfb is None

    del qfy


def test_align():
    """test if 2 strings are aligned correctly"""
    # set up
    qfy = Etym()
    # assert
    assert_frame_equal(qfy.align(left="k a l a", right="h a l"), DataFrame(
        {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}))
    # overwrite
    qfy = Etym(mode="reconstruct")
    # assert
    assert_frame_equal(qfy.align(left="ɟ ɒ l o ɡ", right="j ɑ l.k ɑ"),
                       DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                  "vals": ['-', 'j', 'ɑ', 'l.k', 'ɑ', '-']}))
    # tear down
    del qfy


def test_align_lingpy():
    """check if alignments work correctly"""
    # set up
    qfy = Etym()
    # assert
    assert_frame_equal(qfy.align(left="kala", right="hal"), DataFrame(
        {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}))

    assert_frame_equal(qfy.align(left="aγat͡ʃi", right="aγat͡ʃːɯ"),
                       DataFrame({"keys": ["a", "γ", "a", "t͡ʃː", "ɯ"],
                                  "vals": ["a", "γ", "a", "t͡ʃ", "i"]}))

    assert_frame_equal(qfy.align(left="aldaγ", right="aldaγ"),
                       DataFrame({"keys": ["a", "l", "d", "a", "γ"],
                                  "vals": ["a", "l", "d", "a", "γ"]}))

    assert_frame_equal(qfy.align(left="ajan", right="ajan"), DataFrame(
        {"keys": ["a", "j", "a", "n"], "vals": ["a", "j", "a", "n"]}))

    # tear down
    del qfy


def test_align_clusterwise():
    """check if our own alignment function works correctly"""
    # set up
    qfy = Etym(mode="reconstruct")
    # assert
    assert_frame_equal(qfy.align(left="ɟ ɒ l o ɡ", right="j ɑ l.k ɑ"),
                       DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                  "vals": ['-', 'j', 'ɑ', 'l.k', 'ɑ', '-']}))

    assert_frame_equal(qfy.align(left="k i k i", right="h i h i"),
                       DataFrame({"keys": ['#-', '#k', 'i', 'k', 'i#', '-#'],
                                  "vals": ['-', 'h', 'i', 'h', 'i', '-']}))
    assert_frame_equal(qfy.align(left="k i k i", right="i h i"),
                       DataFrame({"keys": ['#k', 'i', 'k', 'i#', '-#'],
                                  "vals": ['-', 'i', 'h', 'i', '-']}))

    assert_frame_equal(qfy.align(left="i k i", right="h i h i"),
                       DataFrame({"keys": ['#-', '#i', 'k', 'i#', '-#'],
                                  "vals": ['h', 'i', 'h', 'i', '-']}))

    assert_frame_equal(qfy.align(left="u.o.a.e.i.a", right="b.r.r.r.z i.e r.r.r.r.r"),
                       DataFrame({"keys": ['#-', 'u.o.a.e.i.a#', '-#'],
                                  "vals": ['b.r.r.r.z', 'i.e', 'r.r.r.r.r']}))

    assert_frame_equal(qfy.align(left="u.o.a.e.i.a", right="b.r.r.r.z i"),
                       DataFrame({"keys": ['#-', 'u.o.a.e.i.a#', '-#'],
                                  "vals": ['b.r.r.r.z', 'i', '-']}))

    assert_frame_equal(qfy.align(left="u.o.a.e.i.a", right="b.r.r.r.z"),
                       DataFrame({"keys": ['#-', 'u.o.a.e.i.a#'],
                                  "vals": ['b.r.r.r.z', '-']}))

    assert_frame_equal(qfy.align(left="b u d a p e s.t.t.t.t", right="u.a d a s.t"),
                       DataFrame(
                           {"keys": ['#b', 'u', 'd', 'a', 'p', 'e s.t.t.t.t#'],
                            "vals": ['-', 'u.a', 'd', 'a', 's.t', '-']}))

    # the only example in ronatasbertawot
    # where one starts with C, the other with V
    assert_frame_equal(qfy.align(left="i m a d", right="v i m a d"),
                       DataFrame({"keys": ['#-', '#i', 'm', 'a', 'd#', '-#'],
                                  "vals": ['v', 'i', 'm', 'a', 'd', '-']}))

    # tear down
    del qfy


def test_get_sound_corresp():
    """Are sound correspondences correctly extracted from etymological data?"""
    # assert adapt mode works well
    qfy = Etym(forms_csv=PATH2FORMS, source_language="WOT",
              target_language="EAH")
#    print(qfy.left, qfy.dfety) # qfy.dfety[qfy.left]
    out = qfy.get_sound_corresp(write_to=None)
    for s_out, s_exp in zip(out.pop(3),
                            {'VCCVC': ['VCCVC', 'VCVC', 'VCVCV'],
                             'VCVC': ['VCVC', 'VCVCV', 'VCCVC'],
                             'VCVCV': ['VCVCV', 'VCVC', 'VCCVC']}):
        assert set(s_out) == set(s_exp)
    assert out == [
        {'a': ['a'],
         'd': ['d'],
            'j': ['j'],
            'l': ['l'],
            'n': ['n'],
            't͡ʃː': ['t͡ʃ'],
            'γ': ['γ'],
            'ɯ': ['i']},

        {'a<a': 6,
         'd<d': 1,
         'i<ɯ': 1,
         'j<j': 1,
         'l<l': 1,
         'n<n': 1,
         't͡ʃ<t͡ʃː': 1,
         'γ<γ': 2},

        {'a<a': [1, 2, 3],
         'd<d': [2],
         'i<ɯ': [1],
         'j<j': [3],
         'l<l': [2],
         'n<n': [3],
         't͡ʃ<t͡ʃː': [1],
         'γ<γ': [1, 2]},

        {'VCCVC<VCCVC': 1,
         'VCVC<VCVC': 1,
         'VCVCV<VCVCV': 1},
        {'VCCVC<VCCVC': [2],
         'VCVC<VCVC': [3],
         'VCVCV<VCVCV': [1]}
    ]

# make sure reconstruction works
    qfy = Etym(forms_csv=PATH2FORMS, source_language="EAH",
              target_language="H", mode="reconstruct")
#    print(qfy.left, qfy.dfety) # qfy.dfety[qfy.left]
    assert qfy.get_sound_corresp(write_to=None) == [
        {'#-': ['-'],
         '#aː': ['a'],
         '#ɒ': ['a'],
            '-#': ['a t͡ʃ i', 'γ'],
            'aː': ['a'],
            'j.n': ['j'],
            'o z#': ['-'],
            'r': ['n'],
            't͡ʃ#': ['γ'],
            'uː#': ['a'],
            'ɟ': ['l.d']},

        {'#-<*-': 3,
         '#aː<*a': 2,
         '#ɒ<*a': 1,
         '-#<*a t͡ʃ i': 1,
         '-#<*γ': 1,
         'aː<*a': 1,
         'j.n<*j': 1,
         'o z#<*-': 1,
         'r<*n': 1,
         't͡ʃ#<*γ': 1,
         'uː#<*a': 1,
         'ɟ<*l.d': 1},

        {'#-<*-': [1, 2, 3],
         '#aː<*a': [1, 2],
         '#ɒ<*a': [3],
         '-#<*a t͡ʃ i': [1],
         '-#<*γ': [2],
         'aː<*a': [3],
         'j.n<*j': [3],
         'o z#<*-': [3],
         'r<*n': [3],
         't͡ʃ#<*γ': [1],
         'uː#<*a': [2],
         'ɟ<*l.d': [2]},
        {}, {}, {}
    ]

    # assert frst test runs but with write_to and struc=True
    exp = [{'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'], 'n': ['n'],
            't͡ʃː': ['t͡ʃ'], 'γ': ['γ'], 'ɯ': ['i']},
           {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1, 'l<l': 1,
            'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'γ<γ': 2},
           {'a<a': [1, 2, 3], 'd<d': [2], 'i<ɯ': [1],
            'j<j': [3], 'l<l': [2], 'n<n': [3],
            't͡ʃ<t͡ʃː': [1], 'γ<γ': [1, 2]},
           {'VCCVC': ['VCCVC', 'VCVC', 'VCVCV'],
            'VCVC': ['VCVC', 'VCCVC', 'VCVCV'],
            'VCVCV': ['VCVCV', 'VCVC', 'VCCVC']},
           {'VCCVC<VCCVC': 1, 'VCVC<VCVC': 1, 'VCVCV<VCVCV': 1},
           {'VCCVC<VCCVC': [2], 'VCVC<VCVC': [3], 'VCVCV<VCVCV': [1]}]

    qfy = Etym(
        forms_csv=PATH2FORMS, source_language="WOT", target_language="EAH")
    path2test_sc = Path(__file__).parent / "test_sc.txt"

    # dict nr. 3 (struc) has randomness (bc rankcclosest) so pop and use set!
    out = qfy.get_sound_corresp(write_to=path2test_sc)
    # take out dict 3 from list and assign it to new vars
    outpop, exppop = out.pop(3), exp.pop(3)
    # assert that set of dict values are equal
    for s_out, s_exp in zip(outpop, exppop):
        assert set(outpop[s_out]) == set(exppop[s_exp])
    # assert that the other dictionaries are equal
    assert out == exp

    # repeat steps from block above but read same results from file
    with open(path2test_sc, "r", encoding="utf-8") as f:
        out = literal_eval(f.read())
    outpop = out.pop(3)  # we popped exp already above, don't do it again
    for s_out, s_exp in zip(outpop, exppop):
        assert set(outpop[s_out]) == set(exppop[s_exp])
    assert out == exp

    # tear down
    remove(path2test_sc)
    del out, path2test_sc, qfy,


def test_get_phonotactics_corresp():
    """Are phonotactic correspondences correctly extracted from etym. data?"""
    # assert frst test runs but with write_to and struc=True
    exp = [{'VCCVC': ['VCCVC', 'VCVC', 'VCVCV'],
            'VCVC': ['VCVC', 'VCVCV', 'VCCVC'],
            'VCVCV': ['VCVCV', 'VCVC', 'VCCVC']},
           {'VCCVC<VCCVC': 1, 'VCVC<VCVC': 1, 'VCVCV<VCVCV': 1},
           {'VCCVC<VCCVC': [2], 'VCVC<VCVC': [3], 'VCVCV<VCVCV': [1]}]

    qfy = Etym(forms_csv=PATH2FORMS, source_language="WOT",
              target_language="EAH")
    path2test_sc = Path(__file__).parent / "test_sc.txt"

    # assert return value is as expected. lists in 1st dict are random so set.
    out = qfy.get_phonotactics_corresp(write_to=path2test_sc)
    for s_out, s_exp in zip(out[0], exp[0]):
        assert set(out[0][s_out]) == set(exp[0][s_exp])
    assert out[1:] == exp[1:]

    # assert output was written correctly to file
    with open(path2test_sc, "r", encoding="utf-8") as f:
        out = literal_eval(f.read())
    for s_out, s_exp in zip(out[0], exp[0]):
        assert set(out[0][s_out]) == set(exp[0][s_exp])
    assert out[1:] == exp[1:]

    remove(path2test_sc)
    del out, path2test_sc, qfy

    #
