"""integration test for loanpy.qfysc.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from inspect import ismethod
from os import remove
from pathlib import Path

from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from pytest import raises

from loanpy.helpers import clusterise
from loanpy.qfysc import (Etym, InventoryMissingError,
read_scdictbase, read_dst, read_forms,
cldf2pd)

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"
PATH2FORMS2 = Path(__file__).parent / "input_files" / "forms.csv"

def test_get_scdictbase():
    """test if heuristic sound correspondence dictionary
    is calculated correctly"""
    # test with phoneme_inventory manually plugged in
    etym = Etym(phoneme_inventory=["e", "b", "c"])
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
    etym = Etym(forms_csv=PATH2FORMS2, source_language=1, target_language=2)
    scdictbase = etym.get_scdictbase(write_to=False)
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["z", "x", "y"]  # IPA z is most similar to IPA p
    assert scdictbase["h"] == ["x", "z", "y"]
    assert scdictbase["e"] == ["y", "z", "x"]
    assert scdictbase["C"] == ["x", "z"]
    assert scdictbase["V"] == ["y"]
    assert scdictbase["F"] == ["y"]
    assert scdictbase["B"] == []
    del etym, scdictbase

    # test if written correctly and if param most_common works

    # set up
    etym = Etym(phoneme_inventory=["e", "b", "c"])
    path2scdict_integr_test = Path(__file__).parent / "integr_test_scdict.txt"
    etym.get_scdictbase(write_to=path2scdict_integr_test, most_common=2)
    with open(path2scdict_integr_test, "r", encoding="utf-8") as f:
        scdictbase = literal_eval(f.read())

    # assert
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["b", "c"]  # b is obv most similar to p
    assert scdictbase["h"] == ["c", "b"]
    assert scdictbase["e"] == ["e", "b"]
    assert scdictbase["C"] == ["b", "c"]
    assert scdictbase["V"] == ["e"]
    assert scdictbase["F"] == ["e"]
    assert scdictbase["B"] == []

    # tear down
    remove(path2scdict_integr_test)
    del etym, scdictbase, path2scdict_integr_test


def test_rankclosest():
    """test if closest phonemes from inventory are ranked correctly"""
    # assert error is being raised correctly
    etym = Etym()
    with raises(InventoryMissingError) as inventorymissingerror_mock:
        etym.rank_closest(ph="d", howmany=3, inv=None)
    assert str(inventorymissingerror_mock.value
               ) == "define phoneme inventory or forms.csv"
    del etym

    # assert phonemes are ranked correctly
    etym = Etym(phoneme_inventory=["a", "b", "c"])
    assert etym.rank_closest(ph="d") == "b, c, a"
    assert etym.rank_closest(ph="d", howmany=2) == "b, c"
    assert etym.rank_closest(ph="d", inv=["r", "t", "l"], howmany=1) == "t"
    del etym


def test_rankclosest_phonotactics():
    """test if most similar phonotactic profiles from inventory
    are ranked up correctly"""
    # assert error is raised correctly if phoneme_inventory is missing
    etym = Etym()
    with raises(InventoryMissingError) as inventorymissingerror_mock:
        # assert error is raised
        etym.rank_closest_phonotactics(struc="CV", howmany=float("inf"))
        assert str(inventorymissingerror_mock.value
                   ) == "define phonotactic phoneme_inventory or forms.csv"
    del etym

    # assert structures are ranked correctly
    etym = Etym(PATH2FORMS2, source_language=1, target_language=2)
    # phonotactic_inventory is only lg2 aka "xyz"
    assert etym.rank_closest_phonotactics(struc="CVCV") == "CVC"
    assert etym.rank_closest_phonotactics(
        struc="CVCV", howmany=3, inv=[
            "CVC", "CVCVV", "CCCC", "VVVVVV"]) == "CVCVV, CVC, CCCC"
    del etym

def test_read_forms():
    """test if CLDF's forms.csv is read in correctly"""
    # test first break
    assert read_forms(None) is None

    # set up
    dfexp = DataFrame({"Language_ID": [1, 2],
                       "Segments": ["abc", "xyz"],  # pulled together segments
                       "Cognacy": [1, 1]})

    assert read_forms(None) is None
    assert_frame_equal(read_forms(PATH2FORMS2), dfexp)

    # tear down
    del dfexp


def test_cldf2pd():
    """test if the CLDF format is correctly tranformed to a pandas dataframe"""

    # set up
    dfin = read_csv(PATH2FORMS2)
    dfexp = DataFrame({"Target_Form": ["x y z"],
                       "Source_Form": ["a b c"],
                       "Cognacy": [1]})

    # assert
    assert cldf2pd(None, source_language="whatever",
                   target_language="wtvr2") is None
    assert_frame_equal(cldf2pd(dfin, source_language=1,
                               target_language=2), dfexp)

    # tear down
    del dfexp, dfin


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


def test_read_inventory():
    """test if phoneme/cluster inventory is read in correctly"""
    # assert first two exceptions: inv is not None and inv and forms are None
    etym = Etym()
    etym.forms_target_language = "some_inv"
    assert etym.read_inventory("some_formscsv") == "some_formscsv"
    assert etym.read_inventory(None) == set("someinv")  # tokeniser drops "_"
    etym.forms_target_language = None
    assert etym.read_inventory(None, None) is None

    # assert calculations
    etym.forms_target_language = ["a", "aab", "bc"]
    assert etym.read_inventory(None) == set(['a', 'b', 'c'])
    etym.forms_target_language = ["a", "ab", "baac"]
    assert etym.read_inventory(None, clusterise
                               ) == set(['aa', 'bb', 'c'])


def test_get_inventories():
    """test if phoneme/cluster/phonotactic inventories are read in well"""
    # set up instancce
    etym = Etym()
    # run func, assert output
    etym.get_inventories() == (None, None, None)

    # rerun with non-default args
    # create instancce
    etym = Etym()
    # run func, assert output
    etym.get_inventories("param1", "param2", "param3", 4) == (
        "param1", "param2", "param3")

    # rerun with real etym instnace
    etym = Etym(forms_csv=PATH2FORMS2, source_language=1, target_language=2)
    # run func
    etym.get_inventories()
    # assert assigned attributes
    assert etym.phoneme_inventory == {'x', 'y', 'z'}
    assert etym.cluster_inventory == {'x', 'y', 'z'}
    assert etym.phonotactic_inventory == {'CVC'}

    # tear down
    del etym


def test_read_phonotactic_inv():
    """test if phonotactic inventory is read in correctly"""
    # set up rest
    etym = Etym()
    # from forms.csv in CLDF
    etym.forms_target_language = ["ab", "ab", "aa", "bb", "bb", "bb"]
    # assert with different parameter combinations
    assert etym.read_phonotactic_inv(phonotactic_inventory=["a", "b", "c"],
                                     ) == ["a", "b", "c"]
    etym.forms_target_language = None
    assert etym.read_phonotactic_inv(phonotactic_inventory=None,
                                     ) is None
    etym.forms_target_language = ["ab", "ab", "aa", "bb", "bb", "bb"]
    # now just read the most frquent 2 structures. VV is the 3rd frquent. so
    # not in the output.
    assert etym.read_phonotactic_inv(phonotactic_inventory=None,
                                     howmany=2) == {"CC", "VC"}

    # tear down
    del etym

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
    assert len(qfy.__dict__) == 10

    # 6 attributes inherited from Etym
    assert qfy.dfety is None
    assert qfy.phoneme_inventory is None
    assert qfy.cluster_inventory is None
    assert qfy.phonotactic_inventory is None
    ismethod(qfy.distance_measure)
    assert qfy.forms_target_language is None

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
    assert_frame_equal(qfy.align(left="kala", right="hal"), DataFrame(
        {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}))
    # overwrite
    qfy = Etym(mode="reconstruct")
    # assert
    assert_frame_equal(qfy.align(left="ɟɒloɡ", right="jɑlkɑ"),
                       DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                  "vals": ['-', 'j', 'ɑ', 'lk', 'ɑ', '-']}))
    # tear down
    del qfy


def test_align_lingpy():
    """check if alignments work correctly"""
    # set up
    qfy = Etym()
    # assert
    assert_frame_equal(qfy.align(left="kala", right="hal"), DataFrame(
        {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}))

    assert_frame_equal(qfy.align(left="aɣat͡ʃi", right="aɣat͡ʃːɯ"),
                       DataFrame({"keys": ["a", "ɣ", "a", "t͡ʃː", "ɯ"],
                                  "vals": ["a", "ɣ", "a", "t͡ʃ", "i"]}))

    assert_frame_equal(qfy.align(left="aldaɣ", right="aldaɣ"),
                       DataFrame({"keys": ["a", "l", "d", "a", "ɣ"],
                                  "vals": ["a", "l", "d", "a", "ɣ"]}))

    assert_frame_equal(qfy.align(left="ajan", right="ajan"), DataFrame(
        {"keys": ["a", "j", "a", "n"], "vals": ["a", "j", "a", "n"]}))

    # tear down
    del qfy


def test_align_clusterwise():
    """check if our own alignment function works correctly"""
    # set up
    qfy = Etym(mode="reconstruct")
    # assert
    assert_frame_equal(qfy.align(left="ɟɒloɡ", right="jɑlkɑ"),
                       DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                  "vals": ['-', 'j', 'ɑ', 'lk', 'ɑ', '-']}))

    assert_frame_equal(qfy.align(left="kiki", right="hihi"),
                       DataFrame({"keys": ['#-', '#k', 'i', 'k', 'i#', '-#'],
                                  "vals": ['-', 'h', 'i', 'h', 'i', '-']}))
    assert_frame_equal(qfy.align(left="kiki", right="ihi"),
                       DataFrame({"keys": ['#k', 'i', 'k', 'i#', '-#'],
                                  "vals": ['-', 'i', 'h', 'i', '-']}))

    assert_frame_equal(qfy.align(left="iki", right="hihi"),
                       DataFrame({"keys": ['#-', '#i', 'k', 'i#', '-#'],
                                  "vals": ['h', 'i', 'h', 'i', '-']}))

    assert_frame_equal(qfy.align(left="uoaeia", right="brrrzierrrrr"),
                       DataFrame({"keys": ['#-', 'uoaeia#', '-#'],
                                  "vals": ['brrrz', 'ie', 'rrrrr']}))

    assert_frame_equal(qfy.align(left="uoaeia", right="brrrzi"),
                       DataFrame({"keys": ['#-', 'uoaeia#', '-#'],
                                  "vals": ['brrrz', 'i', '-']}))

    assert_frame_equal(qfy.align(left="uoaeia", right="brrrz"),
                       DataFrame({"keys": ['#-', 'uoaeia#'],
                                  "vals": ['brrrz', '-']}))

    assert_frame_equal(qfy.align(left="budapestttt", right="uadast"),
                       DataFrame(
                           {"keys": ['#b', 'u', 'd', 'a', 'p', 'estttt#'],
                            "vals": ['-', 'ua', 'd', 'a', 'st', '-']}))

    # the only example in ronatasbertawot
    # where one starts with C, the other with V
    assert_frame_equal(qfy.align(left="imad", right="vimad"),
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
            'ɣ': ['ɣ'],
            'ɯ': ['i']},

        {'a<a': 6,
         'd<d': 1,
         'i<ɯ': 1,
         'j<j': 1,
         'l<l': 1,
         'n<n': 1,
         't͡ʃ<t͡ʃː': 1,
         'ɣ<ɣ': 2},

        {'a<a': [1, 2, 3],
         'd<d': [2],
         'i<ɯ': [1],
         'j<j': [3],
         'l<l': [2],
         'n<n': [3],
         't͡ʃ<t͡ʃː': [1],
         'ɣ<ɣ': [1, 2]},

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
            '-#': ['at͡ʃi', 'ɣ'],
            'aː': ['a'],
            'jn': ['j'],
            'oz#': ['-'],
            'r': ['n'],
            't͡ʃ#': ['ɣ'],
            'uː#': ['a'],
            'ɟ': ['ld']},

        {'#-<*-': 3,
         '#aː<*a': 2,
         '#ɒ<*a': 1,
         '-#<*at͡ʃi': 1,
         '-#<*ɣ': 1,
         'aː<*a': 1,
         'jn<*j': 1,
         'oz#<*-': 1,
         'r<*n': 1,
         't͡ʃ#<*ɣ': 1,
         'uː#<*a': 1,
         'ɟ<*ld': 1},

        {'#-<*-': [1, 2, 3],
         '#aː<*a': [1, 2],
         '#ɒ<*a': [3],
         '-#<*at͡ʃi': [1],
         '-#<*ɣ': [2],
         'aː<*a': [3],
         'jn<*j': [3],
         'oz#<*-': [3],
         'r<*n': [3],
         't͡ʃ#<*ɣ': [1],
         'uː#<*a': [2],
         'ɟ<*ld': [2]},
        {}, {}, {}
    ]

    # assert frst test runs but with write_to and struc=True
    exp = [{'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'], 'n': ['n'],
            't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']},
           {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1, 'l<l': 1,
            'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2},
           {'a<a': [1, 2, 3], 'd<d': [2], 'i<ɯ': [1],
            'j<j': [3], 'l<l': [2], 'n<n': [3],
            't͡ʃ<t͡ʃː': [1], 'ɣ<ɣ': [1, 2]},
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
