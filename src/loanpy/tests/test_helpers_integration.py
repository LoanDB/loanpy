"""integration tests for loanpy.helpers.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from inspect import ismethod
from os import remove
from pathlib import Path

from networkx import DiGraph
from numpy import array
from numpy.testing import assert_array_equal
from pandas import DataFrame, Index, read_csv
from pandas.testing import assert_frame_equal
from pytest import raises

from loanpy.helpers import (
    Etym,
    InventoryMissingError,
    cldf2pd,
    editops,
    gensim_multiword,
    get_mtx,
    combine_ipalists,
    clusterise,
    mtx2graph,
    read_cvfb,
    read_dst,
    read_forms,
    tuples2editops)

PATH2FORMS = Path(__file__).parent / "input_files" / "forms.csv"


def test_plug_in_model():
    pass  # unittest = integration test


def test_read_cvfb():
    """reading in cvfb.txt which is always in the same place"""
    cvfb = read_cvfb()
    assert isinstance(cvfb, tuple)
    assert len(cvfb) == 2
    # extremely long dictionary, so just type is checked
    assert isinstance(cvfb[0], dict)
    assert isinstance(cvfb[1], dict)
    # extremely long dictionary, here's how long
    assert len(cvfb[0]) == 6358
    assert len(cvfb[1]) == 1240
    assert all(cvfb[0][val] in ["C", "V"] for val in cvfb[0])
    assert all(cvfb[1][val] in ["F", "B", "V"] for val in cvfb[1])

    # verify that the dict is actually based on ipa_all.csv
    # and that "C" corresponds to "+" and "V" "-" in col "ipa".
    dfipa = read_csv(Path(__file__).parent.parent / "ipa_all.csv")
    for i, c in zip(dfipa.ipa, dfipa.cons):
        if c == "+":
            assert cvfb[0][i] == "C"
        elif c == "-":
            assert cvfb[0][i] == "V"
        else:
            assert cvfb[0].get(i, "notindict") == "notindict"

    del dfipa, cvfb


def test_read_forms():
    """test if CLDF's forms.csv is read in correctly"""
    # test first break
    assert read_forms(None) is None

    # set up
    dfexp = DataFrame({"Language_ID": [1, 2],
                       "Segments": ["abc", "xyz"],  # pulled together segments
                       "Cognacy": [1, 1]})

    assert read_forms(None) is None
    assert_frame_equal(read_forms(PATH2FORMS), dfexp)

    # tear down
    del dfexp


def test_cldf2pd():
    """test if the CLDF format is correctly tranformed to a pandas dataframe"""

    # set up
    dfin = read_csv(PATH2FORMS)
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


def test_flatten():
    pass  # unittest == integration test


def test_combine_ipalists():
    """test if combinatorics is correctly applied to phonemes (#odometer)"""
    inlist = [[["k", "g"], ["i", "e"]], [["b", "p"], ["u", "o"]]]
    out = ["ki", "ke", "gi", "ge", "bu", "bo", "pu", "po"]
    assert combine_ipalists(inlist) == out
    del inlist, out


def test_form2list():
    pass  # unit == integration test


def test_init():
    """test if class Etym is initiated correctly"""
    # set up: initiate without args
    mocketym = Etym()

    # assert that the right number of class attributes were instanciated
    assert len(mocketym.__dict__) == 8

    # assert phon2cv was read correctly
    assert isinstance(mocketym.phon2cv, dict)
    assert len(mocketym.phon2cv) == 6358
    assert mocketym.phon2cv["k"] == "C"
    assert mocketym.phon2cv["p"] == "C"
    assert mocketym.phon2cv["l"] == "C"
    assert mocketym.phon2cv["w"] == "C"
    assert mocketym.phon2cv["C"] == "C"
    assert mocketym.phon2cv["a"] == "V"
    assert mocketym.phon2cv["e"] == "V"
    assert mocketym.phon2cv["i"] == "V"
    assert mocketym.phon2cv["o"] == "V"
    assert mocketym.phon2cv["u"] == "V"

    # assert vow2fb was read correctly
    assert isinstance(mocketym.vow2fb, dict)
    assert len(mocketym.vow2fb) == 1240
    assert mocketym.vow2fb["e"] == "F"
    assert mocketym.vow2fb["i"] == "F"
    assert mocketym.vow2fb["o"] == "B"
    assert mocketym.vow2fb["u"] == "B"

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
    assert len(mocketym.__dict__) == 8

    # (1) assert phon2cv was read correctly
    assert isinstance(mocketym.phon2cv, dict)
    assert len(mocketym.phon2cv) == 6358
    assert mocketym.phon2cv["k"] == "C"
    assert mocketym.phon2cv["p"] == "C"
    assert mocketym.phon2cv["l"] == "C"
    assert mocketym.phon2cv["w"] == "C"
    assert mocketym.phon2cv["C"] == "C"
    assert mocketym.phon2cv["a"] == "V"
    assert mocketym.phon2cv["e"] == "V"
    assert mocketym.phon2cv["i"] == "V"
    assert mocketym.phon2cv["o"] == "V"
    assert mocketym.phon2cv["u"] == "V"

    # (2) assert vow2fb was read correctly
    assert isinstance(mocketym.vow2fb, dict)
    assert len(mocketym.vow2fb) == 1240
    assert mocketym.vow2fb["e"] == "F"
    assert mocketym.vow2fb["i"] == "F"
    assert mocketym.vow2fb["o"] == "B"
    assert mocketym.vow2fb["u"] == "B"

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
    etym = Etym(PATH2FORMS, source_language=1, target_language=2)
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


def test_word2phonotactics():
    """test if the phonotactic profile of a word is correctly concluded"""

    etym = Etym()
    assert etym.word2phonotactics("t͡ʃɒlːoːkøz") == "CVCVCVC"
    assert etym.word2phonotactics("hortobaːɟ") == "CVCCVCVC"
    # hashtag is ignored
    assert etym.word2phonotactics("lɒk#sɒkaːlːɒʃ") == "CVCCVCVCVC"
    assert etym.word2phonotactics("boɟ!ɒ") == "CVCV"  # exclam. mark is ignored
    assert etym.word2phonotactics("ɡɛlːeːr") == "CVCVC"

    # tear down
    del etym


def test_word2phonotactics_keepcv():
    """Not used in loanpy. Test if C and V is kept
    during phonotactic profiling"""
    etym = Etym()
    assert etym.word2phonotactics(
        ['C', 'V', 'C', 'V', 'k', 'ø', 'z']) == "CVCVCVC"
    assert etym.word2phonotactics(
        ['h', 'o', 'r', 'C', 'V', 'C', 'V', 'ɟ']) == "CVCCVCVC"
    assert etym.word2phonotactics(
        ['l', 'V', 'k', 's', 'ɒ', 'k', 'V', 'C', 'V', 'ʃ']) == "CVCCVCVCVC"
    assert etym.word2phonotactics(['C', 'o', '!', 'ɟ', 'V']) == "CVCV"
    assert etym.word2phonotactics(["C", "V", "lː", "eː", "r"]) == "CVCVC"
    del etym


def test_harmony():
    """Test if it is detected correctly whether a word does or does not
    have front-back vowel harmony"""
    etym = Etym()
    assert etym.has_harmony(
        ['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']) is False
    assert etym.has_harmony("bot͡sibot͡si") is False
    assert etym.has_harmony("tɒrkɒ") is True
    assert etym.has_harmony("ʃɛfylɛʃɛ") is True
    del etym


def test_repair_harmony():
    """test if words without front-back vowel harmony are repaired correctly"""
    etym = Etym()
    assert etym.repair_harmony(ipalist='kɛsthɛj') == [
        ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']]
    assert etym.repair_harmony(ipalist='ɒlʃoːørʃ') == [
        ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']]
    assert etym.repair_harmony(ipalist=[
        'b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']) == [
        ['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p']]
    assert etym.repair_harmony(ipalist='bɒlɒtonkɛnɛʃɛ') == [
        ['b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'],
        ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']]
    del etym


def test_get_fb():
    """test if front and back vowels are fetched correctly"""
    etym = Etym()
    assert etym.get_fb(ipalist=['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']) == [
        'k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']
    assert etym.get_fb(ipalist=[
        'ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'], turnto="B") == [
        'ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']
    assert etym.get_fb(['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                        'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'], "B") == [
        'b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']
    assert etym.get_fb(['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                        'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'], "F") == [
        'b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']
    del etym


def test_get_scdictbase():
    """test if heuristic sound correspondence dictionary
    is calculated correctly"""
    # test with phoneme_inventory manually plugged in
    etym = Etym(phoneme_inventory=["a", "b", "c"])
    scdictbase = etym.get_scdictbase(write_to=False)
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["b", "c", "a"]  # b is obv most similar to p
    assert scdictbase["h"] == ["c", "b", "a"]
    assert scdictbase["e"] == ["a", "b", "c"]
    assert scdictbase["C"] == ["b", "c"]
    assert scdictbase["V"] == ["a"]
    assert scdictbase["F"] == ["a"]
    assert scdictbase["B"] == []
    del etym, scdictbase

    # test with invetory extracted from forms.csv
    etym = Etym(PATH2FORMS, source_language=1, target_language=2)
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
    etym = Etym(phoneme_inventory=["a", "b", "c"])
    path2scdict_integr_test = Path(__file__).parent / "integr_test_scdict.txt"
    etym.get_scdictbase(write_to=path2scdict_integr_test, most_common=2)
    with open(path2scdict_integr_test, "r", encoding="utf-8") as f:
        scdictbase = literal_eval(f.read())

    # assert
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["b", "c"]  # b is obv most similar to p
    assert scdictbase["h"] == ["c", "b"]
    assert scdictbase["e"] == ["a", "b"]
    assert scdictbase["C"] == ["b", "c"]
    assert scdictbase["V"] == ["a"]
    assert scdictbase["F"] == ["a"]
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
    etym = Etym(PATH2FORMS, source_language=1, target_language=2)
    # phonotactic_inventory is only lg2 aka "xyz"
    assert etym.rank_closest_phonotactics(struc="CVCV") == "CVC"
    assert etym.rank_closest_phonotactics(
        struc="CVCV", howmany=3, inv=[
            "CVC", "CVCVV", "CCCC", "VVVVVV"]) == "CVCVV, CVC, CCCC"
    del etym


def test_gensim_multiword():
    pass  # checked in detail and unit == integration test (!)


def test_list2regex():
    pass  # no patches in unittest


def test_edit_distance_with2ops():
    pass  # no patches in unittest


def test_get_mtx():
    """Dynamic programming. Test if matrix of edit operations is
    calculated correctly"""
    # assert error is raised if one of the input strings is empty
    with raises(IndexError) as indexerror_mock:
        get_mtx("a", "")
    assert str(indexerror_mock.value == "list index out of range")

    with raises(IndexError) as indexerror_mock:
        get_mtx("", "bla")
    assert str(indexerror_mock.value == "list index out of range")

    with raises(IndexError) as indexerror_mock:
        get_mtx("", "")
    assert str(indexerror_mock.value == "list index out of range")

    # illustration with small examples
    # anchor is 0 if first chars are same
    assert_array_equal(get_mtx("a", "a"), array([[0., 1.], [1., 0.]]))

    # anchor is 2 if first chars are different
    assert_array_equal(get_mtx("b", "a"), array([[0., 1.], [1., 2.]]))

    assert_array_equal(get_mtx("xy", "y"),
                       array([[0., 1., 2.],
                              [1., 2., 1.]]))  # 1 insertion

    assert_array_equal(get_mtx("y", "xy"),
                       array([[0., 1.],
                              [1., 2.],
                              [2., 1.]]))  # 1 deletion

    assert_array_equal(get_mtx("z", "xy"),
                       array([[0., 1.],
                              [1., 2.],
                              [2., 3.]]))  # 1 del + 2 ins = 3 esit ops

    assert_array_equal(get_mtx("Bécs", "Pécs"),
                       # delete B insert P: 2 editops
                       array([[0., 1., 2., 3., 4.],
                              [1., 2., 3., 4., 5.],
                              [2., 3., 2., 3., 4.],
                              [3., 4., 3., 2., 3.],
                              [4., 5., 4., 3., 2.]]))

    assert_array_equal(get_mtx("Komárom", "Révkomárom"),
                       array([[0., 1., 2., 3., 4., 5., 6., 7.],
                              [1., 2., 3., 4., 5., 6., 7., 8.],
                              [2., 3., 4., 5., 6., 7., 8., 9.],
                              [3., 4., 5., 6., 7., 8., 9., 10.],
                              [4., 5., 6., 7., 8., 9., 10., 11.],
                              [5., 6., 5., 6., 7., 8., 9., 10.],
                              [6., 7., 6., 5., 6., 7., 8., 9.],
                              [7., 8., 7., 6., 5., 6., 7., 8.],
                              [8., 9., 8., 7., 6., 5., 6., 7.],
                              [9., 10., 9., 8., 7., 6., 5., 6.],
                              [10., 11., 10., 9., 8., 7., 6., 5.]]))

    assert_array_equal(get_mtx("Révkomárom", "Komárom"),
                       array([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                              [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
                              [2., 3., 4., 5., 6., 5., 6., 7., 8., 9., 10.],
                              [3., 4., 5., 6., 7., 6., 5., 6., 7., 8., 9.],
                              [4., 5., 6., 7., 8., 7., 6., 5., 6., 7., 8.],
                              [5., 6., 7., 8., 9., 8., 7., 6., 5., 6., 7.],
                              [6., 7., 8., 9., 10., 9., 8., 7., 6., 5., 6.],
                              [7., 8., 9., 10., 11., 10., 9., 8., 7., 6., 5.]
                              ]))

    assert_array_equal(get_mtx("Tata", "Tatbánya"),
                       array([[0., 1., 2., 3., 4.],
                              [1., 0., 1., 2., 3.],
                              [2., 1., 0., 1., 2.],
                              [3., 2., 1., 0., 1.],
                              [4., 3., 2., 1., 2.],
                              [5., 4., 3., 2., 3.],
                              [6., 5., 4., 3., 4.],
                              [7., 6., 5., 4., 5.],
                              [8., 7., 6., 5., 4.]]))

    assert_array_equal(get_mtx("Budapest", "Debrecen"),
                       array([[0., 1., 2., 3., 4., 5., 6., 7., 8.],
                              [1., 2., 3., 4., 5., 6., 7., 8., 9.],
                              [2., 3., 4., 5., 6., 7., 6., 7., 8.],
                              [3., 4., 5., 6., 7., 8., 7., 8., 9.],
                              [4., 5., 6., 7., 8., 9., 8., 9., 10.],
                              [5., 6., 7., 8., 9., 10., 9., 10., 11.],
                              [6., 7., 8., 9., 10., 11., 10., 11., 12.],
                              [7., 8., 9., 10., 11., 12., 11., 12., 13.],
                              [8., 9., 10., 11., 12., 13., 12., 13., 14.]]))

    assert_array_equal(get_mtx("Debrecen", "Beregszász"),
                       array([[0., 1., 2., 3., 4., 5., 6., 7., 8.],
                              [1., 2., 3., 4., 5., 6., 7., 8., 9.],
                              [2., 3., 2., 3., 4., 5., 6., 7., 8.],
                              [3., 4., 3., 4., 3., 4., 5., 6., 7.],
                              [4., 5., 4., 5., 4., 3., 4., 5., 6.],
                              [5., 6., 5., 6., 5., 4., 5., 6., 7.],
                              [6., 7., 6., 7., 6., 5., 6., 7., 8.],
                              [7., 8., 7., 8., 7., 6., 7., 8., 9.],
                              [8., 9., 8., 9., 8., 7., 8., 9., 10.],
                              [9., 10., 9., 10., 9., 8., 9., 10., 11.],
                              [10., 11., 10., 11., 10., 9., 10., 11., 12.]]))

    assert_array_equal(
        get_mtx("Szentpétervár", "Vlagyivosztok"),
 array([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.],
        [2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.],
        [3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.],
        [4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.],
        [5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.],
        [6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        [7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 16., 17., 18.],
        [8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 17., 18., 19.],
        [9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 18., 19., 20.],
        [10., 11., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.],
        [11., 12., 11., 12., 13., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
        [12., 13., 12., 13., 14., 13., 14., 15., 16., 17., 18., 19., 20., 21.],
        [13., 14., 13., 14., 15., 14., 15., 16., 17., 18., 19., 20., 21., 22.]
        ]))

    assert_array_equal(
        get_mtx("Vlagyivosztok", "az óperenciás tengeren túl"),
array([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.],
      [1., 2., 3., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
      [2., 3., 4., 3., 4., 5., 6., 7., 8., 9., 8., 9., 10., 11.],
      [3., 4., 5., 4., 5., 6., 7., 8., 9., 10., 9., 10., 11., 12.],
      [4., 5., 6., 5., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.],
      [5., 6., 7., 6., 7., 8., 9., 10., 11., 12., 11., 12., 13., 14.],
      [6., 7., 8., 7., 8., 9., 10., 11., 12., 13., 12., 13., 14., 15.],
      [7., 8., 9., 8., 9., 10., 11., 12., 13., 14., 13., 14., 15., 16.],
      [8., 9., 10., 9., 10., 11., 12., 13., 14., 15., 14., 15., 16., 17.],
      [9., 10., 11., 10., 11., 12., 13., 14., 15., 16., 15., 16., 17., 18.],
      [10., 11., 12., 11., 12., 13., 14., 15., 16., 17., 16., 17., 18., 19.],
      [11., 12., 13., 12., 13., 14., 13., 14., 15., 16., 17., 18., 19., 20.],
      [12., 13., 14., 13., 14., 15., 14., 15., 16., 17., 18., 19., 20., 21.],
      [13., 14., 15., 14., 15., 16., 15., 16., 17., 16., 17., 18., 19., 20.],
      [14., 15., 16., 15., 16., 17., 16., 17., 18., 17., 18., 19., 20., 21.],
      [15., 16., 17., 16., 17., 18., 17., 18., 19., 18., 19., 18., 19., 20.],
      [16., 17., 18., 17., 18., 19., 18., 19., 20., 19., 20., 19., 20., 21.],
      [17., 18., 19., 18., 19., 20., 19., 20., 21., 20., 21., 20., 21., 22.],
      [18., 19., 20., 19., 18., 19., 20., 21., 22., 21., 22., 21., 22., 23.],
      [19., 20., 21., 20., 19., 20., 21., 22., 23., 22., 23., 22., 23., 24.],
      [20., 21., 22., 21., 20., 21., 22., 23., 24., 23., 24., 23., 24., 25.],
      [21., 22., 23., 22., 21., 22., 23., 24., 25., 24., 25., 24., 25., 26.],
      [22., 23., 24., 23., 22., 23., 24., 25., 26., 25., 26., 25., 26., 27.],
      [23., 24., 25., 24., 23., 24., 25., 26., 27., 26., 27., 26., 27., 28.],
      [24., 25., 26., 25., 24., 25., 26., 27., 28., 27., 28., 27., 28., 29.],
      [25., 26., 27., 26., 25., 26., 27., 28., 29., 28., 29., 28., 29., 30.],
      [26., 27., 26., 27., 26., 27., 28., 29., 30., 29., 30., 29., 30., 31.]]))


def test_mtx2graph():
    """Test if matrix of edit distances is correctly transformed to a
    directed graph object"""
    # similar to unittest, but 1 patch missing
    expG = DiGraph()
    expG.add_weighted_edges_from([((2, 2), (2, 1), 100),
                                  ((2, 2), (1, 2), 49),
                                  ((2, 2), (1, 1), 0),
                                  ((2, 1), (2, 0), 100),
                                  ((2, 1), (1, 1), 49),
                                  ((2, 0), (1, 0), 49),
                                  ((1, 2), (1, 1), 100),
                                  ((1, 2), (0, 2), 49),
                                  ((1, 1), (1, 0), 100),
                                  ((1, 1), (0, 1), 49),
                                  ((1, 0), (0, 0), 49),
                                  ((0, 2), (0, 1), 100),
                                  ((0, 1), (0, 0), 100)])
    exp = [(e, datadict["weight"]) for e, datadict in expG.edges.items()]

    outtuple = mtx2graph("ló", "hó")
    out = [(e, datadict["weight"])
           for e, datadict in outtuple[0].edges.items()]

    # assert expected and actual output is the same
    assert len(outtuple) == 3
    assert isinstance(outtuple, tuple)
    assert set(out) == set(exp)
    # the height. always 1 longer than the word bc + "#" (#ló)
    assert outtuple[1] == 3
    assert outtuple[2] == 3  # the width.

    # set up2: assert weights are passed on correctly
    expG = DiGraph()
    expG.add_weighted_edges_from([((2, 2), (2, 1), 11),
                                  ((2, 2), (1, 2), 7),
                                  ((2, 2), (1, 1), 0),
                                  ((2, 1), (2, 0), 11),
                                  ((2, 1), (1, 1), 7),
                                  ((2, 0), (1, 0), 7),
                                  ((1, 2), (1, 1), 11),
                                  ((1, 2), (0, 2), 7),
                                  ((1, 1), (1, 0), 11),
                                  ((1, 1), (0, 1), 7),
                                  ((1, 0), (0, 0), 7),
                                  ((0, 2), (0, 1), 11),
                                  ((0, 1), (0, 0), 11)])
    exp = [(e, datadict["weight"]) for e, datadict in expG.edges.items()]

    outtuple = mtx2graph("ló", "hó", w_del=11, w_ins=7)
    out = [(e, datadict["weight"])
           for e, datadict in outtuple[0].edges.items()]

    # assert expected and actual output is the same
    assert len(outtuple) == 3
    assert isinstance(outtuple, tuple)
    assert set(out) == set(exp)
    # the height. always 1 longer than the word bc + "#" (#ló)
    assert outtuple[1] == 3
    assert outtuple[2] == 3  # the width.


def test_tuples2editops():
    """Tuples correspond to points in the matrix (directed graph).
    The list of tuples encodes the edit operations necessary
    to get from string A to string B. Test if this list of tuples
    is correctly converted to a human readable format."""
    assert tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)],
                          "ló", "hó") == ['substitute l by h', 'keep ó']


def test_editops():
    """test if human-readable edit operations are concluded
    correctly from two strings"""
    assert editops("ló", "hó") == [('substitute l by h', 'keep ó')]
    assert editops("CCV", "CV", howmany_paths=2) == [
        ('delete C', 'keep C', 'keep V'),
        ('keep C', 'delete C', 'keep V')]


def test_apply_edit():
    pass  # unit == integration tests (no patches)


def test_get_howmany():
    pass  # unit == integration tests (no patches)


def test_pick_minmax():
    pass  # unit == integration tests (no patches)

    #
