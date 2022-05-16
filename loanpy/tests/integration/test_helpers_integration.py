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

from loanpy.helpers import (Etym, InventoryMissingError, cldf2pd, editops,
gensim_multiword, get_mtx,
combine_ipalists, clusterise, mtx2graph,
read_cvfb, read_dst, read_forms, read_inventory, tuples2editops)

PATH2FORMS = Path(__file__).parent / "input_files" / "forms.csv"

def test_plug_in_model():
    pass #unittest = integration test

def test_read_cvfb():
    """reading in cvfb.txt which is always in the same place"""
    cvfb = read_cvfb()
    assert isinstance(cvfb, tuple)
    assert len(cvfb) == 2
    assert isinstance(cvfb[0], dict) # extremely long dictionary, so just type is checked
    assert isinstance(cvfb[1], dict)
    assert len(cvfb[0]) == 6358 # extremely long dictionary, so just type is checked
    assert len(cvfb[1]) == 1240
    assert all(cvfb[0][val] in ["C", "V"] for val in cvfb[0])
    assert all(cvfb[1][val] in ["F", "B", "V"] for val in cvfb[1])

    #verify that the dict is actually based on ipa_all.csv
    #and that "C" corresponds to "+" and "V" "-" in col "ipa".
    dfipa = read_csv(Path(__file__).parent.parent.parent / "ipa_all.csv")
    for i, c in zip(dfipa.ipa, dfipa.cons):
        if c == "+":
            assert cvfb[0][i] == "C"
        elif c == "-":
            assert cvfb[0][i] == "V"
        else:
            assert cvfb[0].get(i, "notindict") == "notindict"

    del dfipa, cvfb

def test_read_forms():
    #test first break
    assert read_forms(None) is None

    # set up
    dfexp = DataFrame({"Language_ID": [1, 2],
                       "Segments": ["abc", "xyz"],  # pulled together segments
                       "Cognacy": [1, 1]})

    assert read_forms(None) is None
    assert_frame_equal(read_forms(PATH2FORMS), dfexp, check_dtype=False)

    #tear down
    del dfexp

def test_cldf2pd():
    """test if the cldf format is correctly tranformed to a pandas dataframe"""

    # set up
    dfin = read_csv(PATH2FORMS)
    dfexp = DataFrame({"Target_Form": ["x y z"],
                       "Source_Form": ["a b c"],
                       "Cognacy": [1]})

    # assert
    assert cldf2pd(None, srclg="whatever", tgtlg="wtvr2") is None
    assert_frame_equal(cldf2pd(dfin, srclg=1, tgtlg=2), dfexp)

    # tear down
    del dfexp, dfin

def test_read_inventory():
    #assert first two exceptions: inv is not None and inv and forms are None
    assert read_inventory("some_inv", "some_formscsv") == "some_inv"
    assert read_inventory(None, None) is None

    #assert calculations
    assert read_inventory(None,  ["a", "aab", "bc"]) == set(['a', 'b', 'c'])
    assert read_inventory(None, ["a", "ab", "baac"], clusterise
    ) == set(['aa', 'bb', 'c'])

def test_read_dst():
    out = read_dst("weighted_feature_edit_distance")
    assert ismethod(out)

def test_flatten():
    pass # unittest == integration test

def test_combine_ipalists():
    inlist = [[["k", "g"], ["i", "e"]], [["b", "p"], ["u", "o"]]]
    out = ["ki", "ke", "gi", "ge", "bu", "bo", "pu", "po"]
    assert combine_ipalists(inlist) == out
    del inlist, out

def test_form2list():
    pass #unit == integration test

def test_init():

    # set up: initiate without args
    mocketym = Etym()

    #assert that the right number of class attributes were instanciated
    assert len(mocketym.__dict__) == 7

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

    #assert vow2fb was read correctly
    assert isinstance(mocketym.vow2fb, dict)
    assert len(mocketym.vow2fb) == 1240
    assert mocketym.vow2fb["e"] == "F"
    assert mocketym.vow2fb["i"] == "F"
    assert mocketym.vow2fb["o"] == "B"
    assert mocketym.vow2fb["u"] == "B"

    #assert the other 5 attributes were read correctly
    assert mocketym.dfety is None
    assert mocketym.phoneme_inventory is None
    assert mocketym.clusters is None
    assert mocketym.struc_inv is None
    ismethod(mocketym.distance_measure)

    #tear down
    del mocketym

    #set up2: run with advanced parameters
    #input vars for init params
    mocketym = Etym(formscsv=PATH2FORMS, srclg=1, tgtlg=2)

    #assert right number of attributes was initiated (7)
    assert len(mocketym.__dict__) == 7

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

    #assert the other 4 attributes were read correctly
    assert mocketym.phoneme_inventory == {'x', 'y', 'z'}
    assert mocketym.clusters == {'x', 'y', 'z'}
    assert mocketym.struc_inv == {"CVC"}
    ismethod(mocketym.distance_measure)

    #tear down
    del mocketym

def test_read_strucinv():
    # set up rest
    forms = ["ab", "ab", "aa", "bb", "bb", "bb"]  # from forms.csv in cldf
    etym = Etym()
    # assert with different parameter combinations
    assert etym.read_strucinv(struc_inv=["a", "b", "c"],
                              forms=None) == ["a", "b", "c"]
    assert etym.read_strucinv(struc_inv=None,
                              forms=None) is None
    # now just read the most frquent 2 structures. VV is the 3rd frquent. so
    # not in the output.
    assert etym.read_strucinv(struc_inv=None,
                              forms=forms, howmany=2) == ["CC", "VC"]

    #tear down
    del forms, etym

def test_word2struc():
    etym = Etym()
    assert etym.word2struc("t͡ʃɒlːoːkøz") == "CVCVCVC"
    assert etym.word2struc("hortobaːɟ") == "CVCCVCVC"
    assert etym.word2struc("lɒk#sɒkaːlːɒʃ") == "CVCCVCVCVC" # hashtag is ignored
    assert etym.word2struc("boɟ!ɒ") == "CVCV" # exclam. mark is ignored
    assert etym.word2struc("ɡɛlːeːr") == "CVCVC"

    #tear down
    del etym

def test_word2struc_keepcv():
    etym = Etym()
    assert etym.word2struc(['C', 'V', 'C', 'V', 'k', 'ø', 'z']) == "CVCVCVC"
    assert etym.word2struc(['h', 'o', 'r', 'C', 'V', 'C', 'V', 'ɟ']) == "CVCCVCVC"
    assert etym.word2struc(['l', 'V', 'k', 's', 'ɒ', 'k', 'V', 'C', 'V', 'ʃ']) == "CVCCVCVCVC"
    assert etym.word2struc(['C', 'o', '!', 'ɟ', 'V']) == "CVCV"
    assert etym.word2struc(["C", "V", "lː", "eː", "r"]) == "CVCVC"
    del etym

def test_harmony():
    etym = Etym()
    assert etym.harmony(['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']) is False
    assert etym.harmony("bot͡sibot͡si") is False
    assert etym.harmony("tɒrkɒ") is True
    assert etym.harmony("ʃɛfylɛʃɛ") is True
    del etym

def test_adapt_harmony():
    etym = Etym()
    assert etym.adapt_harmony(ipalist='kɛsthɛj') == [
    ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']]
    assert etym.adapt_harmony(ipalist='ɒlʃoːørʃ') == [
    ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']]
    assert etym.adapt_harmony(ipalist=[
    'b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']) == [
    ['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p']]
    assert etym.adapt_harmony(ipalist='bɒlɒtonkɛnɛʃɛ') == [
    ['b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'],
    ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']]
    del etym

def test_get_fb():
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
    #test with inventory manually plugged in
    etym = Etym(inventory=["a", "b", "c"])
    scdictbase = etym.get_scdictbase(write_to=False)
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["b", "c", "a"] #b is obv most similar to p
    assert scdictbase["h"] == ["c", "b", "a"]
    assert scdictbase["e"] == ["a", "b", "c"]
    assert scdictbase["C"] == ["b", "c"]
    assert scdictbase["V"] == ["a"]
    assert scdictbase["F"] == ["a"]
    assert scdictbase["B"] == []
    del etym, scdictbase

    #test with invetory extracted from forms.csv
    etym = Etym(PATH2FORMS, srclg=1, tgtlg=2)
    scdictbase = etym.get_scdictbase(write_to=False)
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["z", "x", "y"] #IPA z is most similar to IPA p
    assert scdictbase["h"] == ["x", "z", "y"]
    assert scdictbase["e"] == ["y", "z", "x"]
    assert scdictbase["C"] == ["x", "z"]
    assert scdictbase["V"] == ["y"]
    assert scdictbase["F"] == ["y"]
    assert scdictbase["B"] == []
    del etym, scdictbase

    #test if written correctly and if param most_common works

    #set up
    etym = Etym(inventory=["a", "b", "c"])
    path2scdict_integr_test = Path(__file__).parent / "integr_test_scdict.txt"
    etym.get_scdictbase(write_to=path2scdict_integr_test, most_common=2)
    with open(path2scdict_integr_test, "r", encoding="utf-8") as f:
        scdictbase = literal_eval(f.read())

    #assert
    assert isinstance(scdictbase, dict)
    assert len(scdictbase) == 6371
    assert scdictbase["p"] == ["b", "c"] #b is obv most similar to p
    assert scdictbase["h"] == ["c", "b"]
    assert scdictbase["e"] == ["a", "b"]
    assert scdictbase["C"] == ["b", "c"]
    assert scdictbase["V"] == ["a"]
    assert scdictbase["F"] == ["a"]
    assert scdictbase["B"] == []

    #tear down
    remove(path2scdict_integr_test)
    del etym, scdictbase, path2scdict_integr_test

def test_rankclosest():

    #assert error is being raised correctly
    etym = Etym()
    with raises(InventoryMissingError) as inventorymissingerror_mock:
        etym.rank_closest(ph="d", howmany=3, inv=None)
    assert str(inventorymissingerror_mock.value
    ) == "define phoneme inventory or forms.csv"
    del etym

    #assert phonemes are ranked correctly
    etym = Etym(inventory=["a", "b", "c"])
    assert etym.rank_closest(ph="d") == "b, c, a"
    assert etym.rank_closest(ph="d", howmany=2) == "b, c"
    assert etym.rank_closest(ph="d", inv=["r", "t", "l"], howmany=1) == "t"
    del etym

def test_rankclosest_struc():
    #assert error is raised correctly if inventory is missing
    etym = Etym()
    with raises(InventoryMissingError) as inventorymissingerror_mock:
        #assert error is raised
        etym.rank_closest_struc(struc="CV", howmany=float("inf"))
        assert str(inventorymissingerror_mock.value
            ) == "define phonotactic inventory or forms.csv"
    del etym

    #assert structures are ranked correctly
    etym = Etym(PATH2FORMS, srclg=1, tgtlg=2)
    assert etym.rank_closest_struc(struc="CVCV") == "CVC" #struc_inv is only lg2 aka "xyz"
    assert etym.rank_closest_struc(struc="CVCV", howmany=3,
    inv=["CVC", "CVCVV", "CCCC", "VVVVVV"]) == "CVCVV, CVC, CCCC"
    del etym

def test_gensim_multiword():
    pass #checked in detail and unit == integration test (!)

def test_list2regex():
    pass #no patches in unittest

def test_edit_distance_with2ops():
    pass #no patches in unittest

def test_get_mtx():
    exp = array([[0., 1., 2., 3., 4.],
                 [1., 2., 3., 4., 5.],
                 [2., 3., 2., 3., 4.],
                 [3., 4., 3., 2., 3.],
                 [4., 5., 4., 3., 2.]])
    assert_array_equal(get_mtx("Bécs", "Pécs"), exp)
    del exp

def test_mtx2graph():
    #similar to unittest, but 1 patch missing
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

def test_tuples2editops():
    assert tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)],
    "ló", "hó") == ['substitute l by h', 'keep ó']

def test_editops():
    assert editops("ló", "hó") == [('substitute l by h', 'keep ó')]
    assert editops("CCV", "CV", howmany_paths=2) == [
        ('delete C', 'keep C', 'keep V'),
        ('keep C', 'delete C', 'keep V')]

def test_apply_edit():
    pass # unit == integration tests (no patches)

def test_get_howmany():
    pass # unit == integration tests (no patches)

def test_pick_mins():
    pass # unit == integration tests (no patches)


























    #
