"""unit tests for loanpy.helpers.py (2.0 BETA) with pytest 7.1.1"""

from unittest.mock import call, patch
from pathlib import Path
from os import remove
from inspect import ismethod
from ast import literal_eval

from pandas import DataFrame, Series, read_csv
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import raises
from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
from numpy import array, float32
from numpy.testing import assert_array_equal
from networkx import DiGraph

from loanpy import helpers as hp  # needed for hp.model, to plug in
from loanpy.helpers import (
    apply_edit,
    combine_ipalists,
    edit_distance_with2ops,
    editops,
    flatten,
    gensim_multiword,
    get_front_back_vowels,
    get_howmany,
    get_mtx,
    has_harmony,
    list2regex,
    model,
    mtx2graph,
    pick_minmax,
    plug_in_model,
    repair_harmony,
    tuples2editops)

from loanpy.qfysc import InventoryMissingError


class EtymMonkey:
    """used throughout the module"""
    pass


def test_plug_in_model():
    """test if model gets plugged into global variable correctly"""
    plug_in_model("xyz")
    assert hp.model == "xyz"
    plug_in_model(None)
    assert hp.model is None

def test_flatten():
    """check if nested lists are flattened and "" thrown out"""
    assert flatten([["a", "b"], ["c"]]) == ["a", "b", "c"]
    assert flatten([["wrd1", "wrd2", ""], ["wrd3", "", ""]]) == [
        "wrd1", "wrd2", "wrd3"]


def test_combine_ipalists():
    """test if old sounds are combined correctly"""

    # set up
    with patch("loanpy.helpers.flatten") as flatten_mock:
        flatten_mock.return_value = ["ki", "ke", "gi", "bu", "bo", "pu"]
        with patch("loanpy.helpers.product", side_effect=[
            [('k', 'i'), ('k', 'e'), ('g', 'i'), ('g', 'e')],
                [('b', 'u'), ('b', 'o'),
                 ('p', 'u'), ('p', 'o')]]) as product_mock:
            inlist = [[["k", "g"], ["i", "e"]], [["b", "p"], ["u", "o"]]]
            out = ["ki", "ke", "gi", "bu", "bo", "pu"]

            # assert
            assert combine_ipalists(inlist) == out

    # assert calls
    flatten_mock.assert_called_with(
        [["k i", "k e", "g i", "g e"], ["b u", "b o", "p u", "p o"]])
    product_mock.assert_has_calls([call(["k", "g"], ["i", "e"]),
                                   call(["b", "p"], ["u", "o"])])

    # tear down
    del inlist, out

def test_get_front_back_vowels():
    """test if front and back vowels are correctly replaced with "F" and "B" """

    #test 1: bociboci
    with patch("loanpy.helpers.token2class",
    side_effect=['x', 'o', 'x', 'i', 'x', 'o', 'x', 'i']) as token2class_mock:
        assert get_front_back_vowels(
        ['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']) == [
        'b', 'B', 't͡s', 'F', 'b', 'B', 't͡s', 'F']

    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']])

    #test2: tarka
    with patch("loanpy.helpers.token2class",
    side_effect=['x', 'o', 'x', 'x', 'o']) as token2class_mock:
        assert get_front_back_vowels(
        ['t', 'ɒ', 'r', 'k', 'ɒ']) ==  ['t', 'B', 'r', 'k', 'B']

    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['t', 'ɒ', 'r', 'k', 'ɒ']])

    #test3: se füle se
    with patch("loanpy.helpers.token2class",
    side_effect=['x', 'i', 'x', 'i', 'x', 'i', 'x', 'i']) as token2class_mock:
        assert get_front_back_vowels(
        ['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ', 'ʃ', 'ɛ']) ==  ['ʃ', 'F', 'f', 'F', 'l', 'F', 'ʃ', 'F']

    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ', 'ʃ', 'ɛ']])

    #test4: Keszthely
    with patch("loanpy.helpers.token2class",
    side_effect=['x', 'i', 'x', 'x', 'x', 'i', 'x']) as token2class_mock:
        assert get_front_back_vowels(
        ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']) ==  ['k', 'F', 's', 't', 'h', 'F', 'j']

    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']])

    #test5: Alsóörs
    with patch("loanpy.helpers.token2class",
    side_effect=['o', 'x', 'x', 'o', 'i', 'x', 'x']) as token2class_mock:
        assert get_front_back_vowels(
        ['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']) == ['B', 'l', 'ʃ', 'B', 'F', 'r', 'ʃ']

    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']])

    #test6: Bélatelep
    with patch("loanpy.helpers.token2class",
    side_effect=["x", "i", "x", "o", "x", "i", "x", "i", "x"]) as token2class_mock:
        assert get_front_back_vowels(
        ['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']) == [
        'b', 'F', 'l', 'B', 't', 'F', 'l', 'F', 'p']

    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']])

def test_has_harmony():
    """test if a words front-back vowel harmony is inferred correctly"""

    # test1 assert without tokenisation
    with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
        get_front_back_vowels_mock.return_value = ['b', 'B', 't͡s', 'F', 'b', 'B', 't͡s', 'F']
        assert has_harmony(
            ['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']) is False

    #assert call
    get_front_back_vowels_mock.assert_called_with(['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i'])

    # test2: assert word has vowel harmony, only back vowels
    with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
        get_front_back_vowels_mock.return_value = ['t', 'B', 'r', 'k', 'B']
        assert has_harmony(['t', 'ɒ', 'r', 'k', 'ɒ'])

    get_front_back_vowels_mock.assert_called_with(['t', 'ɒ', 'r', 'k', 'ɒ'])

    # test 3: assert word has vowel harmony, only front vowels
    with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
        get_front_back_vowels_mock.return_value = ['ʃ', 'F', 'f', 'F', 'l', 'F', 'ʃ', 'F']
        assert has_harmony(['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ', 'ʃ', 'ɛ'])

    get_front_back_vowels_mock.assert_called_with(['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ', 'ʃ', 'ɛ'])

def test_repair_harmony():
    """test if a words vowelharmony is repaired correctly"""

    # test1: input does have vowel harmony, so nothing happens
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']
        with patch("loanpy.helpers.has_harmony") as has_harmony_mock:
            has_harmony_mock.return_value = True
            with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
                get_front_back_vowels_mock.return_value = ['k', 'F', 's', 't', 'h', 'F', 'j']

                # assert that nothing happens if input word has vowel harmony
                assert repair_harmony(ipalist='kɛsthɛj') == [
                    ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']]

    # assert calls
    tokenise_mock.assert_called_with("kɛsthɛj")
    has_harmony_mock.assert_called_with(['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j'])
    get_front_back_vowels_mock.assert_not_called()

    #set up 2: patch tokenise, has_harmmony, get_front_back_vowels, token2class
    #test 2: input has more back vowels than front ones
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']
        with patch("loanpy.helpers.has_harmony") as has_harmony_mock:
            has_harmony_mock.return_value = False
            with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
                get_front_back_vowels_mock.return_value = ['B', 'l', 'ʃ', 'B', 'F', 'r', 'ʃ']
                with patch("loanpy.helpers.token2class",
                side_effect=["o", "x", "x", "o", "i", "x", "x"]) as token2class_mock:
                    # assert that the wrong front vowel ø is replaced by "B"
                    assert repair_harmony(ipalist='ɒlʃoːørʃ') == [
                        ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']]

    # assert calls
    tokenise_mock.assert_called_with("ɒlʃoːørʃ")
    has_harmony_mock.assert_called_with(['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'])
    get_front_back_vowels_mock.assert_called_with(['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ'])
    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']])


    #set up 3: patch tokenise, has_harmmony, get_front_back_vowels, token2class
    #test 3: input has more front vowels than back ones
    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        with patch("loanpy.helpers.has_harmony") as has_harmony_mock:
            has_harmony_mock.return_value = False
            with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
                get_front_back_vowels_mock.return_value = ['b', 'F', 'l', 'B', 't', 'F', 'l', 'F', 'p']
                with patch("loanpy.helpers.token2class",
                side_effect=["x", "i", "x", "o", "x", "i", "x", "i", "x"]) as token2class_mock:
                    # assert that the wrong front vowel "ɒ" is replace by "F"
                    assert repair_harmony(
                        ipalist=['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']) == [
                        ['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p']]

    # assert calls
    tokenise_mock.assert_not_called()
    has_harmony_mock.assert_called_with(['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p'])
    get_front_back_vowels_mock.assert_called_with(['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p'])
    token2class_mock.assert_has_calls([call(i, "asjp") for i in ['b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']])

    # set up4: define repetitive variables
    # test 4: input has same amount of front as back vowels
    bk = ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']
    bk2f = ['b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']
    bk2b = ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']
    bk_fb = ['b', 'B', 'l', 'B', 't', 'B', 'n', 'k', 'F', 'n', 'F', 'ʃ', 'F']

    with patch("loanpy.helpers.tokenise") as tokenise_mock:
        with patch("loanpy.helpers.has_harmony") as has_harmony_mock:
            has_harmony_mock.return_value = False
            with patch("loanpy.helpers.get_front_back_vowels") as get_front_back_vowels_mock:
                get_front_back_vowels_mock.return_value = bk_fb
                with patch("loanpy.helpers.token2class",
                side_effect=["x", "o", "x", "o", "x", "o", "x", "x", "i", "x", "i", "x", "i"]*2) as token2class_mock:
                    # assert words without vowelharmony with equally many front and back vowels
                    # are repaired in both possible ways
                    assert repair_harmony(ipalist=bk) == [bk2f, bk2b]


    # assert calls
    tokenise_mock.assert_not_called()
    has_harmony_mock.assert_called_with(bk)
    get_front_back_vowels_mock.assert_called_with(bk)
    token2class_mock.assert_has_calls([call(i, "asjp") for i in bk])

    # tear down
    del bk, bk2f, bk2b


def test_gensim_multiword():
    """use gensim's built-in test suit to check if it works"""

    # test first where no setup needed
    # test first break without returning wordpair
    assert gensim_multiword(
        recip_transl=None,
        donor_transl=0.1,
        return_wordpair=False) == -1
    # test first break with returning wordpair
    assert gensim_multiword(recip_transl=None,
                            donor_transl=0.1,
                            return_wordpair=True) == (-1,
                                                      "!<class 'NoneType'>!",
                                                      "!<class 'float'>!")

    class GensimMonkey:
        def __init__(self): pass

        def similarity(self, word1, word2): return 0

        def has_index_for(self, w): return True
    # set up: mock gensim.api.load:
    with patch("loanpy.helpers.load") as load_mock:
        mockgensim = GensimMonkey()
        load_mock.return_value = mockgensim

        # assert gensim_multiword's result
        assert gensim_multiword("word1", "word2") == 0
        # assert api would have loadad the correct model
        assert hp.model == mockgensim

        # assert call
        load_mock.assert_called_with("word2vec-google-news-300")

        # tear down hp.model
        hp.model = None

        # plug in different wordvectors
        # assert gensim_multiword's result
        assert gensim_multiword("word1", "word2", wordvectors="somemodel") == 0
        # assert api would have loadad the correct model
        assert hp.model == mockgensim

    # assert call
    load_mock.assert_called_with("somemodel")

    # todo: find out how to mock a MemoryError

    # set up2: plug in a mock word2vec model from gensim's test suite
    hp.model = word2vec.Word2Vec(common_texts, min_count=1).wv

    # assert that some distances are calculated correctly
    assert gensim_multiword("human, computer",
                            "interface") == float32(0.10940766334533691)
    assert gensim_multiword(
        "human, computer",
        "interface",
        return_wordpair=True) == (
        float32(0.10940766),
        'human',
        'interface')
    assert gensim_multiword("computer, human",
                            "interface") == float32(0.10940766334533691)
    assert gensim_multiword(
        "computer, human",
        "interface",
        return_wordpair=True) == (
        float32(0.10940766),
        'human',
        'interface')

    # assert KeyError is skipped
    assert gensim_multiword("human, missingword",
                            "interface") == float32(0.10940766334533691)
    assert gensim_multiword(
        "human, missingword",
        "interface",
        return_wordpair=True) == (
        float32(0.10940766),
        'human',
        'interface')

    # assert missing words result in similarity score of -1
    assert gensim_multiword("human, computer", "missingword") == float32(-1)

    # assert missing src word shows right warning if wordpairs are returned
    assert gensim_multiword("human, computer",
                            "missingword",
                            return_wordpair=True) == (float32(-1),
                                                      '',
                                                      'source word \
not in model')
    # assert missing tgt word shows right warning if wordpairs are returned
    assert gensim_multiword("missingword",
                            "human, computer",
                            return_wordpair=True) == (float32(-1),
                                                      'target word \
not in model',
                                                      '')
    # assert right warning if both words are missing
    assert gensim_multiword("missing1", "missing2", return_wordpair=True) == (
        float32(-1), 'target word not in model', 'source word not in model')

    # assert loop is interrupted as soon as similarity score == 1
    # loop is certainly broken because "1" is not a numpy.float32
    assert gensim_multiword("human, computer", "computer") == 1
    assert gensim_multiword(
        "human, computer",
        "computer",
        return_wordpair=True) == (
        1,
        'computer',
        'computer')

    # tear down hp.model by setting it to its previous value (None)
    hp.model = None  # del hp.model would lead to errors
    del GensimMonkey


def test_list2regex():
    """test if list of phonemes is correctly converted to regular expression"""
    assert list2regex(["b", "k", "v"]) == "(b|k|v)"
    assert list2regex(["b", "k", "-", "v"]) == "(b|k|v)?"
    assert list2regex(["b", "k", "-", "v", "mp"]) == "(b|k|v|mp)?"
    assert list2regex(["b", "k", "-", "v", "mp", "mk"]) == "(b|k|v|mp|mk)?"
    assert list2regex(["o"]) == '(o)'
    assert list2regex(["ʃʲk"]) == '(ʃʲk)'
    assert list2regex(["-"]) == ""


def test_edit_distance_with2ops():
    """test if editdistances are calculated correctly"""

    # default weight is 100 per deletion and 49 per insertion
    # in 80 tests around the world
    assert edit_distance_with2ops("ajka", "Rajka") == 49
    assert edit_distance_with2ops("Rajka", "ajka") == 100
    assert edit_distance_with2ops("Debrecen", "Mosonmagyaróvár") == 1386
    assert edit_distance_with2ops("Bécs", "Hegyeshalom") == 790
    assert edit_distance_with2ops("Hegyeshalom", "Mosonmagyaróvár") == 1388
    assert edit_distance_with2ops("Mosonmagyaróvár", "Győr") == 1398
    # 4 del + 4 ins = 4*49+4*100
    assert edit_distance_with2ops("Győr", "Tata") == 596
    assert edit_distance_with2ops("Tata", "Tatabánya") == 245  # 5 ins: 5*49
    assert edit_distance_with2ops("Tatabánya", "Budapest") == 994
    assert edit_distance_with2ops("Budapest", "Komárom") == 1143
    # 4 ins + 1 del: 4*49+100
    assert edit_distance_with2ops("Komárom", "Révkomárom") == 296
    # 4 del + 1 ins: 4*100+49
    assert edit_distance_with2ops("Révkomárom", "Komárom") == 449
    assert edit_distance_with2ops("Komárom", "Budapest") == 1092
    assert edit_distance_with2ops("Budapest", "Debrecen") == 1043
    assert edit_distance_with2ops("Debrecen", "Beregszász") == 843
    assert edit_distance_with2ops("Beregszász", "Kiev") == 1047
    assert edit_distance_with2ops("Kiev", "Moszkva") == 594
    assert edit_distance_with2ops("Moszkva", "Szenpétervár") == 990
    assert edit_distance_with2ops("Szentpétervár", "Vlagyivosztok") == 1639
    assert edit_distance_with2ops("Vlagyivosztok", "Tokió") == 1247
    assert edit_distance_with2ops("Tokió", "New York") == 594
    assert edit_distance_with2ops("New York", "Bécs") == 996

    # check if custom weights for insertion work. deletion always costs 1.
    assert edit_distance_with2ops("ajka", "Rajka", w_ins=90) == 90
    assert edit_distance_with2ops("Rajka", "ajka", w_ins=90) == 100
    assert edit_distance_with2ops(
        "Debrecen", "Mosonmagyaróvár", w_ins=90) == 1960


def test_get_mtx():
    """test if distance matrix between two words is set up correctly"""

    # set up by mocking numpy.zeros and defining the expected output
    with patch("loanpy.helpers.zeros") as zeros_mock:
        zeros_mock.return_value = array([[0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0.]])

        exp = array([[0., 1., 2., 3., 4.],
                     [1., 2., 3., 4., 5.],
                     [2., 3., 2., 3., 4.],
                     [3., 4., 3., 2., 3.],
                     [4., 5., 4., 3., 2.]])

        # assert
        assert_array_equal(get_mtx("Bécs", "Pécs"), exp)

    # assert call
    zeros_mock.assert_called_with((5, 5))

    # tear down
    del exp


def test_mtx2graph():
    """test if numpy matrix is correctly converted to
     networkx DiGraph (directed graph)"""

    # set up by mocking get_mtx and defining actual and expected output
    # output is tuple: element 1 is a Graph, two is the height, 3 the width.
    # expG is expected output graph. It's a directed graph. The tuples show the
    # nodes in the coordinate system between which the edges go,
    # the number next to the tuples is the weight of each edge
    # graph objects store edges in arbitrary order, cant compare them directly
    # instead have to turn their data into lists and compare their sets

    with patch("loanpy.helpers.get_mtx") as get_mtx_mock:
        get_mtx_mock.return_value = array([[0., 1., 2.],
                                           [1., 2., 3.],
                                           [2., 3., 2.]])
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

    # assert call
    get_mtx_mock.assert_called_with("ló", "hó")

    # set up2: assert weights are passed on correctly
    with patch("loanpy.helpers.get_mtx") as get_mtx_mock:
        get_mtx_mock.return_value = array([[0., 1., 2.],
                                           [1., 2., 3.],
                                           [2., 3., 2.]])
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

    # assert call
    get_mtx_mock.assert_called_with("ló", "hó")

    # tear down
    del out, exp, expG, outtuple


def test_tuples2editops():
    """assert that edit operations coded as tuples
    are converted to natural language correctly"""

    # set up by mocking numpy.subtract and numpy.array_equiv
    arrayequiv_expcalls = [
        (array([0, 1]), [1, 1]),  # first step diagonal? No.
        (array([0, 1]), [0, 1]),  # first step horizontal? Yes.
        (array([1, 0]), [1, 1]),  # second step diagonal? No.
        (array([1, 0]), [0, 1]),  # second step horizontal? No.
        (array([1, 0]), [1, 0]),  # second step vertical? Yes.
        (array([0, 1]), [0, 1]),  # step before (first step) horizontal? Yes.
        (array([1, 1]), [1, 1])  # third step diagonal? Yes.
    ]

    with patch("loanpy.helpers.subtract", side_effect=[
        array([0, 1]), array([1, 0]), array([0, 1]), array([1, 1])]
    ) as subtract_mock:
        with patch("loanpy.helpers.array_equiv", side_effect=[
                False, True, False, False,
                True, True, True]) as array_equiv_mock:

            # assert list of tuples is correctly converted to list of strings
            assert tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)],
                                  "ló", "hó") == ['substitute l by h',
                                                    'keep ó']

    # assert calls
    subtract_mock.assert_has_calls([
        call((0, 1), (0, 0)),  # first step = horizontal
        call((1, 1), (0, 1)),  # second step = vertical
        # recheck first step -> convert to "substitute"
        call((0, 1), (0, 0)),
        call((2, 2), (1, 1))  # last step diagonal
    ])
    # cant use assert_called_with bc nparrays can only be compared
    # through assert_array_equal
    for arraypair0, arraypair1 in zip(
            array_equiv_mock.call_args_list, arrayequiv_expcalls):
        # first [0] turns call object to tuple
        assert_array_equal(arraypair0[0][0], arraypair1[0])
        assert_array_equal(arraypair0[0][1], arraypair1[1])

    # tear down
    del arrayequiv_expcalls


def test_editops():
    """test if all shortest paths of editoperations are extracted correctly"""

    # set up by mocking mtx2graph, networkx.all_shortest_paths, tuples2editops
    G = DiGraph()
    G.add_weighted_edges_from(
        [((2, 2), (2, 1), 100),
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
         ((0, 1), (0, 0), 100)]
    )
    with patch("loanpy.helpers.mtx2graph") as mtx2graph_mock:
        mtx2graph_mock.return_value = (G, 3, 3)
        with patch("loanpy.helpers.shortest_path") as shortest_path_mock:
            shortest_path_mock.return_value = [(2, 2), (1, 1), (0, 1), (0, 0)]
            with patch("loanpy.helpers.tuples2editops", side_effect=[
                    ['substitute l by h', 'keep ó']]) as tuples2editops_mock:

                # assert that 2 strings are correctly converted to editops
                assert editops("ló", "hó") == [('substitute l by h', 'keep ó')]

    # assert calls
    mtx2graph_mock.assert_called_with("ló", "hó", 100, 49)
    shortest_path_mock.assert_called_with(
        G, (2, 2), (0, 0), weight="weight")
    tuples2editops_mock.assert_called_with(
        [(0, 0), (0, 1), (1, 1), (2, 2)], "ló", "hó")

    # set up2: to return 2 paths
    with patch("loanpy.helpers.mtx2graph") as mtx2graph_mock2:
        mtx2graph_mock2.return_value = (G, 3, 4)
        with patch("loanpy.helpers.all_shortest_paths"
                   ) as all_shortest_paths_mock:
            all_shortest_paths_mock.return_value = [
                [(2, 3), (1, 2), (0, 1), (0, 0)],
                [(2, 3), (1, 2), (1, 1), (0, 0)]]
            with patch("loanpy.helpers.tuples2editops", side_effect=[
                ['delete C', 'keep C', 'keep V'],
                ['keep C', 'delete C', 'keep V']]
            ) as tuples2editops_mock:

                # assert that both paths are extracted
                assert editops("CCV", "CV", howmany_paths=2) == [
                    ('delete C', 'keep C', 'keep V'),
                    ('keep C', 'delete C', 'keep V')]

    # assert calls
    mtx2graph_mock2.assert_called_with("CCV", "CV", 100, 49)
    all_shortest_paths_mock.assert_called_with(
        G, (2, 3), (0, 0), weight="weight")
    tuples2editops_mock.assert_has_calls([
        call([(0, 0), (0, 1), (1, 2), (2, 3)], "CCV", "CV"),
        call([(0, 0), (1, 1), (1, 2), (2, 3)], "CCV", "CV")])

    # set up3: assert weights are passed on correctly
    with patch("loanpy.helpers.mtx2graph") as mtx2graph_mock2:
        mtx2graph_mock2.return_value = (G, 3, 4)
        with patch("loanpy.helpers.all_shortest_paths"
                   ) as all_shortest_paths_mock:
            all_shortest_paths_mock.return_value = [
                [(2, 3), (1, 2), (0, 1), (0, 0)],
                [(2, 3), (1, 2), (1, 1), (0, 0)]]
            with patch("loanpy.helpers.tuples2editops", side_effect=[
                ['delete C', 'keep C', 'keep V'],
                ['keep C', 'delete C', 'keep V']]
            ) as tuples2editops_mock:

                # assert that both paths are extracted
                assert editops("CCV", "CV", howmany_paths=2,
                               w_del=4, w_ins=35) == [
                    ('delete C', 'keep C', 'keep V'),
                    ('keep C', 'delete C', 'keep V')]

    # assert calls
    mtx2graph_mock2.assert_called_with("CCV", "CV", 4, 35)
    all_shortest_paths_mock.assert_called_with(
        G, (2, 3), (0, 0), weight="weight")
    tuples2editops_mock.assert_has_calls([
        call([(0, 0), (0, 1), (1, 2), (2, 3)], "CCV", "CV"),
        call([(0, 0), (1, 1), (1, 2), (2, 3)], "CCV", "CV")])

    # tear down
    del G


def test_apply_edit():
    """test if editoperations are correctly applied to words"""
    assert apply_edit("ló", ('substitute l by h', 'keep ó')) == ['h', 'ó']
    assert apply_edit(["l", "ó"],
                      ('substitute l by h', 'keep ó')) == ['h', 'ó']
    assert apply_edit(['f', 'ɛ', 'r', 'i', 'h', 'ɛ', 'ɟ'],
                      ('insert d',
                       'insert u',
                       'insert n',
                       'insert ɒ',
                       'insert p',
                       'substitute f by ɒ',
                       'delete ɛ',
                       'keep r',
                       'delete i',
                       'delete h',
                       'delete ɛ',
                       'substitute ɟ by t')
                      ) == ['d', 'u', 'n', 'ɒ', 'p', 'ɒ', 'r', 't']
    assert apply_edit(['t͡ʃ',
                       'ø',
                       't͡ʃ'],
                      ("substitute t͡ʃ by f",
                       "insert r",
                       "keep ø",
                       "substitute t͡ʃ by t͡ʃː")) == ['f', 'r', 'ø', 't͡ʃː']


def test_get_howmany():
    """test if gethowmany correctly returns a tuple whose product is \
as close as possible, but not less than the first number of the input tuple, \
while the two last elements of the \
output tuple are not higher than the two last numbers of the input tuple"""
    assert get_howmany(10, 2, 2) == (3, 2, 2)
    assert get_howmany(100, 2, 2) == (25, 2, 2)
    assert get_howmany(100, 9, 2) == (8, 7, 2)
    assert get_howmany(1000, 9, 2) == (56, 9, 2)
    assert get_howmany(1000, 3, 2) == (167, 3, 2)
    assert get_howmany(0, 0, 0) == (0, 0, 0)
    assert get_howmany(1000, 1000, 1000) == (10, 10, 10)
    assert get_howmany(500, 0, 2) == (500, 0, 2)


def test_pick_minmax():
    """test if correct number of mins/maxs is picked"""
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf")
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1) == ["c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2) == ["c", "a"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3) == ["c", "a", "b"]
    # test with max
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"),
                       max) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1, max) == ["b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2, max) == ["b", "a"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3,
                       max) == ["b", "a", "c"]

    # test with return_all=True
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"), True
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1, min, True
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2, min, True
                       ) == ["c", "a", "b"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3, min, True
                       ) == ["c", "a", "b"]
    # test with max
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], float("inf"),
                       max, True) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 1, max, True
                       ) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 2, max, True
                       ) == ["b", "a", "c"]
    assert pick_minmax([("a", 5), ("b", 7), ("c", 3)], 3,
                       max, True) == ["b", "a", "c"]
