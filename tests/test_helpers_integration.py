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
    editops,
    get_front_back_vowels,
    gensim_multiword,
    get_mtx,
    has_harmony,
    combine_ipalists,
    clusterise,
    mtx2graph,
    repair_harmony,
    tuples2editops)

from loanpy.qfysc import InventoryMissingError

PATH2FORMS = Path(__file__).parent / "input_files" / "forms.csv"


def test_plug_in_model():
    pass  # unittest = integration test

def test_flatten():
    pass  # unittest == integration test


def test_combine_ipalists():
    """test if combinatorics is correctly applied to phonemes (#odometer)"""
    inlist = [[["k", "g"], ["i", "e"]], [["b", "p"], ["u", "o"]]]
    out = ["ki", "ke", "gi", "ge", "bu", "bo", "pu", "po"]
    assert combine_ipalists(inlist) == out
    del inlist, out

def test_has_harmony():
    """Test if it is detected correctly whether a word does or does not
    have front-back vowel harmony"""
    assert has_harmony(
        ['b', 'o', 't͡s', 'i', 'b', 'o', 't͡s', 'i']) is False
    assert has_harmony("bot͡sibot͡si") is False
    assert has_harmony("tɒrkɒ") is True
    assert has_harmony("ʃɛfylɛʃɛ") is True


def test_repair_harmony():
    """test if words without front-back vowel harmony are repaired correctly"""
    assert repair_harmony('kɛsthɛj') == [
        ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']]
    assert repair_harmony('ɒlʃoːørʃ') == [
        ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ']]
    assert repair_harmony([
        'b', 'eː', 'l', 'ɒ', 't', 'ɛ', 'l', 'ɛ', 'p']) == [
        ['b', 'eː', 'l', 'F', 't', 'ɛ', 'l', 'ɛ', 'p']]
    assert repair_harmony('bɒlɒtonkɛnɛʃɛ') == [
        ['b', 'F', 'l', 'F', 't', 'F', 'n', 'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ'],
        ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n', 'k', 'B', 'n', 'B', 'ʃ', 'B']]


def test_get_front_back_vowels():
    """test if front and back vowels are fetched correctly"""
    assert get_front_back_vowels(['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j']) == [
        'k', 'F', 's', 't', 'h', 'F', 'j']
    assert get_front_back_vowels([
    'ɒ', 'l', 'ʃ', 'oː', 'ø', 'r', 'ʃ']) == [
    'B', 'l', 'ʃ', 'B', 'F', 'r', 'ʃ']
    assert get_front_back_vowels(['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                        'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']) == [
        'b', 'B', 'l', 'B', 't', 'B', 'n', 'k', 'F', 'n', 'F', 'ʃ', 'F']
    assert get_front_back_vowels(['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                        'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']) == [
                        'b', 'B', 'l', 'B', 't', 'B', 'n',
                            'k', 'F', 'n', 'F', 'ʃ', 'F']


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
