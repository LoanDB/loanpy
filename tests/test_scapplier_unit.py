# -*- coding: utf-8 -*-
"""unit tests for loanpy.scapplier with pytest 7.1.2"""

import pytest
from loanpy.scapplier import (Adrc, move_sc, edit_distance_with2ops, apply_edit,
                          list2regex, tuples2editops, get_mtx,
                          mtx2graph, dijkstra, add_edge, substitute_operations)
from unittest.mock import patch, call
from tempfile import TemporaryDirectory
from collections import OrderedDict
from os import remove
from pathlib import Path
import json

def test_init_with_files(tmp_path):
    # Create temporary files for sound correspondence dictionary and inventories
    sc_path = tmp_path / "sound_correspondences.json"
    sc_path.write_text('{"a": {"b": "c"}}')
    prosodic_inventory_path = tmp_path / "inventories.json"
    prosodic_inventory_path.write_text('["CVCV"]')

    # Initialize Adrc object with temporary file paths
    obj = Adrc(sc=str(sc_path), prosodic_inventory=str(prosodic_inventory_path))

    # Assert that sound correspondence dictionary and inventories were loaded properly
    assert obj.sc == {"a": {"b": "c"}}
    assert obj.prosodic_inventory == ["CVCV"]

def test_init_without_files():
    # Initialize Adrc object without files
    obj = Adrc()

    # Assert that sound correspondence dictionary and inventories are None
    assert obj.sc is None
    assert obj.prosodic_inventory is None

def test_set_sc():
    """
    Test if sound correspondences are plugged in correctly into  attribute
    """
    obj = Adrc()
    obj.set_sc("lol")
    assert obj.sc == "lol"

def test_set_prosodic_inventory():
    """
    Test if inventories are plugged in correctly into attribute
    """
    obj = Adrc()
    obj.set_prosodic_inventory("rofl")
    assert obj.prosodic_inventory == "rofl"

class AdrcMonkey:
    def __init__(self):
        self.sc = [{},{},{},{},{},{}]
        self.prosodic_inventory = []

def test_move_sc():
    """test if sound correspondences are moved correctly"""
    # no setup, teardown, or patch needed here
    assert move_sc(sclistlist=[["x", "x"]], whichsound=0,
                   out=[[]]) == ([["x"]], [["x"]])

    assert move_sc(sclistlist=[["x", "x"], ["y", "y"], ["z"]], whichsound=0,
                   out=[["a"], ["b"], ["c"]]) == ([["x"], ["y", "y"], ["z"]],
                                                  [["a", "x"], ["b"], ["c"]])

    assert move_sc(sclistlist=[["x", "x"], ["y", "y"], ["z"]], whichsound=1,
                   out=[["a"], ["b"], ["c"]]) == ([["x", "x"], ["y"], ["z"]],
                                                  [["a"], ["b", "y"], ["c"]])

    assert move_sc(sclistlist=[["", "x", "$"], ["", "y", "$"], ["", "$"]],
                   whichsound=1, out=[["a"], ["b"], ["c"]]) == (
        [["", "x", "$"], ["y", "$"], ["", "$"]], [["a"], ["b", "y"], ["c"]])

    assert move_sc(sclistlist=[["", "$"], ["", "$"], ["Z", "2", "$"]],
                   whichsound=2, out=[["o"], ["r"], ["f"]]) == (
        [["", "$"], ["", "$"], ["2", "$"]], [["o"], ["r"], ["f", "2"]])


def test_get_diff():
    """test if the difference is calculated correctly
    between the first two sound of a sound correspondence list"""

    # test without exception
    # set up: mock class, 2 attributes, 1 var for input-param
    monkey_adrc = AdrcMonkey()
    monkey_adrc.sc[1] = {"k k": 2, "k c": 1, "i e": 2, "i o": 1}
    sclistlist = [["k", "c", "$"], ["e", "o", "$"],
                  ["k", "c", "$"], ["e", "o", "$"]]
    # assert
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i", "k", "i"]) == [1, 1, 1, 1]
    # there were no mock calls, so no calls to assert

    # test first exception
    # set up: mock class, 2 attributes, 1 var for input-param
    monkey_adrc = AdrcMonkey()
    monkey_adrc.sc[1] = {"k k": 2, "k c": 1, "i e": 2, "i o": 1}
    sclistlist = [["k", "c", "x"], ["x", "$"]]
    # assert
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i"]) == [1, float("inf")]
    # no calls made so no calls to assert

    # test second exception
    # set up: mock class, 2 attributes, 1 var for input-param
    monkey_adrc = AdrcMonkey()
    monkey_adrc.sc[1] = {"k k": 0, "k c": 0, "i e": 7, "i o": 1}
    sclistlist = [["k", "c", "x"], ["e", "o", "x"]]

    # assert 1
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i"]) == [9999999, 6]

    # teardown/setup: overwrite attribute nsedict
    monkey_adrc.sc[1] = {"k k": 0, "k c": 0, "i e": 7, "i o": 7}

    # assert 2
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i"]) == [9999999, 0]

    # teardown/setup: overwrite attribute nsedict
    monkey_adrc.sc[1] = {"k k": 0, "k c": 0, "i e": 0, "i o": 0}

    # assert 3
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i"]) == [9999999, 9999999]

    # no calls made so no calls to assert

    # tear down
    del monkey_adrc, sclistlist


def test_read_sc():
    """test if sound correspondences are read correctly"""

    # set up mock class
    class AdrcMonkeyread_sc:
        def __init__(self, get_diff=""):
            self.get_diff_returns = iter(get_diff)
            self.get_diff_called_with = []
            self.sc = [{},{},{},{},{},{}]

        def get_diff(self, sclistlist, ipa):
            self.get_diff_called_with.append((sclistlist, ipa))
            return next(self.get_diff_returns)

    # test if first break works (max)

    # set up mock class, plug in mock sc[0], mock tokenise, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc()
    monkey_adrc.sc[0] = {"k": ["k", "h"], "i": ["e", "o"]}
    with patch("loanpy.scapplier.prod") as prod_mock:
        prod_mock.return_value = 16

        # assert
        assert Adrc.read_sc(self=monkey_adrc, ipa=["k", "i", "k", "i"],
                            howmany=1000) == [
            ["k", "h"], ["e", "o"], ["k", "h"], ["e", "o"]]

    # assert 3 calls: get_diff, prod_mock, tokenise
    assert monkey_adrc.get_diff_called_with == []  # not called!
    prod_mock.assert_called_with([2, 2, 2, 2])

    # test if second break works (min)

    # set up mock class, plug in mock sc[0], mock math.prod
    monkey_adrc = AdrcMonkeyread_sc()
    monkey_adrc.sc[0] = {"k": ["k", "h"], "i": ["e", "o"]}
    with patch("loanpy.scapplier.prod", side_effect=[16, 1]) as prod_mock:

        # assert read_sc works with tokenised list as input
        assert Adrc.read_sc(
            self=monkey_adrc,
            ipa=["k", "i", "k", "i"],
            howmany=1) == [["k"], ["e"], ["k"], ["e"]]

    # assert 2 calls: get_diff, prod_mock
    assert monkey_adrc.get_diff_called_with == []  # not called!
    assert prod_mock.call_args_list == [
        call([2, 2, 2, 2]), call([1, 1, 1, 1])]

    # test while loop with 1 minimum

    # set up mock class, plug in mock sc[0], mock move_sc, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc(get_diff=[[4, 5]])
    monkey_adrc.sc[0] = {"k": ["k", "h"], "i": ["e", "o"]}
    with patch("loanpy.scapplier.move_sc") as move_sc_mock:
        move_sc_mock.return_value = ([["$"], ["o", "$"]], [["k", "h"], ["e"]])
        with patch("loanpy.scapplier.prod", side_effect=[4, 1, 2]) as prod_mock:

            # assert sound correspondences are read in correctly
            assert Adrc.read_sc(self=monkey_adrc, ipa=["k", "i"],
                                howmany=2) == [
                ["k", "h"], ["e"]]

    # assert 3 calls: prod_mock, get_diff, move_sc
    assert prod_mock.call_args_list == [
        call([2, 2]), call([1, 1]), call([2, 1])]
    assert monkey_adrc.get_diff_called_with == [
        ([["k", "h", "$"], ["e", "o", "$"]], ["k", "i"])]
    move_sc_mock.assert_called_with(
        [["k", "h", "$"], ["e", "o", "$"]], 0, [["k"], ["e"]])

    # test while loop with 2 minima (🚲)

    # set up mock class, plug in mock sc[0],
    # def var for side_effect of move_sc, mock move_sc, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc(get_diff=[[2, 2, 5], [2, 2, 5], [3, 2, 5]])
    monkey_adrc.sc[0] = {"k": ["k", "h", "s"], "i": ["e", "o"],
                          "p": ["b", "v"]}
    se_move_sc = [
        ([["s", "$"], ["o", "$"], ["v", "$"]], [["k", "h"], ["e"], ["b"]]),
        ([["s", "$"], ["$"], ["v", "$"]], [["k", "h"], ["e", "o"], ["b"]])
    ]
    with patch("loanpy.scapplier.move_sc", side_effect=se_move_sc) as move_sc_mock:
        with patch("loanpy.scapplier.prod", side_effect=[
                12, 1, 1, 2, 4]) as prod_mock:
            # prod "3" gets only called once, bc difflist1!=difflist2, so 2nd
            # while loop doesnt call prod

            # assert read_sc works
            assert Adrc.read_sc(self=monkey_adrc, ipa=["k", "i", "p"],
                                howmany=3) == [["k", "h"], ["e", "o"], ["b"]]

    # assert calls: prod_mock, get_diff, move_sc
    assert prod_mock.call_args_list == [call([3, 2, 2]), call(
        [1, 1, 1]), call([1, 1, 1]), call([2, 1, 1]), call([2, 2, 1])]
    assert monkey_adrc.get_diff_called_with == [
        ([["k", "h", "s", "$"], ["e", "o", "$"], ["b", "v", "$"]],
         ["k", "i", "p"]),
        ([["s", "$"], ["o", "$"], ["v", "$"]], ["k", "i", "p"]),
        ([["s", "$"], ["$"], ["v", "$"]], ["k", "i", "p"])]
    assert move_sc_mock.call_args_list == [
        call([["k", "h", "s", "$"], ["e", "o", "$"], ["b", "v", "$"]],
             0, [["k"], ["e"], ["b"]]),
        call([["s", "$"], ["o", "$"], ["v", "$"]],
             1, [["k", "h"], ["e"], ["b"]])]

    # tear down
    del AdrcMonkeyread_sc, monkey_adrc, se_move_sc

@patch("loanpy.scapplier.list2regex", side_effect=["(k)", "(i)", "(h)", "(e)"])
def test_reconstruct1(list2regex_mock):
    """test first break: some sounds are not in sc[0]"""

    # set up mock class, plug in mock sc[0], mock clusterise
    monkey_adrc = AdrcMonkey()
    monkey_adrc.sc[0] = {"#-": ["-"], "#k": ["k", "h"], "k": ["h"],
                          "-#": ["-"]}

    # assert reconstruct works
    assert Adrc.reconstruct(
        self=monkey_adrc,
        ipastr="k i k i") == "i not old"

# set up mock class, will be used multiple times throughout this test
class AdrcMonkeyReconstruct:
    def __init__(
        self,
        read_sc_returns):
        self.read_sc_called_with = []
        self.read_sc_returns = read_sc_returns
        self.prosodic_inventory = []
        self.sc = [{},{},{},{},{},{}]

    def read_sc(self, ipalist, howmany):
        self.read_sc_called_with.append((ipalist, howmany))
        return self.read_sc_returns

@patch("loanpy.scapplier.list2regex", side_effect=["(k)", "(i)", "(h)", "(e)"])
def test_reconstruct2(list2regex_mock):
    """Test if reconstructions with howmany=1 work fine"""

    # set up: create instance of mock class
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k"], ["i"], ["h"], ["e"], ["-"]])

    # set up: plug in sound correspondence dictionary into mock class
    monkey_adrc.sc[0] = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "e"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }
    # assert reconstruct works
    assert Adrc.reconstruct(
        self=monkey_adrc,
        ipastr="k i k i",
        ) == "^(k)(i)(h)(e)$"

    # assert 3 calls: tokenise, read_sc, list2regex
    assert monkey_adrc.read_sc_called_with == [
        (['k', 'i', 'k', 'i',], 1)]
    assert list2regex_mock.call_args_list == [
        call(["k"]), call(["i"]),
        call(["h"]), call(["e"])]

@patch("loanpy.scapplier.list2regex", side_effect=["(k|h)", "(i)", "(h)", "(e)"])
def test_reconstruct3(list2regex_mock):
    """Test if reconstructions with howmany=2 work fine"""

    # set up
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i"], ["h"], ["e"], ["-"]])
    monkey_adrc.sc[0] = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "e"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }

    # assert reconstruct works
    assert Adrc.reconstruct(
        self=monkey_adrc,
        ipastr="k i k i",
        howmany=2) == "^(k|h)(i)(h)(e)$"

    # assert 3 calls: clusterise, read_sc, list2regex
    assert monkey_adrc.read_sc_called_with == [
        (['k', 'i', 'k', 'i',], 2)]
    assert list2regex_mock.call_args_list == [
        call(["k", "h"]), call(["i"]),
        call(["h"]), call(["e"])]

@patch("loanpy.scapplier.get_mtx")
@patch("loanpy.scapplier.mtx2graph")
@patch("loanpy.scapplier.dijkstra")
@patch("loanpy.scapplier.tuples2editops")
@patch("loanpy.scapplier.apply_edit")
def test_repair_phonotactics1(apply_edit_mock, tuples2editops_mock,
    dijkstra_mock, mtx2graph_mock, get_mtx_mock):
    """
    test if phonotactic structures are adapted correctly
    when no data available
    """

    # test all in dict
    # set up mock class, used multiple times throughout this test
    class AdrcMonkeyrepair_phonotactics:
        def __init__(self):
            self.get_closest_phonotactics_returns = "V"
            self.get_closest_phonotactics_called_with = []
            self.sc = [{}, {}, {}, {}, {}, {}]

        def get_closest_phonotactics(self, *args):
            self.get_closest_phonotactics_called_with.append([*args])
            return self.get_closest_phonotactics_returns
    # teardown/setup: overwrite mock class, plug in sc[3],
    monkey_adrc = AdrcMonkeyrepair_phonotactics()

    get_mtx_mock.return_value = [[0, 0], [0, 1]]
    mtx2graph_mock.return_value = {
        (0, 0): {(0, 1): 100, (1, 0): 49},
        (0, 1): {(1, 1): 49},
        (1, 0): {(1, 1): 100},
        (1, 1): {}
        }
    dijkstra_mock.return_value = [(0, 0), (1, 0), (1, 1)]
    tuples2editops_mock.return_value = ['substitute C by V']
    apply_edit_mock.return_value = "V"

    # assert repair_phonotactics is working
    assert Adrc.repair_phonotactics(
        self=monkey_adrc,
        ipalist="k",
        prosody="C") == 'V'

    # dijkstra, apply_edit
    assert monkey_adrc.get_closest_phonotactics_called_with == [['C']]
    get_mtx_mock.assert_called_with("C", "V")
    mtx2graph_mock.assert_called_with(get_mtx_mock.return_value)
    dijkstra_mock.assert_called_with(graph=mtx2graph_mock.return_value,
                                     start=(0, 0), end=(1, 1)
                                     )
    tuples2editops_mock.assert_called_with(dijkstra_mock.return_value,
                                           "C", "V")
    apply_edit_mock.assert_called_with("k", tuples2editops_mock.return_value)

@patch("loanpy.scapplier.get_mtx")
@patch("loanpy.scapplier.mtx2graph")
@patch("loanpy.scapplier.dijkstra")
@patch("loanpy.scapplier.tuples2editops")
@patch("loanpy.scapplier.apply_edit")
def test_repair_phonotactics2(apply_edit_mock, tuples2editops_mock,
    dijkstra_mock, mtx2graph_mock, get_mtx_mock):
    """
    test if phonotactic structures are adapted correctly
    when data is available
    """

    # test all in dict
    # set up mock class, used multiple times throughout this test
    class AdrcMonkeyrepair_phonotactics:
        def __init__(self):
            self.sc = [{}, {}, {}, {}, {}, {}]
            self.prosodic_inventory =[]

    # teardown/setup: overwrite mock class, plug in sc[3],
    monkey_adrc = AdrcMonkeyrepair_phonotactics()
    monkey_adrc.sc[3] = {"C": ["V", "CV"]}

    get_mtx_mock.return_value = [[0, 0], [0, 1]]
    mtx2graph_mock.return_value = {
        (0, 0): {(0, 1): 100, (1, 0): 49},
        (0, 1): {(1, 1): 49},
        (1, 0): {(1, 1): 100},
        (1, 1): {}
        }
    dijkstra_mock.return_value = [(0, 0), (1, 0), (1, 1)]
    tuples2editops_mock.return_value = ['substitute C by V']
    apply_edit_mock.return_value = "V"

    # assert repair_phonotactics is working
    assert Adrc.repair_phonotactics(
        self=monkey_adrc,
        ipalist="k",
        prosody="C") == 'V'

    # dijkstra, apply_edit
    get_mtx_mock.assert_called_with("C", "V")
    mtx2graph_mock.assert_called_with(get_mtx_mock.return_value)
    dijkstra_mock.assert_called_with(graph=mtx2graph_mock.return_value,
                                     start=(0, 0), end=(1, 1)
                                     )
    tuples2editops_mock.assert_called_with(dijkstra_mock.return_value,
                                           "C", "V")
    apply_edit_mock.assert_called_with("k", tuples2editops_mock.return_value)

# set up mock class, used multiple times throughout this test.
class AdrcMonkeyAdapt:
    def __init__(self, read_screturns=[
            [["k", "h"], ["e", "o"], ["k"], ["e"]]],
            combineipalistsreturns=[
            "kek", "kok", "hek", "hok",
            "ketke", "kotke", "hetke", "hotke"]):
        self.repair_phonotactics_called_with = None
        self.read_sc_returns = iter(read_screturns)
        self.read_sc_called_with = []
        self.prosodic_inventory = ["CVCCV"]

    def repair_phonotactics(self, *args):
        self.repair_phonotactics_called_with = [*args]
        return 'k i C k i'

    def read_sc(self, *args):
        self.read_sc_called_with.append([*args])
        return next(self.read_sc_returns)

    # create instance of mock class

@patch("loanpy.scapplier.product")
def test_adapt1(product_mock):
    """test if words are adapted correctly without prosody, howmany=4"""

    adrc_monkey = AdrcMonkeyAdapt()
    product_mock.return_value = [
        ("k", "e", "t", "e"), ("k", "o", "t", "e"),
        ("h", "e", "t", "e"), ("h", "o", "t", "e")
                                         ]
    # assert adapt is working
    assert Adrc.adapt(
        self=adrc_monkey,
        ipastr="k i k i",
        howmany=4
        ) == ["kete", "kote", "hete", "hote"]

    assert not adrc_monkey.repair_phonotactics_called_with
    assert adrc_monkey.read_sc_called_with == [[["k", "i", "k", "i"], 4]]
    product_mock.assert_called_with(
            ["k", "h"], ["e", "o"], ["k"], ["e"]
                                   )

@patch("loanpy.scapplier.product")
def test_adapt2(product_mock):
    """test if words are adapted correctly with prosody, howmany=8"""

    adrc_monkey = AdrcMonkeyAdapt()
    product_mock.return_value = [
        ("k", "e", "k"), ("k", "o", "k"), ("h", "e", "k"), ("h", "o", "k"),
        ("k", "e", "t", "k", "e"), ("k", "o", "t", "k", "e"),
        ("h", "e", "t", "k", "e"), ("h", "o", "t", "k", "e")]

    # assert adapt is working
    assert Adrc.adapt(
        self=adrc_monkey,
        ipastr="k i k i",
        prosody="CVCV",
        howmany=8
        ) == ["kek", "kok", "hek", "hok", "ketke", "kotke", "hetke", "hotke"]

class TestRankClosestPhonotactics:
    @pytest.fixture
    def adrc_instance(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "prosodic_inventory.json"
            with open(temp_path, "w+", encoding='utf-8') as f:
                f.write(json.dumps(["CVCV", "CVVC", "VCVC",
                "CCVV", "CVC", "CVV", "VCV", "CV"]))
            yield Adrc(prosodic_inventory=temp_path)

    @patch('loanpy.scapplier.edit_distance_with2ops')
    @patch('loanpy.scapplier.min')
    def test_get_closest_phonotactics_all(self,
            mock_min, mock_edit_distance, adrc_instance):
        mock_edit_distance.side_effect = [0, 1, 1, 2, 2, 2, 2, 2]
        mock_min.return_value = (0, "CVCV")
        result = adrc_instance.get_closest_phonotactics("CVCV")
        assert result == "CVCV"

        calls = [call("CVCV", i) for i in adrc_instance.prosodic_inventory]
        mock_edit_distance.assert_has_calls(calls)
        mock_min.assert_called_once_with([(0, 'CVCV'),
        (1, 'CVVC'), (1, 'VCVC'), (2, 'CCVV'), (2, 'CVC'), (2, 'CVV'),
        (2, 'VCV'), (2, 'CV')])


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
    assert edit_distance_with2ops("Beregszász", "Kiiv") == 1196
    assert edit_distance_with2ops("Kiiv", "Moszkva") == 594
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

def test_list2regex():
    """test if list of phonemes is correctly converted to regular expression"""
    assert list2regex(["b", "k", "v"]) == "(b|k|v)"
    assert list2regex(["b", "k", "-", "v"]) == "(b|k|v)?"
    assert list2regex(["b", "k", "-", "v", "mp"]) == "(b|k|v|mp)?"
    assert list2regex(["b", "k", "-", "v", "mp", "mk"]) == "(b|k|v|mp|mk)?"
    assert list2regex(["o"]) == '(o)'
    assert list2regex(["ʃʲk"]) == '(ʃʲk)'

def test_tuples2editops():
    """assert that edit operations coded as tuples
    are converted to natural language correctly"""


    # assert list of tuples is correctly converted to list of strings
    assert tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)],
        "ló", "hó") == ['substitute l by h', 'keep ó']
    assert tuples2editops([(0, 0), (1, 1), (2, 2), (2, 3)],
        "lóh", "ló") == ['keep l', 'keep ó', 'delete h']
    assert tuples2editops([(0, 0), (1, 1), (2, 2), (3, 3)],
        "foo", "foo") == ['keep f', 'keep o', 'keep o']

def test_get_mtx():
    """test if distance matrix between two words is set up correctly"""

    exp = [[0, 1, 2, 3, 4],
           [1, 2, 3, 4, 5],
           [2, 3, 2, 3, 4],
           [3, 4, 3, 2, 3],
           [4, 5, 4, 3, 2]
          ]

        # assert
    assert get_mtx("Bécs", "Pécs") == exp

    # tear down
    del exp

def test_mtx2graph():
    expected = {(0, 0): {(0, 1): 100, (1, 0): 49},
                (0, 1): {(0, 2): 100, (1, 1): 49},
                (0, 2): {(1, 2): 49},
                (1, 0): {(1, 1): 100, (2, 0): 49},
                (1, 1): {(1, 2): 100, (2, 1): 49, (2, 2): 0},
                (1, 2): {(2, 2): 49},
                (2, 0): {(2, 1): 100},
                (2, 1): {(2, 2): 100},
                (2, 2): {}}

    # "ló", "hó"
    assert mtx2graph([[0, 1, 2], [1, 2, 3], [2, 3, 2]]) == expected

def test_dijkstra():
    # Test 1: Basic graph with a simple shortest path
    graph1 = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 6},
        'C': {'D': 3},
        'D': {}
    }
    assert dijkstra(graph1, 'A', 'D') == ['A', 'B', 'C', 'D']

    # Test 3: Graph with multiple paths to the destination
    graph3 = {
        'A': {'B': 5, 'C': 1},
        'B': {'D': 2},
        'C': {'D': 6},
        'D': {}
    }
    assert dijkstra(graph3, 'A', 'D') == ['A', 'C', 'D']

    # Test 6: Graph with disconnected components
    graph6 = {
        'A': {'B': 1},
        'B': {'C': 2},
        'C': {},
        'D': {'E': 3},
        'E': {'F': 4},
        'F': {}
    }
    assert dijkstra(graph6, 'A', 'F') == (None)

def test_add_edge_new_node():
    graph = {}
    add_edge(graph, 'A', 'B', 5)
    assert graph == {'A': {'B': 5}}


def test_add_edge_existing_node():
    graph = {'A': {'B': 3}}
    add_edge(graph, 'A', 'C', 7)
    assert graph == {'A': {'B': 3, 'C': 7}}


def test_add_edge_overwrite_edge():
    graph = {'A': {'B': 3}}
    add_edge(graph, 'A', 'B', 5)
    assert graph == {'A': {'B': 5}}


def test_add_edge_self_loop():
    graph = {'A': {}}
    add_edge(graph, 'A', 'A', 4)
    assert graph == {'A': {'A': 4}}


def test_add_edge_negative_weight():
    graph = {'A': {}}
    add_edge(graph, 'A', 'B', -2)
    assert graph == {'A': {'B': -2}}


@pytest.mark.parametrize("graph, u, v, weight, expected", [
    ({}, 'A', 'B', 5, {'A': {'B': 5}}),
    ({'A': {'B': 3}}, 'A', 'C', 7, {'A': {'B': 3, 'C': 7}}),
    ({'A': {'B': 3}}, 'A', 'B', 5, {'A': {'B': 5}}),
    ({'A': {}}, 'A', 'A', 4, {'A': {'A': 4}}),
    ({'A': {}}, 'A', 'B', -2, {'A': {'B': -2}}),
])
def test_add_edge_parametrized(graph, u, v, weight, expected):
    add_edge(graph, u, v, weight)
    assert graph == expected


def test_substitute_operations_empty():
    operations = []
    result = substitute_operations(operations)
    assert result == []


def test_substitute_operations_no_substitution():
    operations = ['insert A', 'delete B', 'insert C']
    result = substitute_operations(operations)
    assert result == ['substitute B by A', 'insert C']


def test_substitute_operations_single_substitution():
    operations = ['delete A', 'insert B']
    result = substitute_operations(operations)
    assert result == ['substitute A by B']


def test_substitute_operations_multiple_substitutions():
    operations = ['delete A', 'insert B', 'delete C', 'insert D']
    result = substitute_operations(operations)
    assert result == ['substitute A by B', 'substitute C by D']


def test_substitute_operations_mixed_operations():
    operations = ['insert A', 'delete B', 'insert C', 'delete D', 'insert E']
    result = substitute_operations(operations)
    assert result == ['substitute B by A', 'substitute D by C', 'insert E']
