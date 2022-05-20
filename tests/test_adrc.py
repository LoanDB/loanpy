from collections import OrderedDict
from os import remove
from pathlib import Path
from unittest.mock import patch, call

from pandas import DataFrame

from loanpy.adrc import Adrc, read_scdictlist, move_sc
from loanpy.qfysc import Qfy


class AdrcMonkey:
    pass


def test_read_scdictlist():
    """test if list of sound correspondence dictionaries is read correctly"""

    # no set up needed for this assertion:
    assert read_scdictlist(None) == [None, None, None, None]

    # set up: creat a mock list of dicts and write it to file
    dict0 = {"dict0": "szia"}
    dict1 = {"dict1": "cső"}
    out = [dict0, dict1, dict0, dict1]
    path = Path(__file__).parent / "test_read_scdictlist.txt"
    with open(path, "w") as f:
        f.write(str(out))

    # set up: mock literal_eval
    with patch("loanpy.adrc.literal_eval") as literal_eval_mock:
        literal_eval_mock.return_value = out

        # assert mock dict list is read in correctly
        assert read_scdictlist(path) == out

        #assert call
        literal_eval_mock.assert_called_with(str(out))

    # tear down
    remove(path)
    del dict0, dict1, out, path


def test_move_sc():
    """test if sound correspondences are moved correctly"""
    # no setup, teardown, or patch needed here
    assert move_sc(sclistlist=[["x", "x"], ["y", "y"], ["z"]], whichsound=1,
                   out=[["a"], ["b"], ["c"]]) == ([["x", "x"], ["y"], ["z"]],
                                                  [["a"], ["b", "y"], ["c"]])
    assert move_sc(sclistlist=[["x", "x"], ["y", "y"], ["z"]], whichsound=0,
                   out=[["a"], ["b"], ["c"]]) == ([["x"], ["y", "y"], ["z"]],
                                                  [["a", "x"], ["b"], ["c"]])


def test_init():
    """test if the Adrc-class is initiated properly"""

    # set up: mock super() method and read_scdictlist
    # define side_effect of read_scdictlist in vars
    # initiate the real class
    with patch("loanpy.adrc.Qfy.__init__") as super_method_mock:
        with patch("loanpy.adrc.read_scdictlist") as read_scdictlist_mock:
            read_scdictlist_mock.return_value = [None]*4
            monkey_adrc = Adrc()

            # assert initiation went correctly
            assert monkey_adrc.scdict is None
            assert monkey_adrc.sedict is None
            assert monkey_adrc.edict is None
            assert monkey_adrc.scdict_struc is None
            assert monkey_adrc.workflow == OrderedDict()

            # double check with __dict__
            assert len(monkey_adrc.__dict__) == 5
            assert monkey_adrc.__dict__ == {
                'edict': None,
                'scdict': None,
                'scdict_struc': None,
                'sedict': None,
                'workflow': OrderedDict()}

    #assert calls
    super_method_mock.assert_called_with(formscsv=None, srclg=None, tgtlg=None,
    mode=None, struc_most_frequent=9999999, struc_inv=None, scdictbase=None)
    assert read_scdictlist_mock.call_args_list == [call(None)]

    # set up: mock super() method and read_scdictlist
    # define side_effect of read_scdictlist in vars
    # initiate the real class
    with patch("loanpy.adrc.Qfy.__init__") as super_method_mock:
        d0, d1, d2, d3 = {"d0": None}, {"d1": None}, {
                              "d2": None}, {"d3": None}

        with patch("loanpy.adrc.read_scdictlist") as read_scdictlist_mock:
            read_scdictlist_mock.return_value = [d0, d1, d2, d3]
            monkey_adrc = Adrc(
                scdictlist="soundchanges.txt",
                formscsv="forms.csv",
                mode="reconstruct",
                struc_most_frequent=2)

            # assert initiation went correctly
            assert monkey_adrc.scdict == d0
            assert monkey_adrc.sedict == d1
            assert monkey_adrc.edict == d2
            assert monkey_adrc.scdict_struc == d3
            assert monkey_adrc.workflow == OrderedDict()

            # double check with __dict__
            assert len(monkey_adrc.__dict__) == 5
            assert monkey_adrc.__dict__ == {
                'edict': {'d2': None},
                'scdict': {'d0': None},
                'scdict_struc': {'d3': None},
                'sedict': {'d1': None},
                'workflow': OrderedDict()}

    #assert calls
    super_method_mock.assert_called_with(
        formscsv="forms.csv", srclg=None, tgtlg=None,
        mode="reconstruct",
        struc_most_frequent=2,
        struc_inv=None,
        scdictbase=None)
    assert read_scdictlist_mock.call_args_list == [call("soundchanges.txt")]

    # tear down
    del d0, d1, d2, d3, monkey_adrc


def test_get_diff():
    """test if the difference is calculated correctly
    between the first two sound of a sound correspondence list"""

    # test without exception
    # set up: mock class, 2 attributes, 1 var for input-param
    monkey_adrc = AdrcMonkey()
    monkey_adrc.sedict = {"k<k": 2, "k<c": 1, "i<e": 2, "i<o": 1}
    monkey_adrc.connector = "<"
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
    monkey_adrc.sedict = {"k<k": 2, "k<c": 1, "i<e": 2, "i<o": 1}
    monkey_adrc.connector = "<"
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
    monkey_adrc.sedict = {"k<k": 0, "k<c": 0, "i<e": 7, "i<o": 1}
    monkey_adrc.connector = "<"
    sclistlist = [["k", "c", "x"], ["e", "o", "x"]]

    #assert 1
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i"]) == [9999999, 6]

    # teardown/setup: overwrite attribute nsedict
    monkey_adrc.sedict = {"k<k": 0, "k<c": 0, "i<e": 7, "i<o": 7}

    #assert 2
    assert Adrc.get_diff(
        self=monkey_adrc,
        sclistlist=sclistlist,
        ipa=["k", "i"]) == [9999999, 0]

    # teardown/setup: overwrite attribute nsedict
    monkey_adrc.sedict = {"k<k": 0, "k<c": 0, "i<e": 0, "i<o": 0}

    #assert 3
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

        def get_diff(self, sclistlist, ipa):
            self.get_diff_called_with.append((sclistlist, ipa))
            return next(self.get_diff_returns)

    # test if first break works (max)

    # set up mock class, plug in mock scdict, mock tokenise, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc()
    monkey_adrc.scdict = {"k": ["k", "h"], "i": ["e", "o"]}
    with patch("loanpy.adrc.prod") as prod_mock:
        prod_mock.return_value = 16

        # assert
        assert Adrc.read_sc(self=monkey_adrc, ipa=["k", "i", "k", "i"], howmany=1000) == [
            ["k", "h"], ["e", "o"], ["k", "h"], ["e", "o"]]

    # assert 3 calls: get_diff, prod_mock, tokenise
    assert monkey_adrc.get_diff_called_with == []  # not called!
    prod_mock.assert_called_with([2, 2, 2, 2])

    # test if second break works (min)

    # set up mock class, plug in mock scdict, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc()
    monkey_adrc.scdict = {"k": ["k", "h"], "i": ["e", "o"]}
    with patch("loanpy.adrc.prod", side_effect=[16, 1]) as prod_mock:

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

    # set up mock class, plug in mock scdict, mock move_sc, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc(get_diff=[[4, 5]])
    monkey_adrc.scdict = {"k": ["k", "h"], "i": ["e", "o"]}
    with patch("loanpy.adrc.move_sc") as move_sc_mock:
        move_sc_mock.return_value = ([["$"], ["o", "$"]], [["k", "h"], ["e"]])
        with patch("loanpy.adrc.prod", side_effect=[4, 1, 2]) as prod_mock:

            # assert sound correspondences are read in correctly
            assert Adrc.read_sc(self=monkey_adrc, ipa=["k", "i"], howmany=2) == [
                ["k", "h"], ["e"]]

    # assert 3 calls: prod_mock, get_diff, move_sc
    assert prod_mock.call_args_list == [
        call([2, 2]), call([1, 1]), call([2, 1])]
    assert monkey_adrc.get_diff_called_with == [
        ([["k", "h", "$"], ["e", "o", "$"]], ["k", "i"])]
    move_sc_mock.assert_called_with(
        [["k", "h", "$"], ["e", "o", "$"]], 0, [["k"], ["e"]])

    # test while loop with 2 minima (🚲)

    # set up mock class, plug in mock scdict,
    #def var for side_effect of move_sc, mock move_sc, mock math.prod
    monkey_adrc = AdrcMonkeyread_sc(get_diff=[[2, 2, 5], [2, 2, 5], [3, 2, 5]])
    monkey_adrc.scdict = {"k": ["k", "h", "s"], "i": ["e", "o"], "p": ["b", "v"]}
    se_move_sc = [
        ([["s", "$"], ["o", "$"], ["v", "$"]], [["k", "h"], ["e"], ["b"]]),
        ([["s", "$"], ["$"], ["v", "$"]], [["k", "h"], ["e", "o"], ["b"]])
    ]
    with patch("loanpy.adrc.move_sc", side_effect=se_move_sc) as move_sc_mock:
        with patch("loanpy.adrc.prod", side_effect=[
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
        ([["k", "h", "s", "$"], ["e", "o", "$"], ["b", "v", "$"]], ["k", "i", "p"]),
        ([["s", "$"], ["o", "$"], ["v", "$"]], ["k", "i", "p"]),
        ([["s", "$"], ["$"], ["v", "$"]], ["k", "i", "p"])]
    assert move_sc_mock.call_args_list == [
        call([["k", "h", "s", "$"], ["e", "o", "$"], ["b", "v", "$"]],
        0, [["k"], ["e"], ["b"]]),
        call([["s", "$"], ["o", "$"], ["v", "$"]],
        1, [["k", "h"], ["e"], ["b"]])]

    #tear down
    del AdrcMonkeyread_sc, monkey_adrc, se_move_sc

def test_reconstruct():
    """test if reconstructions based on sound correspondences work"""

    # test first break: some sounds are not in scdict

    # set up mock class, plug in mock scdict, mock clusterise
    monkey_adrc = AdrcMonkey()
    monkey_adrc.scdict = {"#-": ["-"], "#k": ["k", "h"], "k": ["h"], "-#": ["-"]}
    with patch("loanpy.adrc.clusterise") as clusterise_mock:
        clusterise_mock.return_value = list("kiki")

        # assert reconstruct works
        assert Adrc.reconstruct(
            self=monkey_adrc,
            ipastring="kiki") == "i, i# not old"

    # assert 1 call: clusterise
    clusterise_mock.assert_called_with("kiki")

    # test 2nd break: struc and vowelharmony are False

    # set up mock class, will be used multiple times throughout this test
    class AdrcMonkeyReconstruct:
        def __init__(self, read_sc_returns, harmony_returns=""):
            self.read_sc_called_with = []
            self.read_sc_returns = read_sc_returns
            self.word2struc_returns = iter(["CVCV", "CVVCV", "CVCV", "CVVCV"])
            self.word2struc_called_with = []
            self.harmony_returns = iter(harmony_returns)
            self.harmony_called_with = []
            self.get_nse_returns = iter([20, 4, 99, 17])
            self.get_nse_called_with = []

        def read_sc(self, ipalist, howmany):
            self.read_sc_called_with.append((ipalist, howmany))
            return self.read_sc_returns

        def word2struc(self, *args):
            self.word2struc_called_with.append([*args])
            return next(self.word2struc_returns)

        def harmony(self, *args):
            self.harmony_called_with.append([*args])
            return next(self.harmony_returns)

        def get_nse(self, *args):
            self.get_nse_called_with.append([*args])
            return next(self.get_nse_returns)

    # set up: create instance of mock class
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k"], ["i"], ["h"], ["e"], ["-"]])

    # set up: plug in sound correspondence dictionary into mock class
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "e"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }
    # set up: mock tokenise, mock list2regex
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = list("kiki")
        with patch("loanpy.adrc.list2regex", side_effect=[
        "", "(k)", "(i)", "(h)", "(e)", ""]) as list2regex_mock:

            # assert reconstruct works
            assert Adrc.reconstruct(
                self=monkey_adrc,
                ipastring="kiki",
                clusterised=False) == "^(k)(i)(h)(e)$"

    # assert 3 calls: tokenise, read_sc, list2regex
    tokenise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 1)]
    assert list2regex_mock.call_args_list == [
        call(["-"]), call(["k"]), call(["i"]),
        call(["h"]), call(["e"]), call(["-"])]

    # test same break again but param <howmany> is greater than 1 now

    # set up
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i"], ["h"], ["e"], ["-"]])
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "e"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }
    # mock clusterise, list2regex
    with patch("loanpy.adrc.clusterise") as clusterise_mock:
        clusterise_mock.return_value = list("kiki")
        with patch("loanpy.adrc.list2regex", side_effect=[
        "", "(k|h)", "(i)", "(h)", "(e)", ""]) as list2regex_mock:

            # assert reconstruct works
            assert Adrc.reconstruct(
                self=monkey_adrc,
                ipastring="kiki",
                howmany=2) == "^(k|h)(i)(h)(e)$"

    # assert 3 calls: clusterise, read_sc, list2regex
    clusterise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 2)]
    assert list2regex_mock.call_args_list == [
    call(["-"]), call(["k", "h"]), call(["i"]),
    call(["h"]), call(["e"]), call(["-"])]
    # test with struc=True

    # teardown/setup: overwrite mock class, plug in mock scdict,
    # plug in mock struc_inv
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i", "ei"], ["h"], ["e"], ["-"]])
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "ei"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }
    monkey_adrc.struc_inv = ["CVVCV"]

    # set up: mock tokenise
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = list("kiki")

        # assert reconstruct works
        assert Adrc.reconstruct(
            self=monkey_adrc,
            ipastring="kiki",
            howmany=3,
            struc=True,
            clusterised=False) == "^keihe$|^heihe$"

    # assert 3 calls: tokenise, read_sc, word2struc
    tokenise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 3)]
    assert monkey_adrc.word2struc_called_with == [
        ["kihe"], ["keihe"], ["hihe"], ["heihe"]]

    # test struc=True, result empty

    # teardown/setup: overwrite mock class, plug in mock scdict and struc_inv
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i", "ei"], ["h"], ["e"], ["-"]])
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "ei"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }
    monkey_adrc.struc_inv = ["CCCVVVVCV"]

    # set up: mock clusterise
    with patch("loanpy.adrc.clusterise") as clusterise_mock:
        clusterise_mock.return_value = list("kiki")

        # assert reconstruct works
        assert Adrc.reconstruct(
            self=monkey_adrc, ipastring="kiki", howmany=3, struc=True
        ) == "wrong phonotactics"

    # assert 3 calls: clusterise, read_sc, word2struc
    clusterise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 3)]
    assert monkey_adrc.word2struc_called_with == [
        ["kihe"], ["keihe"], ["hihe"], ["heihe"]]

    # test struc=True, result empty, clusterised=False

    # teardown/setup: overwrite mock class, plug in mock scdict, struc_inv
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i", "ei"], ["h"], ["e"], ["-"]])
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "ei"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }
    monkey_adrc.struc_inv = ["CCCVVVVCV"]

    # set up: mock tokenise
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = list("kiki")

        # assert reconstruct works (returns an error message as string)
        assert Adrc.reconstruct(
            self=monkey_adrc,
            ipastring="kiki",
            howmany=3,
            struc=True,
            clusterised=False) == "wrong phonotactics"

    # assert 3 calls: tokenise, read_sc, word2struc
    tokenise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 3)]
    assert monkey_adrc.word2struc_called_with == [
        ["kihe"], ["keihe"], ["hihe"], ["heihe"]]

    # test vowelharmony=True

    # teardown/setup: overwrite mock class, plug in scdict
    monkey_adrc = AdrcMonkeyReconstruct(read_sc_returns=[
        ["-"], ["k", "h"], ["i", "o"], ["h"], ["e"], ["-"]], harmony_returns=[
        True, False, True, False])

    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "o"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }

    # setup: mock tokenise
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = list("kiki")

        # assert reconstruct works
        assert Adrc.reconstruct(
            self=monkey_adrc,
            ipastring="kiki",
            howmany=3,
            vowelharmony=True,
            clusterised=False) == "^kihe$|^hihe$"

    # assert 3 calls: tokenise, read_sc, harmony
    tokenise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 3)]
    assert monkey_adrc.harmony_called_with == [
        ["kihe"], ["kohe"], ["hihe"], ["hohe"]]

    # test vowelharmony=True, result is empty

    # teardowns/setup: overwrite mock class, plug in scdict
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i", "o"], ["h"], ["e"], ["-"]],
        harmony_returns=[False, False, False, False]
    )
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "o"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }

    # setup: mock tokenise
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = list("kiki")

        # assert reconstruct works (returns error message as string)
        assert Adrc.reconstruct(
            self=monkey_adrc,
            ipastring="kiki",
            howmany=3,
            vowelharmony=True,
            clusterised=False) == "wrong vowel harmony"

    # assert 3 calls: tokenise, read_sc, harmony
    tokenise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], 3)]
    assert monkey_adrc.harmony_called_with == [
        ["kihe"], ["kohe"], ["hihe"], ["hohe"]]

    # test sort_by_nse=True

    # teardown/setup: overwrite mock class, plug in scdict
    monkey_adrc = AdrcMonkeyReconstruct(
        read_sc_returns=[["-"], ["k", "h"], ["i", "o"], ["h"], ["e"], ["-"]],
    )
    monkey_adrc.scdict = {
        "#-": ["-"], "#k": ["k", "h"], "i": ["i", "o"],
        "k": ["h"], "i#": ["e", "o"], "-#": ["-"]
    }

    # set up: mock tokenise
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = list("kiki")

        # assert reconstruct works and sorts the result by nse
        assert Adrc.reconstruct(
            self=monkey_adrc,
            ipastring="kiki",
            howmany=float("inf"),
            sort_by_nse=True,
            clusterised=False) == "^hihe$|^kihe$|^hohe$|^kohe$"

    # assert 3 calls: tokenise, read_sc, and get_nse
    tokenise_mock.assert_called_with("kiki")
    assert monkey_adrc.read_sc_called_with == [
        (['#-', '#k', 'i', 'k', 'i#', '-#'], float("inf"))]
    assert monkey_adrc.get_nse_called_with == [["kiki", "kihe"], [
        "kiki", "kohe"], ["kiki", "hihe"], ["kiki", "hohe"]]

    del monkey_adrc, AdrcMonkeyReconstruct


def test_adapt_struc():
    """test if phonotactic structures are adapted correctly"""

    # set up mock class, used multiple times throughout this test
    class AdrcMonkeyadapt_struc:
        def __init__(self):
            self.word2struc_returns = "CVCV"
            self.word2struc_called_with = []
            self.rank_closest_struc_returns = "CVC, CVCCV"
            self.rank_closest_struc_called_with = []

        def word2struc(self, *args):
            self.word2struc_called_with.append([*args])
            return self.word2struc_returns

        def rank_closest_struc(self, *args):
            self.rank_closest_struc_called_with.append([*args])
            return self.rank_closest_struc_returns

    # test first break

    # set up mock class
    monkey_adrc = AdrcMonkeyadapt_struc()
    # set up: mock tokenise
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ["k", "i", "k", "i"]

        # assert adapt_struc works
        assert Adrc.adapt_struc(self=monkey_adrc, ipa_in="kiki", max_struc=1) == [
            ["k", "i", "k", "i"]]

    # assert 3 calls: word2struc, rank_closest_struc, tokenise
    assert monkey_adrc.word2struc_called_with == []
    assert monkey_adrc.rank_closest_struc_called_with == []
    tokenise_mock.assert_called_with("kiki")

    # test all in dict (no break)

    # teardown/setup: overwrite mock class, plug in scdict_struc,
    monkey_adrc = AdrcMonkeyadapt_struc()
    monkey_adrc.scdict_struc = {"CVCV": ["CVC", "CVCCV"]}
    # set up: define vars for side_effect of mock editops
    ops1 = ("keep C", "keep V", "keep C", "delete V")
    ops2 = ("keep C", "keep V", "insert C", "keep C", "keep V")
    # set up: mock tokenise, helpers.editops, helpers.apply_edit
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ["k", "i", "k", "i"]
        with patch("loanpy.adrc.editops", side_effect=[
        [ops1], [ops2]]) as editops_mock:
            with patch("loanpy.adrc.apply_edit", side_effect=[
            "kik", "kiCki"]) as apply_edit_mock:

                #assert adapt_struc is working
                assert Adrc.adapt_struc(
                    self=monkey_adrc, ipa_in="kiki", max_struc=2) == [
                    'kik', 'kiCki']

    # assert 5 calls were made: word2struc, rank_closest_struc,
    #editops, apply_edit, tokenise
    assert monkey_adrc.word2struc_called_with == [
        [["k", "i", "k", "i"]]]
    assert monkey_adrc.rank_closest_struc_called_with == []
    editops_mock.assert_has_calls(
        [call("CVCV", "CVC", 1), call("CVCV", "CVCCV", 1)])
    apply_edit_mock.assert_has_calls(
        [call(['k', 'i', 'k', 'i'], ops1),
        call(['k', 'i', 'k', 'i'], ops2)])
    tokenise_mock.assert_called_with("kiki")

    # test struc missing from dict and rank_closest instead, test show_workflow

    # set up mock class, plug in empty dict to trigger error
    monkey_adrc = AdrcMonkeyadapt_struc()
    monkey_adrc.scdict_struc = {}
    monkey_adrc.workflow = OrderedDict()
    # set up: define side effect of mock-editops
    ops1 = ("keep C", "keep V", "keep C", "delete V")
    ops2 = ("keep C", "keep V", "insert C", "keep C", "keep V")
    # set up: mock tokenise, helpers.editops, helpers.apply_edit
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ["k", "i", "k", "i"]
        with patch("loanpy.adrc.editops", side_effect=[
        [ops1], [ops2]]) as editops_mock:
            with patch("loanpy.adrc.apply_edit", side_effect=[
            "kik", "kiCki"]) as apply_edit_mock:

                #assert adapt_struc is working
                assert Adrc.adapt_struc(
                    self=monkey_adrc,
                    ipa_in="kiki",
                    max_struc=2,
                    show_workflow=True) == ['kik', 'kiCki']

    # assert 6 calls: word2struc, rank_closest_struc, workflow,
    #editops, apply_edit, tokenise
    assert monkey_adrc.word2struc_called_with == [
        [["k", "i", "k", "i"]]]
    assert monkey_adrc.rank_closest_struc_called_with == [["CVCV", 2]]
    assert monkey_adrc.workflow == OrderedDict(
    [('donor_struc', ['CVCV']), ('pred_strucs', [['CVC', 'CVCCV']])])
    editops_mock.assert_has_calls(
        [call("CVCV", "CVC", 1), call("CVCV", "CVCCV", 1)])
    apply_edit_mock.assert_has_calls(
        [call(['k', 'i', 'k', 'i'], ops1),
         call(['k', 'i', 'k', 'i'], ops2)])
    tokenise_mock.assert_called_with("kiki")

    # tear down
    del monkey_adrc, ops1, ops2, AdrcMonkeyadapt_struc


def test_adapt():
    """test if words are adapted correctly with sound correspondence data"""

    # set up mock class, used multiple times throughout this test.
    class AdrcMonkeyAdapt:
        def __init__(self, read_screturns=[
        [["k", "h"], ["e", "o"], ["k"]],
        [["k", "h"], ["e", "o"], ["t"], ["k"], ["e"]]],
        combineipalistsreturns=[
            "kek", "kok", "hek", "hok", "ketke", "kotke", "hetke", "hotke"]):
            self.adapt_struc_called_with = None
            self.read_sc_returns = iter(read_screturns)
            self.read_sc_called_with = []
            self.workflow = OrderedDict()
            self.adapt_harmony_called_with = []
            self.struc_inv = ["CVCCV"]
            self.word2struc_called_with = []
            self.word2struc_returns = iter(["CVC"] * 6 + ["CVCCV"] * 4)
            self.clusters = ["k", "h", "tk", "o", "u", "e"]
            self.get_nse_returns = iter([5, 9])
            self.get_nse_called_with = []

        def adapt_struc(self, *args):
            self.adapt_struc_called_with = [*args]
            self.workflow["donor_struc"] = ["CVCV"]
            self.workflow["pred_strucs"] = [["CVC", "CVCCV"]]
            return [['kik'], ['kiCki']]

        def read_sc(self, *args):
            self.read_sc_called_with.append([*args])
            return next(self.read_sc_returns)

        def adapt_harmony(self, *args):
            self.adapt_harmony_called_with.append([*args])
            return [[['kBk'], ['kiCki']]]

        def word2struc(self, *args):
            self.word2struc_called_with.append([*args])
            return next(self.word2struc_returns)

        def get_nse(self, *args):
            self.get_nse_called_with.append([*args])
            return next(self.get_nse_returns)

    # basic settings

    # create instance of mock class, mock tokenise, mock get_howmany
    monkey_adrc = AdrcMonkeyAdapt()
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        with patch("loanpy.adrc.get_howmany") as get_howmany_mock:
            with patch("loanpy.adrc.combine_ipalists") as combine_ipalists_mock:
                combine_ipalists_mock.return_value = "kek", "kok", "hek", "hok", "ketke", "kotke", "hetke", "hotke"
                get_howmany_mock.return_value = (8, 1, 1)
                tokenise_mock.return_value = ["k", "i", "k", "i"]

                #assert adapt is working
                assert Adrc.adapt(self=monkey_adrc, ipa_in="kiki",
                howmany=8) == "kek, kok, hek, hok, ketke, kotke, hetke, hotke"

    # assert 7 calls: tokenise, adapt_struc, read_sc, combine_ipalists,
    #adapt_harmony, word2struc, get_howmany
    combine_ipalists_mock.assert_called_with([
        [["k", "h"], ["e", "o"], ["k"]],
        [["k", "h"], ["e", "o"], ["t"], ["k"], ["e"]]])
    tokenise_mock.assert_called_with("kiki")
    get_howmany_mock.assert_called_with(8, 1, 1)
    assert monkey_adrc.adapt_struc_called_with == [
        ["k", "i", "k", "i"], 1, 1, False]
    assert monkey_adrc.read_sc_called_with == [
    [['kik'], 8], [['kiCki'], 8]]
    assert monkey_adrc.adapt_harmony_called_with == []
    assert monkey_adrc.word2struc_called_with == []

    # advanced settings

    # teardown/setup: overwrite mock class, mock tokenise,
    # mock flatten, mock get_howmany
    monkey_adrc = AdrcMonkeyAdapt(read_screturns=[
    [["k", "h", "c"], ["o", "u"], ["k"]],
    [["k"], ["o", "u"], ["t", "d"], ["k"], ["e"]]])
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ["k", "i", "k", "i"]
        with patch("loanpy.adrc.flatten") as flatten_mock:
            flatten_mock.return_value = [list('kBk'), list('kiCki')]
            with patch("loanpy.adrc.get_howmany") as get_howmany_mock:
                get_howmany_mock.return_value = (2, 2, 2)
                with patch("loanpy.adrc.combine_ipalists") as combine_ipalists_mock:
                    combine_ipalists_mock.return_value = [
                    'kok', 'kuk', 'hok', 'huk', 'cok',
                    'cuk', 'kotke', 'kodke', 'kutke', 'kudke']

                    # assert adapt works
                    assert Adrc.adapt(
                        self=monkey_adrc,
                        ipa_in="kiki",
                        howmany=6,
                        max_struc=2,
                        max_paths=2,
                        vowelharmony=True,
                        struc_filter=True,
                        sort_by_nse=True,
                        clusterised=True,
                        show_workflow=True) == "kutke, kotke"

    # assert 8 calls: tokenise, flatten, adapt_struc, read_sc,
    #combine_ipalists, adapt_harmony, workflow, get_howmany
    tokenise_mock.assert_called_with("kiki")
    get_howmany_mock.assert_called_with(6, 2, 2)
    combine_ipalists_mock.assert_called_with([
    [["k", "h", "c"], ["o", "u"], ["k"]], [["k"],
    ["o", "u"], ["t", "d"], ["k"], ["e"]]])
    assert list(flatten_mock.call_args_list[0][0][0]) == [
        [[['kBk'], ['kiCki']]], [[['kBk'], ['kiCki']]]]
    assert monkey_adrc.adapt_struc_called_with == [
        ["k", "i", "k", "i"], 2, 2, True]
    assert monkey_adrc.read_sc_called_with == [
        [["k", "B", "k"], 2], [["k", "i", "C", "k", "i"], 2]]
    assert monkey_adrc.adapt_harmony_called_with == [
        [['kik']], [['kiCki']]]
    assert monkey_adrc.get_nse_called_with == [
    ['kiki', 'kotke'], ['kiki', 'kutke']]
    assert monkey_adrc.workflow == OrderedDict(
    [('tokenised', [['k', 'i', 'k', 'i']]),
    ('donor_struc', ['CVCV']), ('pred_strucs',
    [['CVC', 'CVCCV']]),
    ('adapted_struc', [[['kik'], ['kiCki']]]),
    ('adapted_vowelharmony',
    [[['k', 'B', 'k'], ['k', 'i', 'C', 'k', 'i']]]),
    ('before_combinatorics',
    [[[['k', 'h', 'c'], ['o', 'u'], ['k']],
    [['k'], ['o', 'u'], ['t', 'd'], ['k'], ['e']]]])])

    # struc_filter empty

    # teardown/setup: overwrite instance of mock class, plug in struc_inv,
    # mock tokenise
    monkey_adrc = AdrcMonkeyAdapt()
    monkey_adrc.struc_inv = ["CV", "VC"]
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ["k", "i", "k", "i"]
        with patch("loanpy.adrc.get_howmany") as get_howmany_mock:
            get_howmany_mock.return_value = (1, 1, 1)
            with patch("loanpy.adrc.combine_ipalists") as combine_ipalists_mock:
                combine_ipalists_mock.return_value = [
                'kok', 'kuk', 'hok', 'huk', 'cok',
                'cuk', 'kotke', 'kodke', 'kutke', 'kudke']

                # assert adapt returns error message as string
                assert Adrc.adapt(
                    self=monkey_adrc, ipa_in="kiki", struc_filter=True
                ) == "wrong phonotactics"

    # assert 4 calls: tokenise, adapt_struc,
    #read_sc and combine_ipalists
    tokenise_mock.assert_called_with("kiki")
    get_howmany_mock.assert_called_with(1, 1, 1)
    combine_ipalists_mock.assert_called_with([
        [["k", "h"], ["e", "o"], ["k"]], [["k", "h"],
        ["e", "o"], ["t"], ["k"], ["e"]]])
    assert monkey_adrc.adapt_struc_called_with == [
        ["k", "i", "k", "i"], 1, 1, False]
    assert monkey_adrc.read_sc_called_with == [
    [['kik'], 1], [['kiCki'], 1]]

    # clusterised empty filter

    # teardown/setup: overwrite instance of mock class, plug in clusters, mock
    # tokenise
    monkey_adrc = AdrcMonkeyAdapt()
    monkey_adrc.clusters = ["pr", "gr", "vrp", "skrrrr"]
    with patch("loanpy.adrc.tokenise") as tokenise_mock:
        tokenise_mock.return_value = ["k", "i", "k", "i"]
        with patch("loanpy.adrc.get_howmany") as get_howmany_mock:
            get_howmany_mock.return_value = (1, 1, 1)
            with patch("loanpy.adrc.combine_ipalists") as combine_ipalists_mock:
                combine_ipalists_mock.return_value = [
                'kok', 'kuk', 'hok', 'huk', 'cok',
                'cuk', 'kotke', 'kodke', 'kutke', 'kudke']

                # make sure adapt works (i.e. returns error message as string)
                assert Adrc.adapt(
                    self=monkey_adrc, ipa_in="kiki", clusterised=True
                ) == "wrong clusters"

    # assert 4 calls: tokenise, adapt_struc, read_sc, combine_ipalists
    tokenise_mock.assert_called_with("kiki")
    get_howmany_mock.assert_called_with(1, 1, 1)
    combine_ipalists_mock.assert_called_with([
        [["k", "h"], ["e", "o"], ["k"]], [["k", "h"],
        ["e", "o"], ["t"], ["k"], ["e"]]])
    assert monkey_adrc.adapt_struc_called_with == [
        ["k", "i", "k", "i"], 1, 1, False]
    assert monkey_adrc.read_sc_called_with == [
    [['kik'], 1], [['kiCki'], 1]]

    #tear down
    del monkey_adrc, AdrcMonkeyAdapt

def test_get_nse():
    """test if normalised sum of examples is calculated correctly"""

    # test first break, no set up needed here
    assert Adrc.get_nse(self=None, left=None, right=None) == 0

    # create mock class, used multiple times in this test
    class AdrcMonkeyget_nse:
        def __init__(self):
            self.align_called_with = []
            self.sedict = {
                "#-<*-": 10,
                "#ɟ<*j": 9,
                "ɒ<*ɑ": 8,
                "l<*lk": 7,
                "o<*ɑ": 6}
            self.edict = {"#-<*-": [1, 2], "#ɟ<*j": [3, 4],
                          "ɒ<*ɑ": [5], "l<*lk": [6, 7, 8], "o<*ɑ": [9]}
            self.connector = "<*"
            self.mode = "reconstruct"

        def align(self, *args):
            self.align_called_with.append([*args])
            return DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                              "vals": ['-', 'j', 'ɑ', 'lk', 'ɑ', '-']})

    # test default settings (normalised sum of examples)

    # set up instance of mock class
    monkey_adrc = AdrcMonkeyget_nse()
    # assert
    assert Adrc.get_nse(
        self=monkey_adrc, left="ɟɒloɡ", right="jɑlkɑ") == 6.666666666666667
    #assert call
    assert monkey_adrc.align_called_with == [["ɟɒloɡ", "jɑlkɑ"]]

    # test se=False
    # set up by overwriting mock class instance
    monkey_adrc = AdrcMonkeyget_nse()
    # assert
    assert Adrc.get_nse(self=monkey_adrc, left="ɟɒloɡ",
    right="jɑlkɑ", se=False) == [[1, 2], [3, 4], [5], [6, 7, 8], [9], 0]
    #assert call
    assert monkey_adrc.align_called_with == [["ɟɒloɡ", "jɑlkɑ"]]

    # test show_workflow=True
    # set up by overwriting mock class instance
    monkey_adrc = AdrcMonkeyget_nse()
    # assert
    assert Adrc.get_nse(
        self=monkey_adrc, left="ɟɒloɡ", right="jɑlkɑ", show_workflow=True) == (
        6.666666666666667, 40, [10, 9, 8, 7, 6, 0])
    #assert call
    assert monkey_adrc.align_called_with == [["ɟɒloɡ", "jɑlkɑ"]]

    # test se=False, show_workflow=True (same as se=False)
    # set up by overwriting mock class instance
    monkey_adrc = AdrcMonkeyget_nse()
    # assert
    assert Adrc.get_nse(self=monkey_adrc, left="ɟɒloɡ",
    right="jɑlkɑ", se=False) == [
        [1, 2], [3, 4], [5], [6, 7, 8], [9], 0]
    #assert call
    assert monkey_adrc.align_called_with == [["ɟɒloɡ", "jɑlkɑ"]]

    # tear down mock class instance
    del monkey_adrc, AdrcMonkeyget_nse
