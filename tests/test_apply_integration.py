"""unit tests for loanpy.apply.py 2.5 with pytest 7.1.2"""

import pytest
from loanpy.apply import Adrc

def test_adapt(tmp_path):
    sc_path = tmp_path / "sound_correspondences.json"
    sc_path.write_text('[{"d": ["d", "x"], "a": ["a", "x"]}, \
{"d d": 5, "d x": 4, "a a": 7, "a x": 1}, {}, {"CVCV": ["CVC"]}]')

    # no phonotactic repair
    adrc = Adrc(sc=sc_path)
    assert adrc.adapt("d a d a") == "dada"
    assert adrc.adapt("d a d a", 2) == "dada, xada"
    assert adrc.adapt("d a d a", 3) == "dada, daxa, xada"
    assert adrc.adapt("d a d a", 5) == "dada, daxa, dxda, dxxa, xada"
    assert adrc.adapt("d a d a", 100000) == "dada, dadx, daxa, daxx, dxda, \
dxdx, dxxa, dxxx, xada, xadx, xaxa, xaxx, xxda, xxdx, xxxa, xxxx"

    # phonotactic repair from data
    assert adrc.adapt("d a d a", 1, "CVCV") == "dada"

def test_get_diff(tmp_path):
    """test if the difference is calculated correctly
    between the first two sound of a sound correspondence list"""

    sc_path = tmp_path / "sound_correspondences.json"
    sc_path.write_text('[{"d": ["d", "x"], "a": ["a", "x"]}, \
{"d d": 5, "d x": 4, "a a": 7, "a x": 1}]')

    # create instance
    adrc_inst = Adrc(sc=sc_path)

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

    # test if first break works (max)

    # set up mock class, plug in mock sc[0], mock tokenise, mock math.prod
    adrc = Adrc()
    adrc.sc = [0, 1]
    adrc.sc[0] = {"k": ["k", "h"], "i": ["e", "o"]}
    adrc.sc[1] = {"k k": 3, "k h": 5, "i e": 2, "i o": 2}

    # assert
    assert adrc.read_sc(ipa=["k", "i", "k", "i"], howmany=1000) == [
                            ["k", "h"], ["e", "o"], ["k", "h"], ["e", "o"]
                                                                   ]

    # assert read_sc works with tokenised list as input
    assert adrc.read_sc(ipa=["k", "i", "k", "i"], howmany=1) == [
                                                    ["k"], ["e"], ["k"], ["e"]
                                                                ]
    assert adrc.read_sc(ipa=["k", "i"], howmany=2) == [["k", "h"], ["e"]]

    adrc.sc[0] = {"k": ["k", "h", "s"], "i": ["e", "o"], "p": ["b", "v"]}
    adrc.sc[1] = {"k k": 1, "k h": 2, "k s": 17, "i e": 17, "i o": 9,
             "p b": 1, "p v": 1}

    assert adrc.read_sc(ipa=["k", "i", "p"],
                        howmany=3) == [["k", "h", "s"], ["e"], ["b"]]
