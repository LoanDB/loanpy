# -*- coding: utf-8 -*-
"""integration tests for loanpy.apply.py with pytest 7.1.2"""

import pytest
from loanpy.scapplier import Adrc
from pathlib import Path

TESTFILESDIR = Path(__file__).parent / "test_files"
PATH2SC_AD = TESTFILESDIR / "sc_ad.json"
PATH2SC_RC = TESTFILESDIR / "sc_rc.json"
PATH2prosodic_inventory = TESTFILESDIR / "prosodic_inventory.json"

def test_adapt(tmp_path):
    sc_path = tmp_path / "sound_correspondences.json"
    sc_path.write_text('[{"d": ["d", "x"], "a": ["a", "x"]}, \
{"d d": 5, "d x": 4, "a a": 7, "a x": 1}, {}, {"CVCV": ["CVC"]}]')

    # no phonotactic repair
    adrc = Adrc(sc=sc_path)
    assert adrc.adapt("d a d a") == ["dada"]
    assert adrc.adapt("d a d a", 2) == ["dada", "xada"]
    assert adrc.adapt("d a d a", 3) == ["dada", "daxa", "xada"]
    assert adrc.adapt("d a d a", 5) == ["dada", "daxa", "dxda", "dxxa", "xada"]
    assert adrc.adapt("d a d a", 100000) == ["dada", "dadx", "daxa", "daxx",
        "dxda", "dxdx", "dxxa", "dxxx", "xada", "xadx", "xaxa", "xaxx",
        "xxda", "xxdx", "xxxa", "xxxx"]

    # phonotactic repair from data, 1 deletion
    assert adrc.adapt("d a d a", 1, "CVCV") == ["dad"]
    assert adrc.adapt("d a d a", 2, "CVCV") == ["dad", "xad"]
    assert adrc.adapt("d a d a", 3, "CVCV") == ["dad", "dax", "xad"]
    assert adrc.adapt("d a d a", 5, "CVCV") == ["dad", "dax", "dxd", "dxx", "xad"]
    assert adrc.adapt("d a d a", 100000, "CVCV") == ["dad", "dax", "dxd",
        "dxx", "xad", "xax", "xxd", "xxx"]


    # phonotactic repair from heuristics
    sc_path.write_text('[{"d": ["d", "x"], "a": ["a", "x"], "C": ["k"]}, \
{"d d": 5, "d x": 4, "a a": 7, "a x": 1, "C k": 8}, {}, {}]')
    prosodic_inventory_path = tmp_path / "inventories.json"
    # two insertions cheaper than two deletions, so it should pick the 2nd
    prosodic_inventory_path.write_text('["CV", "CVCVCC"]')
    adrc = Adrc(sc=sc_path, prosodic_inventory=prosodic_inventory_path)
    assert adrc.adapt("d a d a", 1, "CVCV") == ["dadakk"]
    assert adrc.adapt("d a d a", 2, "CVCV") == ["dadakk", "xadakk"]
    assert adrc.adapt("d a d a", 3, "CVCV") == ["dadakk", "daxakk", "xadakk"]
    assert adrc.adapt("d a d a", 5, "CVCV") == ["dadakk", "daxakk", "dxdakk",
                                                "dxxakk", "xadakk"]
    assert adrc.adapt("d a d a", 100000, "CVCV") == ["dadakk", "dadxkk",
        "daxakk", "daxxkk", "dxdakk", "dxdxkk", "dxxakk", "dxxxkk", "xadakk",
        "xadxkk", "xaxakk", "xaxxkk", "xxdakk", "xxdxkk", "xxxakk", "xxxxkk"]


    # try substitutions
    prosodic_inventory_path.write_text('["CCCV"]')
    adrc = Adrc(sc=sc_path, prosodic_inventory=prosodic_inventory_path)
    assert adrc.adapt("d a d a", 1, "CVCV") == ["ddka"]

def test_reconstruct():
    """test if reconstructions based on sound correspondences work"""

    # test first break: some sounds are not in scdict

    # set up adrc instance
    adrc_inst = Adrc(sc=PATH2SC_RC)

    # assert reconstruct works when sound changes are missing from data
    assert adrc_inst.reconstruct(
        ipastr="k i k i")[-7:] == "not old"

    # assert it's actually clusterising by default
    assert adrc_inst.reconstruct(
        ipastr="k.r i.e k.r i.e")[-7:] == "not old"

    # try r can be old!
    assert adrc_inst.reconstruct(
        ipastr="k r i e k r i e")[-7:] == "not old"

    # test 2nd break: phonotactics_filter and vowelharmony_filter are False
    assert adrc_inst.reconstruct(
        ipastr="#aː r uː# -#") == "^(a)(n)(a)(at͡ʃi)$"

    # test same break again but param <howmany> is greater than 1 now
    assert adrc_inst.reconstruct(
        ipastr="#aː r uː# -#", howmany=2) == "^(a)(n)(a)(at͡ʃi|γ)$"

    assert adrc_inst.reconstruct(
        ipastr="#aː r uː# -#", howmany=3) == "^(a|o)(n)(a)(at͡ʃi|γ)$"

    assert adrc_inst.reconstruct(
        ipastr="#aː r uː# -#", howmany=100) == "^(a|o)(n)(a)(at͡ʃi|γ)$"

def test_repair_phontactics():
    adrc = Adrc(sc=PATH2SC_AD, prosodic_inventory=PATH2prosodic_inventory)
    # test heuristics: keep, delete, insert
    assert adrc.repair_phonotactics(["l", "o", "l"], "CVC") == ["l", "o", "l"]
    assert adrc.repair_phonotactics(["r", "o", "f", "l"],
                                    "CVCC") == ["r", "o", "f", "l", "V"]
    assert adrc.repair_phonotactics(["b", "l", "a", "b", "l", "a"],
                                    "CCVCCV") == ["b", "a", "b", "l", "a"]
    assert adrc.repair_phonotactics(["b", "r", "b"],
                                    "CCC") == ["b", "V", "r", "b", "V"]
    assert adrc.repair_phonotactics(["a", "e", "i", "o"],
                                    "VVVV") == ["C", "a", "C", "e", "C", "i"]

    # test data
    assert adrc.repair_phonotactics(["d", "a", "d", "a"],
                                    "CVCV") == ["d", "a", "d"]
    assert adrc.repair_phonotactics(["j", "a", "j", "a", "j", "a"],
                                    "CVCVCV") == ["j", "a", "j", "a", "j", "a"]

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

    # set up mock class, plug in mock scdict, mock tokenise, mock math.prod
    adrc_inst = Adrc(sc=PATH2SC_AD)

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

def test_get_closest_phonotactics():
    """test if most similar phonotactic profiles from prosodic_inventory
    is picked correctly"""

    # assert structures are ranked correctly
    adrc = Adrc(prosodic_inventory=PATH2prosodic_inventory)
    # identical one in prosodic_inventory is the closest
    assert adrc.get_closest_phonotactics("CVC") == "CVC"
    assert adrc.get_closest_phonotactics("CVCCV") == "CVCCV"
    assert adrc.get_closest_phonotactics("CVCVCV") == "CVCVCV"

    # prefer insertion over deletion
    assert adrc.get_closest_phonotactics("CVCC") == "CVCCV"  # not CVC
    assert adrc.get_closest_phonotactics("C") == "CVC"  # 2 insertions only
    assert adrc.get_closest_phonotactics("V") == "CVC"
    assert adrc.get_closest_phonotactics("CVCCVCV") == "CVCVCV"
    assert adrc.get_closest_phonotactics("VVV") == "CVCVCV"
    assert adrc.get_closest_phonotactics("CVCVCVVVVCCCCVCVVCVCV") == "CVCVCV"

def test_move_sc():  # pragma: no cover
    pass  # unit == integration test (no patches)

def test_editdistance_with2ops():  # pragma: no cover
    pass # unit == integration test (no patches)

def test_apply_edit():  # pragma: no cover
    pass # unit == integration test (no patches)

def list2regex():  # pragma: no cover
    pass # unit == integration test (no patches)

def tuples2editops():  # pragma: no cover
    pass # unit == integration test (no patches)

def substitute_operations():  # pragma: no cover
    pass # unit == integration test (no patches)

def get_mtx():  # pragma: no cover
    pass # unit == integration test (no patches)

def add_edge():  # pragma: no cover
    pass # unit == integration test (no patches)

def mtx2graph():  # pragma: no cover
    pass # unit == integration test (no patches)

def dijkstra():  # pragma: no cover
    pass # unit == integration test (no patches) (could have patched heapq tho)
