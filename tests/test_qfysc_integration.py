"""integration test for loanpy.qfysc.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from inspect import ismethod
from os import remove
from pathlib import Path

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from loanpy.qfysc import Qfy, read_scdictbase
from loanpy.helpers import Etym

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"


def test_read_mode():
    pass  # no patches in unittest (equal integration test)


def test_read_connector():
    pass  # no patches in unittest (equal integration test)


def test_read_scdictbase():
    """test if scdictbase is generated correctly from ipa_all.csv"""

    # setup
    base = {"a": ["e", "o"], "b": ["p", "v"]}
    path = Path(__file__).parent / "test_read_scdictbase.txt"
    with open(path, "w") as f:
        f.write(str(base))

    # assert
    assert read_scdictbase(base) == base
    assert read_scdictbase(path) == base

    # tear down
    remove(path)
    del base, path


def test_init():
    """test if loanpy.qfy.Qfy is initiated correctly"""
    qfy = Qfy()

    # assert number of attributes (super() + rest)
    assert len(qfy.__dict__) == 12

    # 8 attributes inherited from Etym
    assert isinstance(qfy.phon2cv, dict)
    assert len(qfy.phon2cv) == 6358
    assert isinstance(qfy.vow2fb, dict)
    assert len(qfy.vow2fb) == 1240
    assert qfy.dfety is None
    assert qfy.phoneme_inventory is None
    assert qfy.cluster_inventory is None
    assert qfy.phonotactic_inventory is None
    ismethod(qfy.distance_measure)
    assert qfy.forms_target_language is None

    # 4 attributes initiated in Qfy
    assert qfy.mode == "adapt"
    assert qfy.connector == "<"
    assert qfy.scdictbase == {}
    assert qfy.vfb is None

    del qfy


def test_align():
    """test if 2 strings are aligned correctly"""
    # set up
    qfy = Qfy()
    # assert
    assert_frame_equal(qfy.align(left="kala", right="hal"), DataFrame(
        {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}))
    # overwrite
    qfy = Qfy(mode="reconstruct")
    # assert
    assert_frame_equal(qfy.align(left="ɟɒloɡ", right="jɑlkɑ"),
                       DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                  "vals": ['-', 'j', 'ɑ', 'lk', 'ɑ', '-']}))
    # tear down
    del qfy


def test_align_lingpy():
    """check if alignments work correctly"""
    # set up
    qfy = Qfy()
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
    qfy = Qfy(mode="reconstruct")
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
    qfy = Qfy(forms_csv=PATH2FORMS, source_language="WOT",
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
    qfy = Qfy(forms_csv=PATH2FORMS, source_language="EAH",
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

    qfy = Qfy(
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
    with open(path2test_sc, "r") as f:
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

    qfy = Qfy(forms_csv=PATH2FORMS, source_language="WOT",
              target_language="EAH")
    path2test_sc = Path(__file__).parent / "test_sc.txt"

    # assert return value is as expected. lists in 1st dict are random so set.
    out = qfy.get_phonotactics_corresp(write_to=path2test_sc)
    for s_out, s_exp in zip(out[0], exp[0]):
        assert set(out[0][s_out]) == set(exp[0][s_exp])
    assert out[1:] == exp[1:]

    # assert output was written correctly to file
    with open(path2test_sc, "r") as f:
        out = literal_eval(f.read())
    for s_out, s_exp in zip(out[0], exp[0]):
        assert set(out[0][s_out]) == set(exp[0][s_exp])
    assert out[1:] == exp[1:]

    remove(path2test_sc)
    del out, path2test_sc, qfy

    #
