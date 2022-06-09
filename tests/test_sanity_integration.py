"""integration test for loanpy.sanity.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from datetime import datetime
from os import remove
from pathlib import Path

from pandas import read_csv, DataFrame, RangeIndex
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import raises
from unittest.mock import call, patch

from loanpy.adrc import Adrc
from loanpy.sanity import (
    ArgumentsAlreadyTested,
    cache,
    check_cache,
    eval_adapt,
    eval_recon,
    eval_all,
    eval_one,
    get_crossval_data,
    get_dist,
    get_nse4df,
    get_noncrossval_sc,
    get_tpr_fpr_opt,
    loop_thru_data,
    make_stat,
    plot_roc,
    postprocess,
    postprocess2,
    phonotactics_predicted,
    write_to_cache)

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"
PATH2SC_AD = Path(__file__).parent / "input_files" / "sc_ad_3cogs.txt"
PATH2SC_RC = Path(__file__).parent / "input_files" / "sc_rc_3cogs.txt"
MOCK_CACHE_PATH = Path(__file__).parent / "mock_cache.csv"


def test_check_cache():
    """test if DIY cache is initiated correctly and args checked in it"""

    # make sure this file does not exist (e.g. from previous tests)
    try:
        remove(MOCK_CACHE_PATH)
    except FileNotFoundError:
        pass
    # set up first expected outcome, a pandas data frame
    exp1 = DataFrame(columns=["arg1", "arg2", "arg3", "opt_tpr",
                              "optimal_howmany", "opt_tp", "timing", "date"])

    # assert first break works: cache not found
    check_cache(MOCK_CACHE_PATH, {"arg1": "x", "arg2": "y", "arg3": "z"})
    assert_frame_equal(read_csv(MOCK_CACHE_PATH), exp1)

    # check if nothing happens if arguments were NOT tested already
    # assert that the function runs, does nothing, and returns None
    assert check_cache(MOCK_CACHE_PATH,
                       {"arg1": "a", "arg2": "b", "arg3": "c"}) is None

    # tear down
    remove(MOCK_CACHE_PATH)

    # check if exception is rased if these params were tested already

    # set up mock cache with stored args
    DataFrame({"arg1": ["x"], "arg2": ["y"], "arg3": ["z"]}).to_csv(
        MOCK_CACHE_PATH, encoding="utf-8", index=False)
    # assert exception is raised bc args exist in cache already
    with raises(ArgumentsAlreadyTested) as aat_mock:
        check_cache(MOCK_CACHE_PATH,
                    {"arg1": "x", "arg2": "y", "arg3": "z"})
    assert str(aat_mock.value) == f"These arguments were tested \
already, see {MOCK_CACHE_PATH} line 1! (start counting at 1 in 1st row)"

    # tear down
    remove(MOCK_CACHE_PATH)


def test_write_to_cache():
    """Test if the writing-part of cache functions."""
    init_args_mock = {"forms_csv": "forms.csv", "tgt_lg": "EAH",
                      "src_lg": "WOT", "crossval": True,
                      "path2cache": MOCK_CACHE_PATH,
                      "guesslist": [[2, 4, 6, 8]],
                      "max_phonotactics": 1, "max_paths": 1, "writesc": False,
                      "writesc_phonotactics": False, "vowelharmony": False,
                      "only_documented_clusters": False, "sort_by_nse": False,
                      "phonotactics_filter": False, "show_workflow": False,
                      "write": False,
                      "outname": "viz", "plot_to": None, "plotldnld": False}

    DataFrame(
        columns=list(init_args_mock) + [
            "optimal_howmany",
            "opt_tp",
            "opt_tpr",
            "timing",
            "date"]).to_csv(
        MOCK_CACHE_PATH,
        index=False,
        encoding="utf-8")  # empty cache

    df_exp = DataFrame(
        {"forms_csv": "forms.csv", "tgt_lg": "EAH",
         "src_lg": "WOT", "crossval": True,
         "path2cache": str(MOCK_CACHE_PATH), "guesslist": str([[2, 4, 6, 8]]),
         "max_phonotactics": 1, "max_paths": 1, "writesc": False,
         "writesc_phonotactics": False, "vowelharmony": False,
         "only_documented_clusters": False, "sort_by_nse": False,
         "phonotactics_filter": False, "show_workflow": False, "write": False,
         "outname": "viz", "plot_to": "None", "plotldnld": False,
         "optimal_howmany": 0.501, "opt_tp": 0.6,
         "opt_tpr": 0.099, "timing": "00:00:01",
         "date": datetime.now().strftime("%x %X")},
        index=RangeIndex(start=0, stop=1, step=1))

    # write to mock cache
    write_to_cache(
        stat=(0.501, 0.6, 0.099),
        init_args=init_args_mock,
        path2cache=MOCK_CACHE_PATH, start=1, end=2)

    # assert cache was written correctly
    assert_frame_equal(read_csv(MOCK_CACHE_PATH), df_exp)

    # assert sort functions correctly

    df_exp = DataFrame(
        {"forms_csv": ["forms.csv"] * 2, "tgt_lg": ["EAH"] * 2,
         "src_lg": ["WOT"] * 2, "crossval": [True] * 2,
         "path2cache": [str(MOCK_CACHE_PATH)] * 2,
         "guesslist": [str([[2, 4, 6, 8]])] * 2,
         "max_phonotactics": [1] * 2, "max_paths": [1] * 2,
         "writesc": [False] * 2,
         "writesc_phonotactics": [False] * 2, "vowelharmony": [False] * 2,
         "only_documented_clusters": [False] * 2, "sort_by_nse": [False] * 2,
         "phonotactics_filter": [False] * 2,
         "show_workflow": [False] * 2, "write": [False] * 2,
         "outname": ["viz"] * 2,
         "plot_to": ["None"] * 2, "plotldnld": [False] * 2,
         "optimal_howmany": [0.501] * 2, "opt_tp": [0.6, 0.6],
         "opt_tpr": [0.8, 0.099],
         "timing": ["00:00:01"] * 2,
         "date": [datetime.now().strftime("%x %X")] * 2})

    # write to mock cache
    write_to_cache(
        stat=(0.501, 0.6, 0.8),
        init_args=init_args_mock,
        path2cache=MOCK_CACHE_PATH, start=1, end=2)

    # assert cache was written and sorted correctly
    assert_frame_equal(read_csv(MOCK_CACHE_PATH), df_exp)

    remove(MOCK_CACHE_PATH)
    del df_exp, init_args_mock


def test_cache():
    """Is cache read and written to correctly?"""
    # set up mock path
    mockpath2cache = Path(__file__).parent / "mock_cache.csv"
    try:
        remove(mockpath2cache)  # in case of leftovers from previous tests
    except FileNotFoundError:
        pass
    # test cache with mockfunc

    @cache
    def mockfunc(*args, **kwargs): return "dfety", (1, 2, 3), 4, 5
    assert mockfunc(path2cache=mockpath2cache, a="hi", b="bye") is None
    # check if cache was initiated and written correctly
    assert_frame_equal(read_csv(mockpath2cache), DataFrame({
        "path2cache": [str(mockpath2cache)], "a": ["hi"], "b": ["bye"],
        "opt_tpr": [3], "optimal_howmany": [1], "opt_tp": [2],
        "timing": ["00:00:01"], "date": [datetime.now().strftime("%x %X")]
    }))

    remove(mockpath2cache)

    # todo: once integration test for eval_all works, test @cache eval_all


def test_eval_adapt():
    """Is the main function doing its job in evaluating etymological data?"""
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_AD)

    assert eval_adapt(
        "Apfel",
        adrc_obj,
        "apple",
        10,
        False,
        False,
        False,
        False,
        1,
        1,
        100,
        49,
        False) == {
        'best_guess': 'KeyError',
        'guesses': float("inf")}
    # assert with show_workflow=True
    # KeyError is triggered before 3rd part of workflow is added.
    # max_phonotactics=0, therefore adapted_phonotactics=tokenised
    assert eval_adapt("Apfel", adrc_obj, "apple",
                      10, False, False, False, False, 0, 1, 100, 49, True) == {
        "best_guess": "KeyError", "guesses": float("inf"),
        'tokenised': "['a', 'p', 'p', 'l', 'e']",
        'adapted_phonotactics': "[['a', 'p', 'p', 'l', 'e']]"}

    # assert no keyerror but target missed
    assert eval_adapt(
        "daʃa",
        adrc_obj,
        "dat͡ʃːa",
        10,
        False,
        False,
        False,
        False,
        0,
        1,
        100,
        49,
        False) == {
        'best_guess': 'dat͡ʃa',
        'guesses': float("inf")}

    # assert no keyerror but target missed while showing workflow
    assert eval_adapt("daʃa", adrc_obj, "dat͡ʃːa",
                      10, False, False, False, False, 0, 1, 100, 49, True) == {
        'best_guess': 'dat͡ʃa', 'guesses': float("inf"),
        'adapted_phonotactics': "[['d', 'a', 't͡ʃː', 'a']]",
        'before_combinatorics': "[[['d'], ['a'], ['t͡ʃ'], ['a']]]",
        'tokenised': "['d', 'a', 't͡ʃː', 'a']"}

    # no keyerror, target missed, show workflow, max_phonotactics=1
    assert eval_adapt("daʃa", adrc_obj, "aldajd",
                      10, False, False, False, False, 1, 1, 100, 49, True) == {
        'adapted_phonotactics': "[['a', 'l', 'd', 'a', 'd']]",
        'before_combinatorics': "[[['a'], ['l'], ['d'], ['a'], ['d']]]",
        'best_guess': 'aldad',
        'donor_phonotactics': 'VCCVCC',
        'guesses': float("inf"),
        'predicted_phonotactics': "['VCCVC']",
        'tokenised': "['a', 'l', 'd', 'a', 'j', 'd']"}

    # assert target hit
    assert eval_adapt(
        "dat͡ʃa",
        adrc_obj,
        "dat͡ʃːa",
        10,
        False,
        False,
        False,
        False,
        0,
        1,
        100,
        49,
        False) == {
        'best_guess': 'dat͡ʃa',
        'guesses': 1}

    # assert target hit while showing workflow, no repair_phonotactics
    assert eval_adapt("dat͡ʃa", adrc_obj, "dat͡ʃːa",
                      10, False, False, False, False, 0, 1, 100, 49, True) == {
        'best_guess': 'dat͡ʃa', 'guesses': 1,
        'adapted_phonotactics': "[['d', 'a', 't͡ʃː', 'a']]",
        'before_combinatorics': "[[['d'], ['a'], ['t͡ʃ'], ['a']]]",
        'tokenised': "['d', 'a', 't͡ʃː', 'a']"}

    # assert target hit, show workflow, max_phonotactics=1
    assert eval_adapt("aldad", adrc_obj, "aldajd",
                      10, False, False, False, False, 1, 1, 100, 49, True) == {
        'adapted_phonotactics': "[['a', 'l', 'd', 'a', 'd']]",
        'before_combinatorics': "[[['a'], ['l'], ['d'], ['a'], ['d']]]",
        'best_guess': 'aldad',
        'donor_phonotactics': 'VCCVCC',
        'guesses': 1,
        'predicted_phonotactics': "['VCCVC']",
        'tokenised': "['a', 'l', 'd', 'a', 'j', 'd']"}


def test_eval_recon():
    """Is result of loanpy.adrc.Adrc.reconstruct evaluated?"""
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="H",
                    target_language="EAH", scdictlist=PATH2SC_RC)

    # test else clause (neither short nor long regex)
    assert eval_recon("daʃa", adrc_obj, "dada") == {
        'best_guess': '#d, a, d, a# not old', 'guesses': float("inf")}
    # (no show workflow for reconstruct)
    # test short regex and target missed
    assert eval_recon("daʃa", adrc_obj, "aːruː") == {
        'best_guess': '^(a)(n)(a)(at͡ʃi)$', 'guesses': float("inf")}
    # test long regex (sort_by_nse=True, last arg) and target missed
    assert eval_recon("daʃa", adrc_obj,
                      "aːruː", 1, True, False, False, True) == {
        'best_guess': 'anaat͡ʃi', 'guesses': float("inf")}
    # test long regex, target missed, sort_by_nse=True, howmany=2
    assert eval_recon("daʃa", adrc_obj,
                      "aːruː", 2, True, False, False, True) == {
        'best_guess': 'anaɣ', 'guesses': float("inf")}
    # test long regex, target hit, sort_by_nse=True, howmany=2
    assert eval_recon("anaɣ", adrc_obj,
                      "aːruː", 2, True, False, False, True) == {
        'best_guess': 'anaɣ', 'guesses': 1}
    # test long regex, target hit, sort_by_nse=True, howmany=1
    assert eval_recon("anaat͡ʃi", adrc_obj,
                      "aːruː", 1, True, False, False, True) == {
        'best_guess': 'anaat͡ʃi', 'guesses': 1}
    # test short regex, target hit, sort_by_nse=False, howmany=2
    assert eval_recon("anaat͡ʃi", adrc_obj,
                      "aːruː", 2, True, False, False, False) == {
        'best_guess': '^(a)(n)(a)(at͡ʃi|ɣ)$', 'guesses': 2}
    # test short regex, target hit, sort_by_nse=False, howmany=2, diff target
    assert eval_recon("anaɣ", adrc_obj,
                      "aːruː", 2, True, False, False, False) == {
        'best_guess': '^(a)(n)(a)(at͡ʃi|ɣ)$', 'guesses': 2}


def test_eval_one():
    """Are eval_adapt, eval_recon called and their results evaluated?"""
    # assert None is returned if target word is not in the predictions
    # create instance of Adrc class for input
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_AD)

    # assert keyerror, mode=adapt
    assert eval_one(
        "gaga", adrc_obj, "dada",
        False, False, False, False, 0, 1, 100, 49, False, [
            2, 4, 6], "adapt") == {
        "guesses": float("inf"), "best_guess": "dada"}
    # assert no keyerror, mode=adapt, target missed
    assert eval_one(
        "gaga", adrc_obj, "dada",
        False, False, False, False, False, 0, 1, 100, 49, [
            2, 4, 6], "adapt") == {
        "guesses": float("inf"), "best_guess": "dada"}
    # assert target hit on first try, mode=adapt
    assert eval_one(
        "dada",
        adrc_obj,
        "dada",
        False,
        False,
        False,
        False,
        0,
        1,
        100,
        49,
        False,
        [1],
        "adapt") == {
        "guesses": 1,
        "best_guess": "dada"}
    # assert target hit on first try, mode=adapt, show_workflow=True
    assert eval_one(
        "dada",
        adrc_obj,
        "dada",
        False,
        False,
        False,
        False,
        0,
        1,
        100,
        49,
        True,
        [1],
        "adapt") == {
        "guesses": 1,
        "best_guess": "dada",
        'adapted_phonotactics': "[['d', 'a', 'd', 'a']]",
        'before_combinatorics': "[[['d'], ['a'], ['d'], ['a']]]",
        'tokenised': "['d', 'a', 'd', 'a']"}

    # assert reconstruct
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_RC)

    # assert keyerror, mode=reconstruct
    assert eval_one(
        "gaga", adrc_obj, "dada",
        False, False, False, False, 0, 1, 100, 49, False, [
            2, 4, 6], "reconstruct") == {
        "guesses": float("inf"), "best_guess": '#d, a, d, a# not old'}
    # assert no keyerror, mode=reconstruct, target missed
    assert eval_one(
        "gaga", adrc_obj, "aːruː",
        False, False, False, False, False, 0, 1, 100, 49, [
            2, 4, 6], "reconstruct") == {
        "guesses": float("inf"), "best_guess": "^(a)(n)(a)(at͡ʃi|ɣ)$"}
    # assert target hit on first try, mode=reconstruct
    assert eval_one(
        "anaat͡ʃi",
        adrc_obj,
        "aːruː",
        False,
        False,
        False,
        False,
        0,
        1,
        100,
        49,
        False,
        [1],
        "reconstruct") == {
        "guesses": 1,
        "best_guess": "^(a)(n)(a)(at͡ʃi)$"}


def test_get_noncrossval_sc():
    """Are non-crossvalidated sound correspondences extracted and assigned?"""
    # set up
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH")
    # run function
    adrc_obj_out = get_noncrossval_sc(adrc_obj, None)
    # assert result
    assert adrc_obj_out.scdict == {
        'a': ['a'],
        'd': ['d'],
        'j': ['j'],
        'l': ['l'],
        'n': ['n'],
        't͡ʃː': ['t͡ʃ'],
        'ɣ': ['ɣ'],
        'ɯ': ['i']}
    assert adrc_obj_out.sedict == {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1,
                                   'l<l': 1, 'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2}
    for struc, exp_phonotactics in zip(adrc_obj_out.scdict_phonotactics, [
            ['VCCVC', 'VCVC', 'VCVCV'], ['VCVC', 'VCCVC', 'VCVCV']]):
        assert set(adrc_obj_out.scdict_phonotactics[struc]) == set(
            exp_phonotactics)

    # test with mode=reconstruct
    # set up
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="H",
                    target_language="EAH", mode="reconstruct")
    # run function
    adrc_obj_out = get_noncrossval_sc(adrc_obj, None)
    # assert result
    assert adrc_obj_out.scdict == {
        '#-': ['-'],
        '#a': [
            'aː',
            'ɒ'],
        '-#': ['oz'],
        'a': [
            'uː',
            'aː'],
        'at͡ʃi#': ['-'],
        'j': ['jn'],
        'ld': ['ɟ'],
        'n#': ['r'],
        'ɣ': ['t͡ʃ'],
        'ɣ#': ['-']}
    assert adrc_obj_out.sedict == {
        '#-<*-': 3,
        '#a<*aː': 2,
        '#a<*ɒ': 1,
        '-#<*oz': 1,
        'a<*aː': 1,
        'a<*uː': 1,
        'at͡ʃi#<*-': 1,
        'j<*jn': 1,
        'ld<*ɟ': 1,
        'n#<*r': 1,
        'ɣ#<*-': 1,
        'ɣ<*t͡ʃ': 1}
    assert adrc_obj_out.scdict_phonotactics == {}

    # test with write=Path
    # set up
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH")
    # set up path
    path2noncrossval = Path(__file__).parent / "test_get_noncrossval_sc.txt"
    # run function
    adrc_obj_out = get_noncrossval_sc(adrc_obj, path2noncrossval)
    # read and assert result
    out = literal_eval(open(path2noncrossval).read())
    # phonotactic inventory has randomness
    assert set(out.pop(3)) == {'VCVC', 'VCVCV', 'VCCVC'}
    assert out == [{'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'], 'n': ['n'],
                    't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']},
                   {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1, 'l<l': 1, 'n<n': 1,
                    't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2},
                   {'a<a': [1, 2, 3], 'd<d': [2],
                    'i<ɯ': [1], 'j<j': [3], 'l<l': [2],
                    'n<n': [3], 't͡ʃ<t͡ʃː': [1], 'ɣ<ɣ': [1, 2]},
                   {'VCCVC<VCCVC': 1, 'VCVC<VCVC': 1, 'VCVCV<VCVCV': 1},
                   {'VCCVC<VCCVC': [2], 'VCVC<VCVC': [3], 'VCVCV<VCVCV': [1]}]
    # tear down
    remove(path2noncrossval)
    del adrc_obj_out, adrc_obj, path2noncrossval


def test_get_crossval_data():
    """check if correct row is dropped from df for cross-validation"""
    # set up
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH")
    # run function
    # first cog isolated, missing sc: a ɣ a t͡ʃ i - a ɣ a t͡ʃː ɯ
    adrc_obj_out = get_crossval_data(adrc_obj, 0, None)
    # assert result
    assert adrc_obj_out.scdict == {'a': ['a'], 'd': ['d'], 'j': ['j'],
                                   'l': ['l'], 'n': ['n'], 'ɣ': ['ɣ']}
    assert adrc_obj_out.sedict == {'a<a': 4, 'd<d': 1, 'j<j': 1, 'l<l': 1,
                                   'n<n': 1, 'ɣ<ɣ': 1}
    for struc, exp_phonotactics in zip(adrc_obj_out.scdict_phonotactics, [
            ['VCCVC', 'VCVC'], ['VCVC', 'VCCVC']]):
        assert set(adrc_obj_out.scdict_phonotactics[struc]) == set(
            exp_phonotactics)

    # isolate different cogset
    # set up
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH")
    # run function
    # first cog isolated, missing sc: aː l ɟ uː - a l d a w
    adrc_obj_out = get_crossval_data(adrc_obj, 1, None)
    # assert result
    assert adrc_obj_out.scdict == {'a': ['a'], 'j': ['j'], 'n': ['n'],
                                   't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']}
    assert adrc_obj_out.sedict == {'a<a': 4, 'i<ɯ': 1, 'j<j': 1,
                                   'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 1}
    for struc, exp_phonotactics in zip(adrc_obj_out.scdict_phonotactics, [
            ['VCVC', 'VCVCV'], ['VCVC', 'VCVCV']]):
        assert set(adrc_obj_out.scdict_phonotactics[struc]) == set(
            exp_phonotactics)

    # isolate yet another cogset
    # set up
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH")
    # run function
    # first cog isolated, missing sc: a j a n - a j a n
    adrc_obj_out = get_crossval_data(adrc_obj, 2, None)
    # assert result
    assert adrc_obj_out.scdict == {'a': ['a'], 'd': ['d'], 'l': ['l'],
                                   't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']}
    assert adrc_obj_out.sedict == {'a<a': 4, 'd<d': 1, 'i<ɯ': 1, 'l<l': 1,
                                   't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2}
    for struc, exp_phonotactics in zip(adrc_obj_out.scdict_phonotactics, [
            ['VCVCV', 'VCCVC'], ['VCVCV', 'VCCVC']]):
        assert set(adrc_obj_out.scdict_phonotactics[struc]) == set(
            exp_phonotactics)

    # test with mode="reconstruct"
    # set up
    adrc_obj = Adrc(
        forms_csv=PATH2FORMS,
        source_language="H",
        target_language="EAH",
        mode="reconstruct")
    # run function
    # first cog isolated, missing sc: a j a n - a j a n
    adrc_obj_out = get_crossval_data(adrc_obj, 2, None)
    # assert result
    assert adrc_obj_out.scdict == {
        '#-': ['-'],
        '#a': ['aː'],
        'a': ['uː'],
        'at͡ʃi#': ['-'],
        'ld': ['ɟ'],
        'ɣ': ['t͡ʃ'],
        'ɣ#': ['-']}
    assert adrc_obj_out.sedict == {
        '#-<*-': 2,
        '#a<*aː': 2,
        'a<*uː': 1,
        'at͡ʃi#<*-': 1,
        'ld<*ɟ': 1,
        'ɣ#<*-': 1,
        'ɣ<*t͡ʃ': 1}
    assert adrc_obj_out.scdict_phonotactics == {}

    # test with writesc = Path()
    # set up
    path2outfolder = Path(__file__).parent / \
        "output_files"  # has to be folder!
    adrc_obj = Adrc(
        forms_csv=PATH2FORMS,
        source_language="H",
        target_language="EAH",
        mode="reconstruct")
    # run function
    # first cog isolated, missing sc: a j a n - a j a n
    adrc_obj_out = get_crossval_data(adrc_obj, 2, path2outfolder)
    # assert result
    assert adrc_obj_out.scdict == {
        '#-': ['-'],
        '#a': ['aː'],
        'a': ['uː'],
        'at͡ʃi#': ['-'],
        'ld': ['ɟ'],
        'ɣ': ['t͡ʃ'],
        'ɣ#': ['-']}
    assert adrc_obj_out.sedict == {
        '#-<*-': 2,
        '#a<*aː': 2,
        'a<*uː': 1,
        'at͡ʃi#<*-': 1,
        'ld<*ɟ': 1,
        'ɣ#<*-': 1,
        'ɣ<*t͡ʃ': 1}
    assert adrc_obj_out.scdict_phonotactics == {}
    # assert file written correctly
    assert literal_eval(open(path2outfolder / "sc2isolated.txt").read()) == [
        {'#-': ['-'], '#a': ['aː'], 'a': ['uː'], 'at͡ʃi#': ['-'],
         'ld': ['ɟ'], 'ɣ': ['t͡ʃ'], 'ɣ#': ['-']},
        {'#-<*-': 2, '#a<*aː': 2, 'a<*uː': 1,
         'at͡ʃi#<*-': 1, 'ld<*ɟ': 1, 'ɣ#<*-': 1, 'ɣ<*t͡ʃ': 1},
        {'#-<*-': [1, 2], '#a<*aː': [1, 2], 'a<*uː': [2], 'at͡ʃi#<*-': [1],
         'ld<*ɟ': [2], 'ɣ#<*-': [2], 'ɣ<*t͡ʃ': [1]}, {}, {}, {}]

    # tear down
    remove(path2outfolder / "sc2isolated.txt")
    del adrc_obj_out, adrc_obj, path2outfolder


def test_loop_thru_data():
    """Is cross-validation called and loop run?"""
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH")
    # assert output is correct
    assert loop_thru_data(
        adrc_obj, 1, 1, 100, 49, False, False, False, False, False, [
            10, 50, 100, 500, 1000], 'adapt', False, True) == adrc_obj
    # set up expected output
    df_exp = DataFrame([
        ("aɣat͡ʃi", "aɣat͡ʃːɯ", 1, float("inf"), "KeyError"),
        ("aldaɣ", "aldaɣ", 2, float("inf"), "KeyError"),
        ("ajan", "ajan", 3, float("inf"), "KeyError")],
        columns=['Target_Form', 'Source_Form',
                 'Cognacy', 'guesses', 'best_guess'])
    # assert output.dfety is correct
    assert_frame_equal(adrc_obj.dfety, df_exp)
    # assert popped words were plugged back in consistently in loop
    assert adrc_obj.forms_target_language == ["aɣat͡ʃi", "aldaɣ", "ajan"]
    # tear down
    del df_exp, adrc_obj


def test_getnse4df():
    """Is normalised sum of examples for data frame calculated correctly?"""
    # test with mode="adapt"
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_AD)

    out_adrc_obj = get_nse4df(adrc_obj, "Target_Form")
    # assert output was correct
    assert_frame_equal(out_adrc_obj.dfety, read_csv(
        Path(__file__).parent / "expected_files" / "getnse4df_ad.csv"))

    # test with mode="reconstruct"
    adrc_obj = Adrc(
        forms_csv=PATH2FORMS,
        source_language="H",
        target_language="EAH",
        scdictlist=PATH2SC_RC,
        mode="reconstruct")

    out_adrc_obj = get_nse4df(adrc_obj, "Target_Form")
    # assert output was correct
    assert_frame_equal(out_adrc_obj.dfety, read_csv(
        Path(__file__).parent / "expected_files" / "getnse4df_rc.csv"))

    # tear down
    del adrc_obj, out_adrc_obj


def test_phonotactics_predicted():
    """Correct boolean returned when checking if phonotactics was predicted?"""
    adrc_obj = Adrc()

    df_in = DataFrame({
        "Target_Form": ["abc", "def", "ghi"],
        "predicted_phonotactics": [["CCC", "VVV"], ["CVC"], ["CCV", "CCC"]]})

    df_exp = df_in.assign(phonotactics_predicted=[False, True, True])
    adrc_obj.dfety = df_in

    assert_frame_equal(phonotactics_predicted(adrc_obj).dfety, df_exp)

    # tear down
    del adrc_obj, df_in, df_exp


def test_get_dist():
    """Are (normalised) Levenshtein Distances calculated correctly?"""
    adrc_obj = Adrc()

    # set up input
    dfety = DataFrame({
        "best_guess": ["will not buy", "record", "scratched"],
        "Target_Form": ["won't buy", "tobacconists", "scratched"]})

    adrc_obj.dfety = dfety

    # set up expected outcome
    df_exp = dfety.assign(
        fast_levenshtein_distance_best_guess_Target_Form=[5, 10, 0],
        fast_levenshtein_distance_div_maxlen_best_guess_Target_Form=[
            0.42, 0.83, 0.00])
    assert_frame_equal(get_dist(adrc_obj, "best_guess").dfety, df_exp)

    # tear down
    del adrc_obj, dfety, df_exp


def test_postprocess():
    """Is result of loanpy.sanity.loop_thru_data postprocessed correctly?"""
    # test with mode="adapt"
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_AD)
    # pretend guesses are already made
    adrc_obj.dfety["best_guess"] = ["aɣa", "bla", "ajan"]
    # run function with show_workflow=False
    adrc_obj_out = postprocess(adrc_obj)
    assert_frame_equal(adrc_obj_out.dfety, read_csv(
        Path(__file__).parent / "expected_files" / "postprocess_ad.csv"))

    # test with mode="reconstruct"
    adrc_obj = Adrc(
        forms_csv=PATH2FORMS,
        source_language="H",
        target_language="EAH",
        scdictlist=PATH2SC_RC,
        mode="reconstruct")
    # pretend guesses are already made
    adrc_obj.dfety["best_guess"] = ["aːt͡ʃ", "bla", "ɒjnaːr"]
    # run function with show_workflow=False
    adrc_obj_out = postprocess(adrc_obj)
    rfile = read_csv(
        Path(__file__).parent / "expected_files" / "postprocess_rc.csv")
    for i in adrc_obj_out.dfety.columns:
        print(adrc_obj_out.dfety[i])
    for j in rfile:
        print(rfile[j])
    assert_frame_equal(adrc_obj_out.dfety, read_csv(
        Path(__file__).parent / "expected_files" / "postprocess_rc.csv"))

    # test with show_workflow
    # test with mode="adapt"
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_AD)
    # pretend guesses are already made
    adrc_obj.dfety["best_guess"] = ["aɣa", "bla", "ajan"]
    adrc_obj.dfety["predicted_phonotactics"] = [
        "['VCV', 'CCC']", "['CCC', 'VCC']", "['VCVC']"]
    # run function with show_workflow=False
    adrc_obj_out = postprocess(adrc_obj)
    assert_frame_equal(
        adrc_obj_out.dfety,
        read_csv(
            Path(__file__).parent /
            "expected_files" /
            "postprocess_ad_workflow.csv"))


def test_make_stat():
    pass  # unittest = integrationtest, there was nothing to mock.


def test_gettprfpr():
    pass  # unittest = integrationtest, there was nothing to mock.


def test_plot_roc():
    """Is result plotted correctly to .jpg? Check result manually at
    output_files / "mockplot.jpg!"""

    path2mockplot = Path(__file__).parent / "output_files" / "mockplot.jpg"

    plot_roc(guesslist=[1, 2, 3],
             plot_to=path2mockplot, tpr_fpr_opt=(
        [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
        [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
        (0.501, 0.6, 0.099)), opt_howmany=1,
        opt_tpr=0.6, len_df=3, mode="adapt")

    # verify manually that results in output_files and expected_files are same


def test_postprocess2():
    """Is result of loanpy.sanity.postprocess postprocessed correctly?"""
    adrc_obj = Adrc(forms_csv=PATH2FORMS, source_language="WOT",
                    target_language="EAH", scdictlist=PATH2SC_AD)
    # pretend guesses are already made
    adrc_obj.dfety["guesses"] = [1, 2, 3]

    assert postprocess2(adrc_obj, [4, 5, 6], "adapt") == (5, '3/3', '100%')

    # define path for output
    path2out = Path(__file__).parent / "postprocess2_integration.csv"
    # assert write_to works
    assert postprocess2(adrc_obj, [4, 5, 6], "adapt", path2out
                        ) == (5, '3/3', '100%')
    # since guesses were manually inserted!
    assert_frame_equal(read_csv(path2out), adrc_obj.dfety)

    # tear down
    remove(path2out)
    remove(Path(str(path2out)[:-4] + ".jpg"))
    del path2out, adrc_obj

    #
