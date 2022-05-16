#todo: test plot_ld_nld

from collections import OrderedDict
from datetime import datetime
from os import remove
from pathlib import Path
from time import struct_time
from unittest.mock import call, patch

from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from pytest import raises

from loanpy.sanity import (ArgumentsAlreadyTested, check_cache, eval_all,
                           eval_one, get_sc, gettprfpr, make_stat, plot_roc,
                           write_to_cache, write_workflow)
from loanpy import sanity as sy # for monkey patches

def test_make_cache():
    """test if DIY cache is initiated correctly and args checked in it"""

    #set up path of mock cache
    mockpath = Path(__file__).parent / "test_opt_param.csv"
    #make sure this file does not exist (e.g. from previous tests)
    try: remove(mockpath)
    except FileNotFoundError: pass
    #set up first expected outcome, a pandas data frame
    exp1 = DataFrame(columns=["arg1", "arg2", "arg3", "opt_tpr",
    "optimal_howmany", "opt_tp", "timing", "date"])

    #assert first break works: cache not found
    check_cache(mockpath, {"arg1": "x", "arg2": "y", "arg3": "z"})
    assert_frame_equal(read_csv(mockpath), exp1)

    #tear down
    remove(mockpath)

    #check if exception is rased if these params were tested already

    #mock read_csv
    with patch("loanpy.sanity.read_csv") as read_csv_mock:
        read_csv_mock.return_value = DataFrame({"arg1": ["x"], "arg2": ["y"],
        "arg3": ["z"]})
        # assert exception is raised bc args exist in cache already
        with raises(ArgumentsAlreadyTested) as aat_mock:
            check_cache(mockpath,
            {"arg1": "x", "arg2": "y", "arg3": "z"})
        assert str(aat_mock.value) == f"These arguments were tested \
already, see {mockpath} line 1! (start counting at 1 in 1st row)"

    #assert call for read_csv
    read_csv_mock.assert_called_with(mockpath, usecols=["arg1", "arg2", "arg3"])

    #check if nothing happens if arguments were NOT tested already
    with patch("loanpy.sanity.read_csv") as read_csv_mock:
        read_csv_mock.return_value = DataFrame({"arg1": ["a"], "arg2": ["b"],
        "arg3": ["c"]})
        #assert that the function runs, does nothing, and returns None
        assert check_cache(mockpath,
        {"arg1": "x", "arg2": "y", "arg3": "z"}) is None

    #assert call for read_csv
    read_csv_mock.assert_called_with(mockpath, usecols=["arg1", "arg2", "arg3"])

def test_get_sc():
    """check if correct row is dropped from df for cross-validation"""

    #set up mock class for input and instantiate it
    class AdrcMonkey:
        def __init__(self):
            self.get_sound_corresp_called_with = []
            self.dfety = DataFrame({"fruit": ["apple", "banana", "cherry"], "color": ["green", "yellow", "red"]})
        def get_sound_corresp(self, *args):
            self.get_sound_corresp_called_with.append([*args])
            return [{"d1": "scdict"}, {}, {}, {"d3": "scdict_struc"}, {}, {}]

    adrcmock = AdrcMonkey()
    #set up actual output as variable
    actual_out = get_sc(adrcmock, 1, None)

    #assert scdict and scdict_struc were correctly plugged into adrc_class
    assert actual_out.scdict == {"d1": "scdict"}
    assert actual_out.scdict_struc == {"d3": "scdict_struc"}
    assert adrcmock.get_sound_corresp_called_with == [[None]]

    #tear down
    del AdrcMonkey, adrcmock, actual_out

def test_eval_one():

    #set up mock class
    class AdrcMonkey:
        def __init__(self, adapt_returns):
            self.adapt_returns = iter(adapt_returns)
            self.adapt_called_with = []
            self.workflow = None
        def adapt(self, *args):
            self.adapt_called_with.append([*args])
            self.workflow = "gowiththeflow"
            return next(self.adapt_returns)

    #set up: mock get_howmany
    with patch("loanpy.sanity.get_howmany", side_effect=[
    (2, 1, 1), (4, 1, 1), (6, 1, 1)]) as get_howmany_mock:

        #assert None is returned if target word is not in the predictions

        #set up, create instance of mock class
        adrcmock = AdrcMonkey(adapt_returns=[
        "kek, kok", "kek, kok, hek, hok", "kek, kok, hek, hok, ketke, kotke"])

        #assert None/float(inf) is in dict if target word is not in the predictions
        assert eval_one(adrc_obj=adrcmock, srcwrd="kiki", tgtwrd="hotke",
        guesslist=[2, 4, 6], max_struc=1, max_paths=1, vowelharmony=False,
        clusterised=False, sort_by_nse=False, struc_filter=False,
        show_workflow=False, mode="adapt") == {"sol_idx_plus1": float("inf"), "best_guess": "kek"}

        #assert 2 mock calls: get_howmany, adapt
        get_howmany_mock.has_calls = [
        call(2, 1, 1), call(4, 1, 1), call(6, 1, 1)]
        #assert adapt was called thrice, once for each element of guesslist
        assert adrcmock.adapt_called_with == [
        ["kiki", 2, 1, 1, False, False, False, False, False],
        ["kiki", 4, 1, 1, False, False, False, False, False],
        ["kiki", 6, 1, 1, False, False, False, False, False]]

    #set up: mock get_howmany
    with patch("loanpy.sanity.get_howmany", side_effect=[
        (2, 1, 1), (4, 1, 1), (6, 1, 1), (8, 1, 1)]) as get_howmany_mock:

        #assert list index is returned if target word was in predictions

        #set up: overwrite mock class
        adrcmock = AdrcMonkey(adapt_returns=[
        "kek, kok", "kek, kok, hek, hok", "kek, kok, hek, hok, ketke, kotke",
        "kek, kok, hek, hok, ketke, kotke, hetke, hotke"])

        #assert list index+1 is returned if target word was in predictions
        assert eval_one(adrc_obj=adrcmock, srcwrd="kiki", tgtwrd="hotke",
        guesslist=[2, 4, 6, 8], max_struc=1, max_paths=1, vowelharmony=False,
        clusterised=False, sort_by_nse=False, struc_filter=False,
        show_workflow=False, mode="mode") == {'best_guess': '', 'sol_idx_plus1': float("inf")}

    #set up: mock get_howmany
    with patch("loanpy.sanity.get_howmany", side_effect=[
        (2, 1, 1), (4, 1, 1), (6, 1, 1), (8, 1, 1)]) as get_howmany_mock:

        #assert workflow is returned correctly if show_workflow == True

        #set up: overwrite mock class
        adrcmock = AdrcMonkey(adapt_returns=[
        "kek, kok", "kek, kok, hek, hok", "kek, kok, hek, hok, ketke, kotke",
        "kek, kok, hek, hok, ketke, kotke, hetke, hotke"])

        #assert list index+1 is returned if target word was in predictions
        assert eval_one(adrc_obj=adrcmock, srcwrd="kiki", tgtwrd="hotke",
        guesslist=[2, 4, 6, 8], max_struc=1, max_paths=1, vowelharmony=False,
        clusterised=False, sort_by_nse=False, struc_filter=False,
        show_workflow=True, mode="adapt") == {'best_guess': 'hotke', 'sol_idx_plus1': 8, 'workflow': 'gowiththeflow'}

def test_make_stat():
    #no set up or tear down needed. Nothing to mock.
    assert make_stat(opt_fp=0.099, opt_tp=0.6, max_fp=1000, len_df=10
    ) == (100, "6/10", "60%")

def test_gettprfpr():
    #no steup or teardown or mock needed
    assert gettprfpr(
    step7_fp=[10, None, 20, 4, 17, None, None, 8, 9, 120],
    fplist=[1, 3, 5, 7, 9, 99, 999],
    len_step7=10) == (
    [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
    [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
    (0.501, 0.6, 0.099))

def test_write_to_cache():

    #set up path to mock cache
    mock_cache_path = Path(__file__).parent / "mock_cache.csv"
    #set up mock init args, used in 2 places so var necessary
    init_args_mock = {"formscsv": "forms.csv", "tgt_lg": "EAH",
    "src_lg": "WOT", "crossval": True,
    "opt_param_path": mock_cache_path, "guesslist": [[2, 4, 6, 8]],
    "max_struc": 1, "max_paths": 1, "writesc": False,
    "writesc_struc": False, "vowelharmony": False,
    "only_documented_clusters": False, "sort_by_nse": False,
    "struc_filter": False, "show_workflow": False, "write": False,
    "outname": "viz", "plot_to": None, "plotldnld": False}
    #set up return value of mocked read_csv, used in 2 places
    read_csv_mock_returns = DataFrame(columns=list(init_args_mock)+["opt_tpr",
    "optimal_howmany", "opt_tp", "timing", "date"])
    #set up 2 dfs with which concat is being valled
    df_concat_call1 = read_csv_mock_returns
    df_concat_call2 = DataFrame({"formscsv": ["forms.csv"], "tgt_lg": ["EAH"],
    "src_lg": ["WOT"], "crossval": ["True"],
    "opt_param_path": [str(mock_cache_path)], "guesslist": ["[[2, 4, 6, 8]]"],
    "max_struc": ["1"], "max_paths": ["1"], "writesc": ["False"],
    "writesc_struc": ["False"], "vowelharmony": ["False"],
    "only_documented_clusters": ["False"], "sort_by_nse": ["False"],
    "struc_filter": ["False"], "show_workflow": ["False"], "write": ["False"],
    "outname": ["viz"], "plot_to": ["None"], "plotldnld": ["False"],
    "optimal_howmany": [0.501], "opt_tp": [0.6], "opt_tpr": [0.099],
    "timing": ["14:00"], "date": [datetime.now().strftime("%x %X")]})
    #set up mock cache
    DataFrame().to_csv(mock_cache_path, index=False, encoding="utf-8")
    #set up expected new cache
    df_exp = DataFrame(
    {"fruit": ["cherry", "banana", "apple"],
    "color": ["red", "yellow", "green"],
    "opt_tpr": [0.8, 0.6, 0.4]})
    #mock pandas concat
    with patch("loanpy.sanity.concat") as concat_mock:
        concat_mock.return_value = DataFrame(
        {"fruit": ["apple", "banana", "cherry"],
        "color": ["green", "yellow", "red"],
        "opt_tpr": [0.4, 0.6, 0.8]})
        #mock pandas read_csv
        with patch("loanpy.sanity.read_csv") as read_csv_mock:
            read_csv_mock.return_value = read_csv_mock_returns
            #mock strftime
            with patch("loanpy.sanity.strftime") as strftime_mock:
                strftime_mock.return_value = "14:00"

                #write to mock cache
                write_to_cache(
                stat=(0.501, 0.6, 0.099),
                init_args=init_args_mock,
                opt_param_path=mock_cache_path, start=1, end=2)

                #assert cache was written correctly
                assert_frame_equal(read_csv(mock_cache_path), df_exp)

                #assert pandas concat call
                assert_frame_equal(concat_mock.call_args_list[0][0][0][0], df_concat_call1)
                assert_frame_equal(concat_mock.call_args_list[0][0][0][1], df_concat_call2, check_dtype=False)
                #assert pandas read_csv called with
                read_csv_mock.assert_called_with(mock_cache_path)
                #assert strftime call
                strftime_mock.assert_called()

    #tear down
    remove(mock_cache_path)
    del mock_cache_path, df_exp, read_csv_mock_returns, init_args_mock

def test_eval_all():

    #set up mock init_args (mocks locals())
    init_args_mock = {"opt_param_path": "mockpath",
    "formscsv": "forms.csv", "tgt_lg": "EAH",
    "src_lg": "WOT", "crossval": True,
     "guesslist": [1, 2, 3],
    "max_struc": 1, "max_paths": 1, "writesc": False,
    "vowelharmony": False,
    "clusterised": False, "sort_by_nse": False,
    "struc_filter": False, "show_workflow": False,
    "scdictbase": False,
    "mode": "adapt",
    "write_to": None,
    "plot_to": None, "plotldnld": False}

    #set up mock df for both cldf2pd AND read_csv
    df_mock_read = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
    "Target_Form": ["hehe", "pupa", "hapa"]})

    #set up expected output data frame
    df_exp = df_mock_read.assign(guesses=[1, None, 3], best_guess=["hehe", "pupa", "hapa"])

    #set up mock class for var "self", get_sc will return these
    class GetMonkeySc:
        def __init__(self, scdict=None, scdict_struc=None):
            self.scdict = scdict
            self.scdict_struc = scdict_struc
            self.dfety = df_mock_read

    #create 3 instances of mock class, to be returned by get_sc (sdie eff.)
    #one for each element of guesslist
    #simulate the sound changes that would have been extracted from
    #each crossvalidated data frame
    mockgetsc1 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "i" missing b/c "kiki" isolated
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc2 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "u" missing b/c "buba" isolated
                                   "i": ["e"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc3 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"], "i": ["e"],  # nth missing b/c sc in "kaba" are captured in "kiki" and "buba"
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})

    #set up: mock 9 functions: check_cache, cldf2pd, read_forms, get_sc, eval_one,
    #gettprfpr, make_stat, write_to_cache, time
    with patch("loanpy.sanity.check_cache") as check_cache_mock:
        check_cache_mock.return_value = None
        #with patch("loanpy.sanity.cldf2pd") as cldf2pd_mock:
        #    cldf2pd_mock.return_value = df_mock_read
        #    with patch("loanpy.sanity.read_forms") as read_forms_mock:
        #        read_forms_mock.return_value = df_mock_read
        with patch("loanpy.sanity.Adrc") as Adrc_mock:
            Adrc_mock.return_value = GetMonkeySc()
            with patch("loanpy.sanity.get_sc",
            side_effect=[
            mockgetsc1, mockgetsc2, mockgetsc3]) as get_sc_mock:
                with patch("loanpy.sanity.eval_one",
                side_effect=[{"sol_idx_plus1": 1, "best_guess": "hehe"},
                             {"sol_idx_plus1": None, "best_guess": "pupa"},
                             {"sol_idx_plus1": 3, "best_guess": "hapa"}]) as eval_one_mock:
                    with patch("loanpy.sanity.gettprfpr") as gettprfpr_mock:
                        gettprfpr_mock.return_value = (
                        [0.0, 0.6, 0.7],
                        [0.001, 0.099, 1.0],
                        (0.501, 0.6, 0.099))
                        with patch("loanpy.sanity.make_stat") as make_stat_mock:
                            make_stat_mock.return_value = (1, '2/3', '60%')
                            with patch("loanpy.sanity.time", side_effect=[
                            20, 22]) as time_mock:
                                with patch("loanpy.sanity.write_to_cache") as write_to_cache_mock:
                                    write_to_cache_mock.return_value = ""

                                    #assert evaluation runs correctly
                                    assert_frame_equal(eval_all(
                                    opt_param_path="mockpath",
                                    formscsv="forms.csv",
                                    tgt_lg="EAH",
                                    src_lg="WOT",
                                    crossval=True,

                                    guesslist=[1,2,3],
                                    max_struc=1,
                                    max_paths=1,
                                    writesc=False,
                                    vowelharmony=False,
                                    clusterised=False,
                                    sort_by_nse=False,
                                    struc_filter=False,
                                    show_workflow=False,

                                    write_to=None,
                                    plot_to=None,
                                    plotldnld=False
                                    ), df_exp)

    #assert calls

    #assert first mock (check_cache) called correctly
    check_cache_mock.assert_called_with(
    "mockpath", init_args_mock)

    #assert 4th mock (get_sc) called correctly
    #first call get_sc
    assert type(get_sc_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert get_sc_mock.call_args_list[0][0][1:] == (0, False)
    #2nd call get_sc
    assert type(get_sc_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert get_sc_mock.call_args_list[1][0][1:] == (1, False)
    #3rd call get_sc
    assert type(get_sc_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert get_sc_mock.call_args_list[2][0][1:] == (2, False)

    #assert 5th mock (eval_one) called correctly
    #first call
    assert type(eval_one_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[0][0][1:] == (
    ('kiki', 'hehe', [1, 2, 3], 1, 1, False, False, False, False, False, "adapt"))  # kwargs!
    assert eval_one_mock.call_args_list[0][1] == {}  # args!
    #2nd call eval_mock
    assert type(eval_one_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[1][0][1:] == (
    'buba', 'pupa', [1, 2, 3], 1, 1, False, False, False, False, False, "adapt") # kwargs!
    assert eval_one_mock.call_args_list[1][1] == {}  # args!
    #3rd call eval mock
    assert type(eval_one_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[2][0][1:] == (
    "kaba", "hapa", [1, 2, 3], 1, 1, False, False, False, False, False, "adapt")  # kwargs!
    assert eval_one_mock.call_args_list[2][1] == {}  # args!

    #assert 6th mock call was correct
    gettprfpr_mock.assert_called_with([1, None, 3], [0, 1, 2], 3)

    #assert 7th mock call went correctly
    make_stat_mock.assert_called_with(0.099, 0.6, 2, 3)  # opt_tp, opt_fp, guesslist[-1], len_df

    #assert 8th mock call went correctly
    write_to_cache_mock.assert_called_with((1, '2/3', '60%'), {
    'opt_param_path': 'mockpath', 'formscsv': 'forms.csv', 'tgt_lg': 'EAH',
    'src_lg': 'WOT', 'crossval': True, 'guesslist': [1, 2, 3], 'max_struc': 1,
    'max_paths': 1, 'writesc': False, 'vowelharmony': False, 'clusterised': False,
    'sort_by_nse': False, 'struc_filter': False, 'show_workflow': False,
    'scdictbase': False, 'mode': "adapt",
    'write_to': None, 'plot_to': None, 'plotldnld': False},
    'mockpath', 20, 22)

    #assert 9th mock call went correctly
    time_mock.assert_called_with()

    del df_exp, df_mock_read, init_args_mock, GetMonkeySc

    #same as before but mode= "reconstruct"

    #set up mock init_args (mocks locals())
    init_args_mock = {"opt_param_path": "mockpath",
    "formscsv": "forms.csv", "tgt_lg": "EAH",
    "src_lg": "H", "crossval": True,
     "guesslist": [1, 2, 3],
    "max_struc": 1, "max_paths": 1, "writesc": False,
    "vowelharmony": False,
    "clusterised": False, "sort_by_nse": False,
    "struc_filter": False, "show_workflow": False,
    "scdictbase": False,
    "mode": "reconstruct",
    "write_to": None,
    "plot_to": None, "plotldnld": False}

    #set up mock df for both cldf2pd AND read_csv
    df_mock_read = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
    "Target_Form": ["hehe", "pupa", "hapa"]})

    #set up expected output data frame
    df_exp = df_mock_read.assign(guesses=[1, None, 3], best_guess=["hehe", "pupa", "hapa"])

    #set up mock class for var "self", get_sc will return these
    class GetMonkeySc:
        def __init__(self, scdict=None, scdict_struc=None):
            self.scdict = scdict
            self.scdict_struc = scdict_struc
            self.dfety = df_mock_read

    #create 3 instances of mock class, to be returned by get_sc (sdie eff.)
    #one for each element of guesslist
    #simulate the sound changes that would have been extracted from
    #each crossvalidated data frame
    mockgetsc1 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "i" missing b/c "kiki" isolated
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc2 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "u" missing b/c "buba" isolated
                                   "i": ["e"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc3 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"], "i": ["e"],  # nth missing b/c sc in "kaba" are captured in "kiki" and "buba"
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})

    #set up: mock 9 functions: check_cache, cldf2pd, read_forms, get_sc, eval_one,
    #gettprfpr, make_stat, write_to_cache, time
    with patch("loanpy.sanity.check_cache") as check_cache_mock:
        check_cache_mock.return_value = None
        #with patch("loanpy.sanity.cldf2pd") as cldf2pd_mock:
        #    cldf2pd_mock.return_value = df_mock_read
        #    with patch("loanpy.sanity.read_forms") as read_forms_mock:
        #        read_forms_mock.return_value = df_mock_read
        with patch("loanpy.sanity.Adrc") as Adrc_mock:
            Adrc_mock.return_value = GetMonkeySc()
            with patch("loanpy.sanity.get_sc",
            side_effect=[
            mockgetsc1, mockgetsc2, mockgetsc3]) as get_sc_mock:
                with patch("loanpy.sanity.eval_one",
                side_effect=[{"sol_idx_plus1": 1, "best_guess": "hehe"},
                             {"sol_idx_plus1": None, "best_guess": "pupa"},
                             {"sol_idx_plus1": 3, "best_guess": "hapa"}]) as eval_one_mock:
                    with patch("loanpy.sanity.gettprfpr") as gettprfpr_mock:
                        gettprfpr_mock.return_value = (
                        [0.0, 0.6, 0.7],
                        [0.001, 0.099, 1.0],
                        (0.501, 0.6, 0.099))
                        with patch("loanpy.sanity.make_stat") as make_stat_mock:
                            make_stat_mock.return_value = (1, '2/3', '60%')
                            with patch("loanpy.sanity.time", side_effect=[
                            20, 22]) as time_mock:
                                with patch("loanpy.sanity.write_to_cache") as write_to_cache_mock:
                                    write_to_cache_mock.return_value = ""

                                    #assert evaluation runs correctly
                                    assert_frame_equal(eval_all(
                                    opt_param_path="mockpath",
                                    formscsv="forms.csv",
                                    tgt_lg="EAH",
                                    src_lg="H",
                                    crossval=True,

                                    guesslist=[1,2,3],
                                    max_struc=1,
                                    max_paths=1,
                                    writesc=False,
                                    vowelharmony=False,
                                    clusterised=False,
                                    sort_by_nse=False,
                                    struc_filter=False,
                                    show_workflow=False,
                                    mode="reconstruct",

                                    write_to=None,
                                    plot_to=None,
                                    plotldnld=False
                                    ), df_exp)

    #assert calls

    #assert first mock (check_cache) called correctly
    check_cache_mock.assert_called_with(
    "mockpath", init_args_mock)

    #assert 4th mock (get_sc) called correctly
    #first call get_sc
    assert type(get_sc_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert get_sc_mock.call_args_list[0][0][1:] == (0, False)
    #2nd call get_sc
    assert type(get_sc_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert get_sc_mock.call_args_list[1][0][1:] == (1, False)
    #3rd call get_sc
    assert type(get_sc_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert get_sc_mock.call_args_list[2][0][1:] == (2, False)

    #assert 5th mock (eval_one) called correctly
    #first call
    assert type(eval_one_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[0][0][1:] == (
    ('kiki', 'hehe', [1, 2, 3], 1, 1, False, False, False, False, False, "reconstruct"))  # kwargs!
    assert eval_one_mock.call_args_list[0][1] == {}  # args!
    #2nd call eval_mock
    assert type(eval_one_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[1][0][1:] == (
    'buba', 'pupa', [1, 2, 3], 1, 1, False, False, False, False, False, "reconstruct") # kwargs!
    assert eval_one_mock.call_args_list[1][1] == {}  # args!
    #3rd call eval mock
    assert type(eval_one_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[2][0][1:] == (
    "kaba", "hapa", [1, 2, 3], 1, 1, False, False, False, False, False, "reconstruct")  # kwargs!
    assert eval_one_mock.call_args_list[2][1] == {}  # args!

    #assert 6th mock call was correct
    gettprfpr_mock.assert_called_with([1, None, 3], [0, 1, 2], 3)

    #assert 7th mock call went correctly
    make_stat_mock.assert_called_with(0.099, 0.6, 2, 3)  # opt_tp, opt_fp, guesslist[-1], len_df

    #assert 8th mock call went correctly
    write_to_cache_mock.assert_called_with((1, '2/3', '60%'), {
    'opt_param_path': 'mockpath', 'formscsv': 'forms.csv', 'tgt_lg': 'EAH',
    'src_lg': 'H', 'crossval': True, 'guesslist': [1, 2, 3], 'max_struc': 1,
    'max_paths': 1, 'writesc': False, 'vowelharmony': False, 'clusterised': False,
    'sort_by_nse': False, 'struc_filter': False, 'show_workflow': False,
    'scdictbase': False, 'mode': "reconstruct",
    'write_to': None, 'plot_to': None, 'plotldnld': False},
    'mockpath', 20, 22)

    #assert 9th mock call went correctly
    time_mock.assert_called_with()

    del df_exp, df_mock_read, init_args_mock, GetMonkeySc

def test_plotroc():
    #set up path to mock plot
    path2mockplot = Path(__file__).parent / "mockplot.jpg"
    #mock first input arg
    df_forms_mock = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
     "Target_Form": ["hehe", "pupa", "hapa"],
     "guesses": [1, None, 3]})

    # test with lev_dist and norm_lev_dist == False (default settings)

    # mock all matplotlib funcs without mock return values
    with patch("loanpy.sanity.xlabel") as xlabel_mock:
        with patch("loanpy.sanity.ylabel") as ylabel_mock:
            with patch("loanpy.sanity.plot") as plot_mock:
                with patch("loanpy.sanity.scatter") as scatter_mock:
                    with patch("loanpy.sanity.text") as text_mock:
                        with patch("loanpy.sanity.title") as title_mock:
                            with patch("loanpy.sanity.legend") as legend_mock:
                                with patch("loanpy.sanity.savefig") as savefig_mock:
                                #mocked savefig won't write any file

                                    #run function
                                    plot_roc(df=df_forms_mock, fplist=[0,1,2],
                                    plot_to=path2mockplot, tpr_fpr_opt=(
                                    [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
                                    [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
                                    (0.501, 0.6, 0.099)), opt_howmany=1,
                                    opt_tpr=0.6, len_df=3, mode="adapt")
                                    #mock write fig writes no file so nth 2 remove

    # assert all calls were made with the right args
    xlabel_mock.assert_called_with("fpr")
    ylabel_mock.assert_called_with("tpr")
    plot_mock.assert_called_with([0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
    [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7], label='loanpy.adrc.Adrc.adapt')
    scatter_mock.assert_called_with(0.099, 0.6, marker='x', c='blue', label='Optimum:\nhowmany=0 -> tpr: 0.6')
    text_mock.assert_called_with(0.7, 0.0, 'adapt: 100%=3')
    title_mock.assert_called_with('Predicting loanword adaptation with loanpy.adrc.Adrc.adapt')
    legend_mock.assert_called_with()
    savefig_mock.assert_called_with(path2mockplot)

    # test with lev_dist == True

    # mock all matplotlib funcs without mock return values
    with patch("loanpy.sanity.xlabel") as xlabel_mock:
        with patch("loanpy.sanity.ylabel") as ylabel_mock:
            with patch("loanpy.sanity.plot") as plot_mock:
                with patch("loanpy.sanity.scatter") as scatter_mock:
                    with patch("loanpy.sanity.text") as text_mock:
                        with patch("loanpy.sanity.title") as title_mock:
                            with patch("loanpy.sanity.legend") as legend_mock:
                                with patch("loanpy.sanity.savefig") as savefig_mock:
                                #mocked savefig won't write any file

                                    #run function
                                    plot_roc(df=df_forms_mock, fplist=[0,1,2],
                                    plot_to=path2mockplot, tpr_fpr_opt=(
                                    [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
                                    [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
                                    (0.501, 0.6, 0.099)), opt_howmany=1,
                                    opt_tpr=0.6, len_df=3, mode="adapt",
                                    lev_dist=True, norm_lev_dist=False)
                                    #mock write fig writes no file so nth 2 remove

    # assert all calls were made with the right args
    xlabel_mock.assert_called_with("fpr")
    ylabel_mock.assert_called_with("tpr")
    plot_mock.assert_called_with([0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0], [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7], label='Levenshtein Distance')
    scatter_mock.assert_called_with(0.001, 0.0, marker='x', c='orange', label='tpr: 0% -> LD=1')
    text_mock.assert_called_with(0, 0.6, 'LD: 100%=2')
    title_mock.assert_called_with('loanpy.adrc.Adrc.adapt vs Leveshtein Distance')
    legend_mock.assert_called_with()
    savefig_mock.assert_called_with(path2mockplot)

    # test with norm_lev_dist == True

    # mock all matplotlib funcs without mock return values
    with patch("loanpy.sanity.xlabel") as xlabel_mock:
        with patch("loanpy.sanity.ylabel") as ylabel_mock:
            with patch("loanpy.sanity.plot") as plot_mock:
                with patch("loanpy.sanity.scatter") as scatter_mock:
                    with patch("loanpy.sanity.text") as text_mock:
                        with patch("loanpy.sanity.title") as title_mock:
                            with patch("loanpy.sanity.legend") as legend_mock:
                                with patch("loanpy.sanity.savefig") as savefig_mock:
                                #mocked savefig won't write any file

                                    #run function
                                    plot_roc(df=df_forms_mock, fplist=[0,1,2],
                                    plot_to=path2mockplot, tpr_fpr_opt=(
                                    [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
                                    [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
                                    (0.501, 0.6, 0.099)), opt_howmany=1,
                                    opt_tpr=0.6, len_df=3, mode="adapt",
                                    lev_dist=False, norm_lev_dist=True)
                                    #mock write fig writes no file so nth 2 remove

    # assert all calls were made with the right args
    xlabel_mock.assert_called_with("fpr")
    ylabel_mock.assert_called_with("tpr")
    plot_mock.assert_called_with([0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0], [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7], label='Normalised Lev. Dist.')
    scatter_mock.assert_called_with(0.001, 0.0, marker='x', c='green', label='tpr: 0% -> NLD=0.0')
    text_mock.assert_called_with(0, 0.49999999999999994, 'NLD: 100%=1')
    title_mock.assert_called_with('loanpy.adrc.Adrc.adapt vs Normalised Leveshtein Distance')
    legend_mock.assert_called_with()
    savefig_mock.assert_called_with(path2mockplot)

def test_write_workflow():
    # 3 monkey patches needed.To complicated to mock pandas though

    # set up mock helpers.Etym class
    class MonkeyEtym:
        def __init__(self, *args):
            self.word2struc_called_with = []
            self.called_with = [*args]
        def word2struc(self, *args):
            self.word2struc_called_with.append([*args])
            return "CVCV"

    # set up mock levenshtein distance class
    class MonkeyLD:
        def __init__(self):
            self.ld_returns = iter([0.9, 0.1, 0])
            self.nld_returns = iter([0.8, 0.4, 0])
            self.ld_called_with = []
            self.nld_called_with = []
        def fast_levenshtein_distance(self, *args):
            self.ld_called_with.append([*args])
            return next(self.ld_returns)
        def fast_levenshtein_distance_div_maxlen(self, *args):
            self.nld_called_with.append([*args])
            return next(self.nld_returns)

    #set up the 2 input params of function write_workflow
    #step0: predicted strucutre, step4: actual structure
    workflow_mock = OrderedDict(
    [("target", ["hehe"]), ("source", ["kiki"]), ("sol_idx_plus1", [27]),
    ('tokenised', [['k', 'i', 'k', 'i']]),
    ('donor_struc', ['CVCV']), ('pred_strucs',
    [['CVC', 'CVCCV']]),
    ('adapted_struc', [[['kik'], ['kiCki']]]),
    ('adapted_vowelharmony',
    [[['k', 'B', 'k'], ['k', 'i', 'C', 'k', 'i']]]),
    ('before_combinatorics',
    [[[['k', 'h', 'c'], ['o', 'u'], ['k']],
    [['k'], ['o', 'u'], ['t', 'd'], ['k'], ['e']]]])])

    #expected outcome
    df_exp = DataFrame(workflow_mock).assign(comment=[""],
    struc_predicted=[False], LD_bestguess_target=[0.9], NLD_bestguess_target=[0.8])

    #create instances of monkey class
    monkey_etym, monkey_ld = MonkeyEtym(), MonkeyLD()

    # patch two classes: helpers.Etym and panphon.distance.Distance
    with patch("loanpy.sanity.Etym") as etym_mock:
        etym_mock.return_value = monkey_etym
        with patch("loanpy.sanity.Distance") as distance_mock:
            distance_mock.return_value = monkey_ld

            #assert data frame workflow is returned correctly
            assert_frame_equal(
            write_workflow(workflow_mock, ["kok"], Path()), df_exp)
            #assert same data frame was written correctly
            assert str(read_csv("workflow.csv")) == str(df_exp)

    #assert calls
    #assert call 1: init DataFrame: do later
    #assert call 2: init Etym
    etym_mock.assert_called_with()
    #assert call 3: init Distance
    distance_mock.assert_called_with()
    #assert call 4: word2struc
    assert monkey_etym.word2struc_called_with == [["hehe"]]
    #assert call 5 and 6: the two levenshteins
    assert monkey_ld.ld_called_with == [['kok', 'hehe']]
    assert monkey_ld.nld_called_with == [['kok', 'hehe']]

    #tear down
    remove("workflow.csv")
    del (MonkeyLD, MonkeyEtym, monkey_ld, monkey_etym, df_exp, workflow_mock)

def test_plot_ld_nld():
    pass
































#
