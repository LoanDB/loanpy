from collections import OrderedDict
from datetime import datetime
from os import remove
from pathlib import Path
from time import struct_time
from unittest.mock import call, patch

from numpy import nan
from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from pytest import raises

from loanpy.sanity import (ArgumentsAlreadyTested, check_cache, eval_all,
                           eval_one, get_crossval_sc, gettprfpr, make_stat, plot_roc,
                           write_to_cache, postprocess)
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

def test_get_crossval_sc():
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
    actual_out = get_crossval_sc(adrcmock, 1, None)

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
        guesslist=[2, 4, 6], max_struc=1, max_paths=1,
        deletion_cost=100, insertion_cost=49, vowelharmony=False,
        clusterised=False, sort_by_nse=False, struc_filter=False,
        show_workflow=False, mode="adapt") == {"sol_idx_plus1": float("inf"), "best_guess": "kek"}

        #assert 2 mock calls: get_howmany, adapt
        get_howmany_mock.has_calls = [
        call(2, 1, 1), call(4, 1, 1), call(6, 1, 1)]
        #assert adapt was called thrice, once for each element of guesslist
        assert adrcmock.adapt_called_with == [
        ["kiki", 2, 1, 1, 100, 49, False, False, False, False, False],
        ["kiki", 4, 1, 1, 100, 49, False, False, False, False, False],
        ["kiki", 6, 1, 1, 100, 49, False, False, False, False, False]]

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
        guesslist=[2, 4, 6, 8], max_struc=1, max_paths=1,
        deletion_cost=100, insertion_cost=49, vowelharmony=False,
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
        guesslist=[2, 4, 6, 8], max_struc=1, max_paths=1,
        deletion_cost=100, insertion_cost=49, vowelharmony=False,
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
    init_args_mock = {'formscsv': 'forms.csv', 'tgt_lg': 'EAH',
    'src_lg': 'WOT', 'mode': 'adapt', 'opt_param_path': 'mockpath',
    'crossval': True, 'guesslist': [10, 50, 100, 500, 1000], 'max_struc': 1,
    'deletion_cost': 100, 'insertion_cost': 49,
    'max_paths': 1, 'writesc': False, 'vowelharmony': False,
    'clusterised': False, 'sort_by_nse': False, 'struc_filter': False,
    'show_workflow': False, 'struc_most_frequent': 9999999, 'struc_inv': None,
    'connector': None, 'scdictbase': {'foo': 'bar'}, 'vfb': None,
    'write_to': None, 'plot_to': None, 'plotldnld': False}

    #set up mock df for both cldf2pd AND read_csv
    df_mock_read = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
    "Target_Form": ["hehe", "pupa", "hapa"]})

    #set up expected output data frame
    df_exp = df_mock_read.assign(guesses=[1, None, 3], best_guess=["hehe", "pupa", "hapa"],
    LD_bestguess_TargetForm=[0.9, 0.1, 0.2], NLD_bestguess_TargetForm=[0.8, 0.3, 0.4],
    comment="")

    #set up mock class for var "self", get_crossval_sc will return these
    class GetMonkeySc:
        def __init__(self, scdict=None, scdict_struc=None):
            self.scdict = scdict
            self.scdict_struc = scdict_struc
            self.dfety = df_mock_read
            self.nseiter = iter([0.12, 0.34, 0.56]*100)
            self.get_nse_called_with = []
        def get_nse(self, *args):
            self.get_nse_called_with.append([*args])
        #    print("Called with:", [*args])
        #    print("self.get_nse_called_with:", self.get_nse_called_with)
            return next(self.nseiter)

    #create 3 instances of mock class, to be returned by get_crossval_sc (sdie eff.)
    #one for each element of guesslist
    #simulate the sound changes that would have been extracted from
    #each crossvalidated data frame
    mockgetsc1 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "i" missing b/c "kiki" isolated
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc2 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "u" missing b/c "buba" isolated
                                   "i": ["e"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc3 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"], "i": ["e"],  # nth missing b/c sc in "kaba" are captured in "kiki" and "buba"
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})

    #create instance of mock adrc class
    adrc_monkey = GetMonkeySc()
    adrc_monkey.get_nse("a", "b")
    #set up: mock 8 functions: check_cache, read_forms, get_crossval_sc, eval_one,
    #gettprfpr, make_stat, write_to_cache, time
    with patch("loanpy.sanity.check_cache") as check_cache_mock:
        check_cache_mock.return_value = None
        with patch("loanpy.sanity.Adrc") as Adrc_mock:
            Adrc_mock.return_value = adrc_monkey
            with patch("loanpy.sanity.get_crossval_sc",
            side_effect=[  # 1 scdictlist per crossvalidation round
            mockgetsc1, mockgetsc2, mockgetsc3]) as get_crossval_sc_mock:
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
                                    with patch("loanpy.sanity.postprocess") as postprocess_mock:
                                        postprocess_mock.return_value = df_exp

                                        #assert evaluation runs correctly
                                        assert_frame_equal(eval_all(
                                        formscsv="forms.csv",
                                        tgt_lg="EAH",
                                        src_lg="WOT",
                                        opt_param_path="mockpath",
                                        scdictbase={"foo": "bar"},
                                        ), df_exp)

    #assert calls
    #assert first mock (check_cache) called correctly
    check_cache_mock.assert_called_with(
    "mockpath", init_args_mock)

    #assert 4th mock (get_crossval_sc) called correctly
    #first call get_crossval_sc
    assert type(get_crossval_sc_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert get_crossval_sc_mock.call_args_list[0][0][1:] == (0, False)
    #2nd call get_crossval_sc
    assert type(get_crossval_sc_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert get_crossval_sc_mock.call_args_list[1][0][1:] == (1, False)
    #3rd call get_crossval_sc
    assert type(get_crossval_sc_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert get_crossval_sc_mock.call_args_list[2][0][1:] == (2, False)

    #assert 5th mock (eval_one) called correctly
    #first call
    assert type(eval_one_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[0][0][1:] == (
    ('kiki', 'hehe', [10, 50, 100, 500, 1000], 1, 1, 100, 49,
    False, False, False, False, False, "adapt"))  # kwargs!
    assert eval_one_mock.call_args_list[0][1] == {}  # args!
    #2nd call eval_mock
    assert type(eval_one_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[1][0][1:] == (
    'buba', 'pupa', [10, 50, 100, 500, 1000], 1, 1, 100, 49,
    False, False, False, False, False, "adapt") # kwargs!
    assert eval_one_mock.call_args_list[1][1] == {}  # args!
    #3rd call eval mock
    assert type(eval_one_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[2][0][1:] == (
    "kaba", "hapa", [10, 50, 100, 500, 1000], 1, 1, 100, 49,
    False, False, False, False, False, "adapt")  # kwargs!
    assert eval_one_mock.call_args_list[2][1] == {}  # args!

    #assert 6th mock call was correct
    gettprfpr_mock.assert_called_with([1, None, 3], [10, 50, 100, 500, 1000], 3)

    #assert 7th mock call went correctly
    make_stat_mock.assert_called_with(0.099, 0.6, 1000, 3)  # opt_tp, opt_fp, guesslist[-1], len_df

    #assert 8th mock call went correctly
    write_to_cache_mock.assert_called_with((1, '2/3', '60%'), init_args_mock,
    'mockpath', 20, 22)

    #assert 9th mock call went correctly
    time_mock.assert_called_with()

    # assert postprocess was called correctly
    assert type(postprocess_mock.call_args_list[0][0][0]) == type(GetMonkeySc())

    # assert get_nse was called correctly
    # adrc_monkey gets replaced in the crossval loop by mocksc1, mocksc2, mocksc3
    # mocksc3 is the last one in the loop so thats where get_nse will be called
    assert mockgetsc3.get_nse_called_with == [
    ['kiki', 'hehe'], ['buba', 'pupa'], ['kaba', 'hapa'],
    ['kiki', 'hehe'], ['buba', 'pupa'], ['kaba', 'hapa']]

    del df_exp, df_mock_read, init_args_mock, GetMonkeySc

    #same as before but mode= "reconstruct" opt_param_path is None, and
    # scdictbase is None

    #set up mock init_args (mocks locals())
    init_args_mock = {
    'formscsv': Path('forms.csv'),
    'tgt_lg': 'EAH',
    'src_lg': 'H',
    'mode': 'reconstruct',
    'opt_param_path': Path('etc/opt_params_H_EAH.csv'),
    'crossval': True,
    'guesslist': [1, 2, 3],
    'max_struc': 1,
    'max_paths': 1,
    'deletion_cost': 100,
    'insertion_cost': 49,
    'writesc': False,
    'vowelharmony': False,
    'clusterised': False,
    'sort_by_nse': False,
    'struc_filter': False,
    'show_workflow': False,
    'struc_most_frequent': 9999999,
    'struc_inv': None,
    'connector': None,
    'scdictbase': None,
    'vfb': None,
    'write_to': None,
    'plot_to': None,
    'plotldnld': False}

    #set up mock df for both cldf2pd AND read_csv
    df_mock_read = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
    "Target_Form": ["hehe", "pupa", "hapa"]})

    #set up expected output data frame
    df_exp = df_mock_read.assign(guesses=[1, None, 3], best_guess=["hehe", "pupa", "hapa"],
    LD_bestguess_TargetForm=[0.9, 0.1, 0.2], NLD_bestguess_TargetForm=[0.8, 0.3, 0.4],
    comment="")

    #set up mock class for var "self", get_crossval_sc will return these
    class GetMonkeySc:
        def __init__(self, scdict=None, scdict_struc=None):
            self.scdict = scdict
            self.scdict_struc = scdict_struc
            self.dfety = df_mock_read
            self.nseiter = iter([0.12, 0.34, 0.56]*100)
            self.get_nse_called_with = []
        def get_nse(self, *args):
            self.get_nse_called_with.append([*args])
            return next(self.nseiter)
    #create 3 instances of mock class, to be returned by get_crossval_sc (sdie eff.)
    #one for each element of guesslist
    #simulate the sound changes that would have been extracted from
    #each crossvalidated data frame
    mockgetsc1 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "i" missing b/c "kiki" isolated
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc2 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "u" missing b/c "buba" isolated
                                   "i": ["e"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc3 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"], "i": ["e"],  # nth missing b/c sc in "kaba" are captured in "kiki" and "buba"
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})

    #set up: mock 8 functions: check_cache, read_forms, get_crossval_sc, eval_one,
    #gettprfpr, make_stat, write_to_cache, time
    with patch("loanpy.sanity.check_cache") as check_cache_mock:
        check_cache_mock.return_value = None
        with patch("loanpy.sanity.Adrc") as Adrc_mock:
            Adrc_mock.return_value = GetMonkeySc()
            with patch("loanpy.sanity.get_crossval_sc",
            side_effect=[  # 1 scdictlist per crossvalidation round
            mockgetsc1, mockgetsc2, mockgetsc3]) as get_crossval_sc_mock:
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
                                    with patch("loanpy.sanity.postprocess") as postprocess_mock:
                                        postprocess_mock.return_value = df_exp
                                        #assert evaluation runs correctly
                                        assert_frame_equal(eval_all(
                                        formscsv=Path("forms.csv"),
                                        tgt_lg="EAH",
                                        src_lg="H",
                                        guesslist=[1,2,3],
                                        mode="reconstruct",
                                        opt_param_path=None,
                                        ), df_exp)

    #assert calls

    #assert first mock (check_cache) called correctly
    check_cache_mock.assert_called_with(
    Path("etc/opt_params_H_EAH.csv"), init_args_mock)

    #assert 4th mock (get_crossval_sc) called correctly
    #first call get_crossval_sc
    assert type(get_crossval_sc_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert get_crossval_sc_mock.call_args_list[0][0][1:] == (0, False)
    #2nd call get_crossval_sc
    assert type(get_crossval_sc_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert get_crossval_sc_mock.call_args_list[1][0][1:] == (1, False)
    #3rd call get_crossval_sc
    assert type(get_crossval_sc_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert get_crossval_sc_mock.call_args_list[2][0][1:] == (2, False)

    #assert 5th mock (eval_one) called correctly
    #first call
    assert type(eval_one_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[0][0][1:] == (
    ('kiki', 'hehe', [1, 2, 3], 1, 1, 100, 49,
    False, False, False, False, False, "reconstruct"))  # kwargs!
    assert eval_one_mock.call_args_list[0][1] == {}  # args!
    #2nd call eval_mock
    assert type(eval_one_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[1][0][1:] == (
    'buba', 'pupa', [1, 2, 3], 1, 1, 100, 49,
    False, False, False, False, False, "reconstruct") # kwargs!
    assert eval_one_mock.call_args_list[1][1] == {}  # args!
    #3rd call eval mock
    assert type(eval_one_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[2][0][1:] == (
    "kaba", "hapa", [1, 2, 3], 1, 1, 100, 49,
    False, False, False, False, False, "reconstruct")  # kwargs!
    assert eval_one_mock.call_args_list[2][1] == {}  # args!

    #assert 6th mock call was correct
    gettprfpr_mock.assert_called_with([1, None, 3], [1, 2, 3], 3)

    #assert 7th mock call went correctly
    make_stat_mock.assert_called_with(0.099, 0.6, 3, 3)  # opt_tp, opt_fp, guesslist[-1], len_df

    #assert 8th mock call went correctly
    write_to_cache_mock.assert_called_with((1, '2/3', '60%'),
    init_args_mock, Path('etc/opt_params_H_EAH.csv'), 20, 22)

    #assert 9th mock call went correctly
    time_mock.assert_called_with()

    # assert postprocess was called correctly
    assert type(postprocess_mock.call_args_list[0][0][0]) == type(GetMonkeySc())

    assert mockgetsc3.get_nse_called_with == [
    ['kiki', 'hehe'], ['buba', 'pupa'], ['kaba', 'hapa'],
    ['kiki', 'hehe'], ['buba', 'pupa'], ['kaba', 'hapa']]

    # assert that by default cache is not read or written (otherwise same as above)

    mockgetsc1 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "i" missing b/c "kiki" isolated
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc2 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"],  # "u" missing b/c "buba" isolated
                                   "i": ["e"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})
    mockgetsc3 = GetMonkeySc(scdict={"k": ["h"], "b": ["p"], "i": ["e"],  # nth missing b/c sc in "kaba" are captured in "kiki" and "buba"
                                   "u": ["u"], "a": ["a"]}, scdict_struc={"CVCV": ["CVCV"]})

    #set up: mock 8 functions: check_cache, read_forms, get_crossval_sc, eval_one,
    #gettprfpr, make_stat, write_to_cache, time
    with patch("loanpy.sanity.check_cache") as check_cache_mock:
        check_cache_mock.return_value = None
        with patch("loanpy.sanity.Adrc") as Adrc_mock:
            Adrc_mock.return_value = GetMonkeySc()
            with patch("loanpy.sanity.get_crossval_sc",
            side_effect=[  # 1 scdictlist per crossvalidation round
            mockgetsc1, mockgetsc2, mockgetsc3]) as get_crossval_sc_mock:
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
                                    with patch("loanpy.sanity.postprocess") as postprocess_mock:
                                        postprocess_mock.return_value = df_exp

                                        #assert evaluation runs correctly
                                        assert_frame_equal(eval_all(
                                        formscsv=Path("forms.csv"),
                                        tgt_lg="EAH",
                                        src_lg="H",
                                        guesslist=[1,2,3],
                                        mode="reconstruct",
                                        #default: opt_param_path=False,
                                        ), df_exp)

    #assert calls
    # only assert that cache is neither read nor written by default
    check_cache_mock.assert_not_called()
    write_to_cache_mock.assert_not_called()

    del df_exp, df_mock_read, init_args_mock, GetMonkeySc

    #same as first one but different params

    #set up path to write sound changes to
    writesc_mockpath = Path(__file__).parent / "eval_all_test_sc.txt"
    #set up path to write final output to
    path2output_test_eval_all = Path(__file__).parent / 'eval_all_test_output.csv'
    #set up path to write plot to
    path2plot_test_eval_all = Path(__file__).parent / "\
integration" / "output_files" / 'eval_all_plot.jpg'

    #set up mock init_args (mocks locals())

    init_args_mock = {
    'formscsv': 'forms.csv',
    'tgt_lg': 'EAH',
    'src_lg': 'WOT',
    'mode': 'adapt',
    'opt_param_path': Path('mockpath'),
    'crossval': False,
    'guesslist': [10, 50, 100, 500, 1000],
    'max_struc': 2,
    'max_paths': 3,
    'deletion_cost': 100,
    'insertion_cost': 49,
    'writesc': writesc_mockpath,
    'vowelharmony': True,
    'clusterised': True,
    'sort_by_nse': True,
    'struc_filter': True,
    'show_workflow': True,
    'struc_most_frequent': 9999999,
    'struc_inv': None,
    'connector': None,
    'scdictbase': None,
    'vfb': None,
    'mode': 'adapt',
    'write_to': path2output_test_eval_all,
    'plot_to': path2plot_test_eval_all,
    'plotldnld': False}

    #set up mock df for both cldf2pd AND read_csv
    df_mock_read = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
    "Target_Form": ["hehe", "pupa", "hapa"]})

    #set up expected output data frame
    df_exp_postpro = df_mock_read.assign(guesses=[1, None, 3], best_guess=["hehe", "pupa", "hapa"],
    LD_bestguess_TargetForm=[0.9, 0.1, 0.2], NLD_bestguess_TargetForm=[0.8, 0.3, 0.4],
    comment="")

    df_exp_concat = df_exp_postpro.assign(comment="", tokenised=[None]*3,
    adapted_struc=[None]*3, adapted_vowelharmony=[None]*3,
    before_combinatorics=[None]*3, donor_struc=[None]*3, pred_strucs=[""]*3)

    #set up mock class for var "self", get_crossval_sc will return these
    class GetMonkeySc:
        def __init__(self, scdict=None, scdict_struc=None):
            self.scdict = scdict
            self.scdict_struc = scdict_struc
            self.dfety = df_mock_read
            self.get_sound_corresp_called_with = []
            self.nseiter = iter([0.12, 0.34, 0.56]*100)
            self.get_nse_called_with = []
        def get_nse(self, *args):
            self.get_nse_called_with.append([*args])
            return next(self.nseiter)
        def get_sound_corresp(self, *args):
            self.get_sound_corresp_called_with.append([*args])
            out = [{"k": ["h"], "b": ["p"], "i": ["e"],
                     "u": ["u"], "a": ["a"]}, {}, {},
                     {"CVCV": ["CVCV"]}, {}, {}]
            with open(writesc_mockpath, "w") as f:
                    f.write(str(out))
            return out

    # set up mock helpers.Etym class
    class MonkeyEtym:
        def __init__(self, *args):
            self.word2struc_called_with = []
            self.called_with = [*args]
        def word2struc(self, *args):
            self.word2struc_called_with.append([*args])
            return "CVCV"

    #create instance of monkey class
    monkey_adrc, monkey_etym = GetMonkeySc(), MonkeyEtym()
    #set up: mock 8 functions: check_cache, read_forms, get_crossval_sc, eval_one,
    #gettprfpr, make_stat, write_to_cache, time
    with patch("loanpy.sanity.check_cache") as check_cache_mock:
        check_cache_mock.return_value = None
        with patch("loanpy.sanity.Adrc") as Adrc_mock:
            Adrc_mock.return_value = monkey_adrc
            with patch("loanpy.sanity.get_crossval_sc") as get_crossval_sc_mock:
                # won't get called b/c crossval=False
                with patch("loanpy.sanity.eval_one",
                side_effect=[{"sol_idx_plus1": 1, "best_guess": "hehe",
                              "workflow": {"stepX": "whathappened"}},
                             {"sol_idx_plus1": None, "best_guess": "pupa",
                              "workflow": {"stepX": "whathappened"}},
                             {"sol_idx_plus1": 3, "best_guess": "hapa",
                              "workflow": {"stepX": "whathappened"}}]) as eval_one_mock:
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
                                    with patch("loanpy.sanity.postprocess") as postprocess_mock:
                                        postprocess_mock.return_value = df_exp_postpro
                                        with patch("loanpy.sanity.concat") as concat_mock:
                                            concat_mock.return_value = df_exp_concat
                                            with patch("loanpy.sanity.Etym") as etym_mock:
                                                etym_mock.return_value = monkey_etym
                                                #assert evaluation runs correctly
                                                assert_frame_equal(eval_all(
                                                formscsv="forms.csv",
                                                tgt_lg="EAH",
                                                src_lg="WOT",
                                                opt_param_path=Path("mockpath"),
                                                crossval=False,
                                                max_struc=2,
                                                max_paths=3,
                                                writesc=writesc_mockpath,
                                                vowelharmony=True,
                                                clusterised=True,
                                                sort_by_nse=True,
                                                struc_filter=True,
                                                show_workflow=True,
                                                write_to=path2output_test_eval_all,
                                                plot_to=path2plot_test_eval_all,
                                                ), df_exp_concat)

    #assert calls

    #assert first mock (check_cache) called correctly
    check_cache_mock.assert_called_with(
    Path("mockpath"), init_args_mock)

    #assert get_crossval_sc_mock was not called
    get_crossval_sc_mock.assert_not_called()

    #assert get_sound_corresp was called correctly
    assert monkey_adrc.get_sound_corresp_called_with == [[writesc_mockpath]]

    #assert 5th mock (eval_one) called correctly
    #first call
    assert type(eval_one_mock.call_args_list[0][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[0][0][1:] == (
    ('kiki', 'hehe', [10, 50, 100, 500, 1000], 2, 3, 100, 49,
    True, True, True, True, True, "adapt"))  # kwargs!
    assert eval_one_mock.call_args_list[0][1] == {}  # args!
    #2nd call eval_mock
    assert type(eval_one_mock.call_args_list[1][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[1][0][1:] == (
    'buba', 'pupa', [10, 50, 100, 500, 1000], 2, 3, 100, 49,
    True, True, True, True, True, "adapt") # kwargs!
    assert eval_one_mock.call_args_list[1][1] == {}  # args!
    #3rd call eval mock
    assert type(eval_one_mock.call_args_list[2][0][0]) == type(GetMonkeySc())
    assert eval_one_mock.call_args_list[2][0][1:] == (
    "kaba", "hapa", [10, 50, 100, 500, 1000], 2, 3, 100, 49,
    True, True, True, True, True, "adapt")  # kwargs!
    assert eval_one_mock.call_args_list[2][1] == {}  # args!

    #assert 6th mock call was correct
    gettprfpr_mock.assert_called_with([1, None, 3], [10, 50, 100, 500, 1000], 3)

    #assert 7th mock call went correctly
    make_stat_mock.assert_called_with(0.099, 0.6, 1000, 3)  # opt_tp, opt_fp, guesslist[-1], len_df

    #assert 8th mock call went correctly
    write_to_cache_mock.assert_called_with((1, '2/3', '60%'), init_args_mock,
    Path('mockpath'), 20, 22)

    #assert 9th mock call went correctly
    time_mock.assert_called_with()

    # assert postprocess was called correctly
    assert type(postprocess_mock.call_args_list[0][0][0]) == type(GetMonkeySc())

    # assert get_nse was called correctly
    # now, monkey_adrc got not replaced with mocksc3 b/c crossval=False
    assert monkey_adrc.get_nse_called_with == [
    ['kiki', 'hehe'], ['buba', 'pupa'], ['kaba', 'hapa'], ['kiki', 'hehe'],
    ['buba', 'pupa'], ['kaba', 'hapa']]

    # assert concat was called correctly
    assert_frame_equal(concat_mock.call_args_list[0][0][0][0], DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"], "Target_Form": ["hehe", "pupa", "hapa"],
    "guesses": [1.0, nan, 3.0],
    "best_guess": ["hehe", "pupa", "hapa"], "LD_bestguess_TargetForm": [0.9, 0.1, 0.2],
    "NLD_bestguess_TargetForm": [0.8, 0.3, 0.4], "comment": [""]*3}))
    assert_frame_equal(concat_mock.call_args_list[0][0][0][1], DataFrame(
    {"tokenised": [None]*3, "adapted_struc": [None]*3,
    "adapted_vowelharmony": [None]*3,
    "before_combinatorics": [None]*3, "donor_struc": [None]*3,
    "pred_strucs": [""]*3}))

    # assert etym was called correctly
    assert monkey_etym.word2struc_called_with == [['hehe'], ['pupa'], ['hapa']]

    remove(writesc_mockpath)
    remove(path2output_test_eval_all)

    del (df_exp_concat, df_exp_postpro, df_mock_read, init_args_mock,
    monkey_adrc, GetMonkeySc, etym_mock)


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

def test_postprocess():
    #set up mock class for input and instantiate it
    class AdrcMonkey:
        def __init__(self):
            self.dfety = DataFrame({"best_guess": ["apple"], "Target_Form": ["banana"]})

    # set up mock levenshtein distance class
    class MonkeyLD:
        def __init__(self):
            self.ld_returns = iter([0.9])
            self.nld_returns = iter([0.8])
            self.ld_called_with = []
            self.nld_called_with = []
        def fast_levenshtein_distance(self, *args):
            self.ld_called_with.append([*args])
            return next(self.ld_returns)
        def fast_levenshtein_distance_div_maxlen(self, *args):
            self.nld_called_with.append([*args])
            return next(self.nld_returns)

    # create two monkey instances
    monkey_ld, monkey_adrc = MonkeyLD(), AdrcMonkey()

    #expected outcome
    df_exp = DataFrame(monkey_adrc.dfety).assign(
    LD_bestguess_TargetForm=[0.9], NLD_bestguess_TargetForm=[0.8],
    comment=[""])

    # patch panphon.distance.Distance
    with patch("loanpy.sanity.Distance") as distance_mock:
        distance_mock.return_value = monkey_ld

        #assert data frame workflow is returned correctly
        assert_frame_equal(postprocess(monkey_adrc), df_exp)

    #assert call init Distance
    distance_mock.assert_called_with()
    #assert call for two levenshteins
    assert monkey_ld.ld_called_with == [['apple', 'banana']]
    assert monkey_ld.nld_called_with == [['apple', 'banana']]

    #tear down
    del (MonkeyLD, AdrcMonkey, monkey_ld, monkey_adrc, df_exp)

def test_plot_ld_nld():
    pass
































#
