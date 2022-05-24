from ast import literal_eval
from collections import OrderedDict
from datetime import datetime
from os import remove
from pathlib import Path

from pandas import DataFrame, RangeIndex, read_csv
from pandas.testing import assert_frame_equal
from pytest import raises

from loanpy.adrc import Adrc
from loanpy.helpers import Etym
from loanpy.sanity import (
ArgumentsAlreadyTested,
check_cache,
get_crossval_sc,
eval_one,
eval_all,
plot_roc,
write_to_cache,
postprocess)

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"
PATH2SC_TEST = Path(__file__).parent / "input_files" / "sc_ad_3cogs.txt"
MOCK_CACHE_PATH = Path(__file__).parent / "mock_cache.csv"
PATH2MOCKWORKFLOW = Path(__file__).parent / "workflow.csv"
PATH2OUT = Path(__file__).parent / "output_files" # just a folder!
PATH2EXP = Path(__file__).parent / "expected_files" # just a folder!

def test_make_cache():
    """test if DIY cache is initiated correctly and args checked in it"""

    #make sure this file does not exist (e.g. from previous tests)
    try: remove(MOCK_CACHE_PATH)
    except FileNotFoundError: pass
    #set up first expected outcome, a pandas data frame
    exp1 = DataFrame(columns=["arg1", "arg2", "arg3", "opt_tpr",
    "optimal_howmany", "opt_tp", "timing", "date"])

    #assert first break works: cache not found
    check_cache(MOCK_CACHE_PATH, {"arg1": "x", "arg2": "y", "arg3": "z"})
    assert_frame_equal(read_csv(MOCK_CACHE_PATH), exp1)

    #check if nothing happens if arguments were NOT tested already
    #assert that the function runs, does nothing, and returns None
    assert check_cache(MOCK_CACHE_PATH,
    {"arg1": "a", "arg2": "b", "arg3": "c"}) is None

    #tear down
    remove(MOCK_CACHE_PATH)

    #check if exception is rased if these params were tested already

    #set up mock cache with stored args
    DataFrame({"arg1": ["x"], "arg2": ["y"],
    "arg3": ["z"]}).to_csv(MOCK_CACHE_PATH, encoding="utf-8", index=False)
    # assert exception is raised bc args exist in cache already
    with raises(ArgumentsAlreadyTested) as aat_mock:
        check_cache(MOCK_CACHE_PATH,
        {"arg1": "x", "arg2": "y", "arg3": "z"})
    assert str(aat_mock.value) == f"These arguments were tested \
already, see {MOCK_CACHE_PATH} line 1! (start counting at 1 in 1st row)"

    #tear down
    remove(MOCK_CACHE_PATH)

def test_get_crossval_sc():
    #create instance of Adrc class for input
    adrc_obj = Adrc(formscsv=PATH2FORMS, srclg="WOT", tgtlg="EAH")
    #set up actual output as variable
    out = get_crossval_sc(adrc_obj, 1, False)
    assert out.scdict == { #'l': ['l'] and 'l': ['d'] are missing bc got isolated
     'a': ['a'],
     'j': ['j'],
     'n': ['n'],
     't͡ʃː': ['t͡ʃ'],
     'ɣ': ['ɣ'],
     'ɯ': ['i']}

    exp_dict = {
    'VCVC': ['VCVC', 'VCCVC', 'VCVCV'], 'VCVCV': ['VCVCV', 'VCVC', 'VCCVC']}

    #VCCVC and VCVCV have the same distance from VCVC: 0.49 (one insertion)
    #so rank_closest_struc sometimes ranks VCCVC first sometimes VCVCV
    #so we have to compare the sets of the dictionary values
    for struc, exp in zip(out.scdict_struc, exp_dict):
        assert set(out.scdict_struc[struc]) == set(exp_dict[exp])

    #isolate word number 2 now
    adrc_obj = Adrc(formscsv=PATH2FORMS, srclg="WOT", tgtlg="EAH")
    out = get_crossval_sc(adrc_obj, 2, None)
    assert out.scdict == { #'l': ['l'] and 'l': ['d'] are missing bc got isolated
     'a': ['a'],
     'd': ['d'],
     'l': ['l'],
     't͡ʃː': ['t͡ʃ'],
     'ɣ': ['ɣ'],
     'ɯ': ['i']}

    exp_dict = {
    'VCCVC': ['VCCVC', 'VCVC', 'VCVCV'], 'VCVCV': ['VCVCV', 'VCVC', 'VCCVC']}

    #VCCVC and VCVCV have the same distance from VCVC: 0.49 (one insertion)
    #so rank_closest_struc sometimes ranks VCCVC first sometimes VCVCV
    #so we have to compare the sets of the dictionary values
    for struc, exp in zip(out.scdict_struc, exp_dict):
        assert set(out.scdict_struc[struc]) == set(exp_dict[exp])

def test_eval_one():

    #assert None is returned if target word is not in the predictions
    #create instance of Adrc class for input
    adrc_obj = Adrc(formscsv=PATH2FORMS, srclg="WOT", tgtlg="EAH",
                    scdictlist=PATH2SC_TEST)

    eval_one(adrc_obj=adrc_obj, srcwrd="dada", tgtwrd="gaga",
    guesslist=[2, 4, 6], max_struc=1, max_paths=1,
    deletion_cost=100, insertion_cost=49, vowelharmony=False,
    clusterised=False, sort_by_nse=True, struc_filter=False,
    show_workflow=False, mode="adapt") == {"sol_idx_plus1": float("inf"), "best_guess": "dada"}

    #assert list index is returned if target word was in predictions
    #assert list index+1 is returned if target word was in predictions
    assert eval_one(adrc_obj=adrc_obj, srcwrd="dada", tgtwrd="dada",
    guesslist=[2, 4, 6, 8], max_struc=1, max_paths=1,
    deletion_cost=100, insertion_cost=49, vowelharmony=False,
    clusterised=False, sort_by_nse=False, struc_filter=False,
    show_workflow=False, mode="adapt") == {'best_guess': 'dada', 'sol_idx_plus1': 1}

    #assert workflow is returned correctly
    #assert list index+1 is returned if target word was in predictions
    assert eval_one(adrc_obj=adrc_obj, srcwrd="dada", tgtwrd="dada",
    guesslist=[2, 4, 6, 8], max_struc=1, max_paths=1,
    deletion_cost=100, insertion_cost=49, vowelharmony=False,
    clusterised=False, sort_by_nse=False, struc_filter=False,
    show_workflow=True, mode="adapt") == {'best_guess': 'dada', 'sol_idx_plus1': 1,
    'workflow': OrderedDict([(
    'tokenised', "['d', 'a', 'd', 'a']"),
    ('adapted_struc', "[['d', 'a', 'd', 'a']]"),
    ('before_combinatorics', "[[['d'], ['a'], ['d'], ['a']]]")])}

def test_make_stat():
    pass # unittest = integrationtest, there was nothing to mock.

def test_gettprfpr():
    pass # unittest = integrationtest, there was nothing to mock.

def test_write_to_cache():

    init_args_mock = {"formscsv": "forms.csv", "tgt_lg": "EAH",
    "src_lg": "WOT", "crossval": True,
    "opt_param_path": MOCK_CACHE_PATH, "guesslist": [[2, 4, 6, 8]],
    "max_struc": 1, "max_paths": 1, "writesc": False,
    "writesc_struc": False, "vowelharmony": False,
    "only_documented_clusters": False, "sort_by_nse": False,
    "struc_filter": False, "show_workflow": False, "write": False,
    "outname": "viz", "plot_to": None, "plotldnld": False}

    DataFrame(columns=list(init_args_mock)+["optimal_howmany", "opt_tp", "opt_tpr", "timing", "date"]
    ).to_csv(MOCK_CACHE_PATH, index=False, encoding="utf-8") #empty cache

    df_exp = DataFrame(
    {"formscsv": "forms.csv", "tgt_lg": "EAH",
    "src_lg": "WOT", "crossval": True,
    "opt_param_path": str(MOCK_CACHE_PATH), "guesslist": str([[2, 4, 6, 8]]),
    "max_struc": 1, "max_paths": 1, "writesc": False,
    "writesc_struc": False, "vowelharmony": False,
    "only_documented_clusters": False, "sort_by_nse": False,
    "struc_filter": False, "show_workflow": False, "write": False,
    "outname": "viz", "plot_to": "None", "plotldnld": False,
    "optimal_howmany": 0.501, "opt_tp": 0.6, "opt_tpr": 0.099, "timing": "00:00:01",
    "date": datetime.now().strftime("%x %X")},
    index=RangeIndex(start=0, stop=1, step=1))

    #write to mock cache
    write_to_cache(
    stat=(0.501, 0.6, 0.099),
    init_args=init_args_mock,
    opt_param_path=MOCK_CACHE_PATH, start=1, end=2)

    #assert cache was written correctly
    assert_frame_equal(read_csv(MOCK_CACHE_PATH), df_exp, check_dtype=False)

    #assert sort functions correctly

    df_exp = DataFrame(
        {"formscsv": ["forms.csv"]*2, "tgt_lg": ["EAH"]*2,
        "src_lg": ["WOT"]*2, "crossval": [True]*2,
        "opt_param_path": [str(MOCK_CACHE_PATH)]*2, "guesslist": [str([[2, 4, 6, 8]])]*2,
        "max_struc": [1]*2, "max_paths": [1]*2, "writesc": [False]*2,
        "writesc_struc": [False]*2, "vowelharmony": [False]*2,
        "only_documented_clusters": [False]*2, "sort_by_nse": [False]*2,
        "struc_filter": [False]*2, "show_workflow": [False]*2, "write": [False]*2,
        "outname": ["viz"]*2, "plot_to": ["None"]*2, "plotldnld": [False]*2,
        "optimal_howmany": [0.501]*2, "opt_tp": [0.6, 0.6], "opt_tpr": [0.8, 0.099],
        "timing": ["00:00:01"]*2, "date": [datetime.now().strftime("%x %X")]*2})

    #write to mock cache
    write_to_cache(
    stat=(0.501, 0.6, 0.8),
    init_args=init_args_mock,
    opt_param_path=MOCK_CACHE_PATH, start=1, end=2)

    #assert cache was written an sorted correctly
    assert_frame_equal(read_csv(MOCK_CACHE_PATH), df_exp, check_dtype=False)

    remove(MOCK_CACHE_PATH)
    del df_exp, init_args_mock

def test_eval_all():
    #set up expected outcome

    #remove cache if it didn't get deleted in previous tests
    try: remove(MOCK_CACHE_PATH)
    except FileNotFoundError: pass

    df_exp = DataFrame(
    {"Target_Form": ["aɣat͡ʃi", "aldaɣ", "ajan"],
     "Source_Form": ["aɣat͡ʃːɯ", "aldaɣ", "ajan"],
     "Cognacy": [1, 2, 3],
     "target_nse": [2.0, 2.4, 2.0],
     "guesses": [float("inf")]*3,
     "best_guess": ["KeyError"]*3,
     "bestguess_nse": [0, 0, 0],
     'LD_bestguess_TargetForm': [float("inf")]*3,
     'NLD_bestguess_TargetForm': [float("inf")]*3,
     'comment': [""]*3})

    #check based on those 3 cognates that it can't predict anything
    assert_frame_equal(eval_all(
    opt_param_path=MOCK_CACHE_PATH,
    formscsv=PATH2FORMS,
    tgt_lg="EAH",
    src_lg="WOT",
    crossval=True,

    guesslist=[1, 2, 3],
    max_struc=1,
    max_paths=1,
    writesc=False,
    vowelharmony=False,
    clusterised=False,
    sort_by_nse=False,
    struc_filter=False,
    show_workflow=False,

    scdictbase=None,
    mode="adapt",

    write_to=None,
    plot_to=None,
    plotldnld=False
    ), df_exp, check_dtype=False)

    remove(MOCK_CACHE_PATH)

    #now check based on 27 cognates how much it could predict already
    path2cog27 = Path(__file__).parent / "input_files" / "forms_27cogs.csv"

    df_exp = DataFrame(
    {"Target_Form": ['aɣat͡ʃi', 'aldaɣ', 'ajan', 'aːl', 'alat͡ʃ', 'alat͡ʃ', 'alat͡ʃ', 'alat͡ʃ', 'alma', 'altalaɡ', 'altalaɡ', 'op', 'oporo', 'opuruɣ', 'orat', 'orat', 'aːr', 'aːrtat', 'arkan', 'aːruk', 'arpa', 'aritan', 'aski'],
     "Source_Form": ['aɣat͡ʃt͡ʃɯ', 'aldaɣ', 'ajan', 'al', 'alat͡ʃɯ', 'alat͡ʃu', 'alat͡ʃo', 'ɒlɒt͡ʃ', 'alma', 'altɯlɯɡ', 'altɯlɯɡ', 'op', 'opura', 'opuruɣ', 'orat', 'or', 'ar', 'artat', 'arkan', 'aruk', 'arpa', 'arɯtan', 'askɯ'],
     "Cognacy": [1, 2, 3] + list(range(8, 28)), #4,5,6,7 ist dropped bc has no EAH
     "target_nse": [10.2, 12.6, 13.0, 5.5, 12.6, 12.6, 12.6, 3.75, 15.0, 7.71, 7.71, 4.5, 3.8, 4.67, 11.25, 3.75, 5.0, 9.2, 12.6, 3.75, 15.5, 11.17, 6.5],
     "guesses": [float("inf"), float("inf"), float("inf"), 2.0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 1.0, float("inf"), float("inf"), 1.0, float("inf"), 2.0, 2.0, 1.0, 3.0, 1.0, float("inf"), float("inf")],
     "best_guess": ['aɣat͡ʃt͡ʃa', "KeyError", "KeyError", 'aːl', 'alat͡ʃa', 'alat͡ʃu', 'alat͡ʃo', "KeyError", "KeyError", 'altiliɡ', 'altiliɡ', 'op', 'opura', 'oprɣ', 'orat', 'or', 'aːr', 'aːrtat', 'arkan', 'aːruk', 'arpa', 'aratan', "KeyError"],
     "bestguess_nse": [10.2, 0.0, 0.0, 5.5, 12.6, 13.2, 13.6, 0.0, 0.0, 7.71, 7.71, 4.5, 9.2, 3.67, 11.25, 7.5, 5.0, 9.2, 12.6, 3.75, 15.5, 11.17, 0.0],
     'LD_bestguess_TargetForm': [4.0, float("inf"), float("inf"), 0.0, 1.0, 1.0, 1.0, float("inf"), float("inf"), 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float("inf")],
     'NLD_bestguess_TargetForm': [0.4, float("inf"), float("inf"), 0.0, 0.14, 0.14, 0.14, float("inf"), float("inf"), 0.29, 0.29, 0.0, 0.4, 0.33, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17, float("inf")],
     'comment': [""]*23})

    assert_frame_equal(eval_all(
    opt_param_path=MOCK_CACHE_PATH,
    formscsv=path2cog27,
    tgt_lg="EAH",
    src_lg="WOT",
    crossval=True,

    guesslist=[1, 2, 3],
    max_struc=1,
    max_paths=1,
    writesc=False,
    vowelharmony=False,
    clusterised=False,
    sort_by_nse=True,
    struc_filter=False,
    show_workflow=False,

    write_to=None,
    plot_to=None,
    plotldnld=False
    ), df_exp, check_dtype=False)

    remove(MOCK_CACHE_PATH)

    df_exp = DataFrame(
    {"Target_Form": ['aɣat͡ʃi', 'aldaɣ', 'ajan', 'aːl', 'alat͡ʃ', 'alat͡ʃ', 'alat͡ʃ', 'alat͡ʃ', 'alma', 'altalaɡ', 'altalaɡ', 'op', 'oporo', 'opuruɣ', 'orat', 'orat', 'aːr', 'aːrtat', 'arkan', 'aːruk', 'arpa', 'aritan', 'aski'],
     "Source_Form": ['aɣat͡ʃt͡ʃɯ', 'aldaɣ', 'ajan', 'al', 'alat͡ʃɯ', 'alat͡ʃu', 'alat͡ʃo', 'ɒlɒt͡ʃ', 'alma', 'altɯlɯɡ', 'altɯlɯɡ', 'op', 'opura', 'opuruɣ', 'orat', 'or', 'ar', 'artat', 'arkan', 'aruk', 'arpa', 'arɯtan', 'askɯ'],
     "Cognacy": [1, 2, 3] + list(range(8, 28)), #4,5,6,7 ist dropped bc has no EAH
     "target_nse": [10.2, 12.6, 13.0, 5.5, 12.6, 12.6, 12.6, 3.75, 15.0, 7.71, 7.71, 4.5, 3.8, 4.67, 11.25, 3.75, 5.0, 9.2, 12.6, 3.75, 15.5, 11.17, 6.5],
     "guesses": [float("inf"), float("inf"), float("inf"), 2.0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 4.0, 4.0, 1.0, float("inf"), float("inf"), 1.0, float("inf"), 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, float("inf")],
     "best_guess": ['aɣat͡ʃt͡ʃa', "KeyError", "KeyError", 'aːl', 'alat͡ʃa', 'alat͡ʃu', 'alat͡ʃo', "KeyError", "KeyError", 'altalaɡ', 'altalaɡ', 'op', 'opura', 'opurɣ', 'orat', 'or', 'aːr', 'aːrtat', 'arkan', 'aːruk', 'arpa', 'aritan', "KeyError"],
     "bestguess_nse": [10.2, 0.0, 0.0, 5.5, 12.6, 13.2, 13.6, 0.0, 0.0, 7.71, 7.71, 4.5, 9.2, 4.17, 11.25, 7.5, 5.0, 9.2, 12.6, 3.75, 15.5, 11.17, 0.0],
     'LD_bestguess_TargetForm': [4.0, float("inf"), float("inf"), 0.0, 1.0, 1.0, 1.0, float("inf"), float("inf"), 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf")],
     'NLD_bestguess_TargetForm': [0.4, float("inf"), float("inf"), 0.0, 0.14, 0.14, 0.14, float("inf"), float("inf"), 0.0, 0.0, 0.0, 0.4, 0.17, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf")],
     'comment': [""]*23})

    #now stick to the 27 cognates but try a longer guesslist
    assert_frame_equal(eval_all(
    opt_param_path=MOCK_CACHE_PATH,
    formscsv=path2cog27,
    tgt_lg="EAH",
    src_lg="WOT",
    crossval=True,

    guesslist=[1, 2, 3, 5, 10],
    max_struc=1,
    max_paths=1,
    writesc=False,
    vowelharmony=False,
    clusterised=False,
    sort_by_nse=True,
    struc_filter=False,
    show_workflow=False,

    scdictbase=None,
    mode="adapt",

    write_to=None,
    plot_to=None,
    plotldnld=False
    ), df_exp, check_dtype=False)

    remove(MOCK_CACHE_PATH)

    #now go wild with params and see what happens

    path2test_out_eval_all = Path(__file__).parent / "test_out_eval_all.csv"

    #generate scdictbase
    scb = Etym(formscsv=PATH2FORMS, tgtlg="EAH"
    ).get_scdictbase() # takes up 1.6MB. Better to generate and throw out

    df_exp = read_csv(
    Path(__file__).parent / "expected_files" / "eval_all_exp.csv").fillna("")
    #for col in ["tokenised", "adapted_struc"]:

    sol = eval_all(
    opt_param_path=MOCK_CACHE_PATH,
    formscsv=path2cog27,
    tgt_lg="EAH",
    src_lg="WOT",
    crossval=True,

    guesslist=[1, 5, 10, 100, 1000],
    max_struc=2,
    max_paths=3,
    writesc=None, # just a folder!
    vowelharmony=True,
    clusterised=True,
    sort_by_nse=True,
    struc_filter=True,
    show_workflow=True,

    scdictbase=scb, #rankclosest ranks equally similar sounds randomly!
    #causing the results in fp and best_guess to be random as well
    write_to=path2test_out_eval_all,
    plot_to=None,
    plotldnld=False)
    sol.to_csv("solyushan.csv", index=False, encoding="utf-8")

    #values of fp and best_guess always vary that's why cant assert them.
    assert all(isinstance(i, float) for i in sol["guesses"])
    assert all(isinstance(i, float) for i in sol["LD_bestguess_TargetForm"])
    assert all(isinstance(i, float) for i in sol["NLD_bestguess_TargetForm"])
    assert all(isinstance(i, float) for i in sol["bestguess_nse"])
    assert all(isinstance(i, str) for i in sol["best_guess"])
    assert all(isinstance(literal_eval(i), list) for i in sol["adapted_struc"])
    assert all(isinstance(literal_eval(i), list) for i in sol["adapted_vowelharmony"])
    assert all(isinstance(literal_eval(i), list) for i in sol["before_combinatorics"])
    assert all(i=="" or isinstance(literal_eval(i), list) for i in sol["pred_strucs"])
    assert all(isinstance(i, str) or i is None for i in sol["donor_struc"])
    del (sol["guesses"], sol["best_guess"], sol["LD_bestguess_TargetForm"],
    sol["NLD_bestguess_TargetForm"], sol["adapted_struc"], sol["adapted_vowelharmony"],
    sol["before_combinatorics"], sol["donor_struc"], sol["pred_strucs"],
    sol["bestguess_nse"],
    df_exp["guesses"], df_exp["best_guess"], df_exp["LD_bestguess_TargetForm"],
    df_exp["NLD_bestguess_TargetForm"], df_exp["adapted_struc"],
    df_exp["adapted_vowelharmony"], df_exp["before_combinatorics"],
    df_exp["donor_struc"], df_exp["pred_strucs"], df_exp["bestguess_nse"])
    assert_frame_equal(sol, df_exp, check_dtype=False)

    #assert file was written correctly
    sol = read_csv(path2test_out_eval_all).fillna("")
    assert all(isinstance(i, float) for i in sol["guesses"])
    assert all(isinstance(i, float) for i in sol["LD_bestguess_TargetForm"])
    assert all(isinstance(i, float) for i in sol["NLD_bestguess_TargetForm"])
    assert all(isinstance(i, float) for i in sol["bestguess_nse"])
    assert all(isinstance(i, str) for i in sol["best_guess"])
    assert all(isinstance(literal_eval(i), list) for i in sol["adapted_struc"])
    assert all(isinstance(literal_eval(i), list) for i in sol["adapted_vowelharmony"])
    assert all(isinstance(literal_eval(i), list) for i in sol["before_combinatorics"])
    assert all(i=="" or isinstance(literal_eval(i), list) for i in sol["pred_strucs"])
    assert all(isinstance(i, str) or i is None for i in sol["donor_struc"])
    del (sol["guesses"], sol["best_guess"], sol["LD_bestguess_TargetForm"],
    sol["NLD_bestguess_TargetForm"], sol["adapted_struc"], sol["adapted_vowelharmony"],
    sol["before_combinatorics"], sol["donor_struc"], sol["pred_strucs"],
    sol["bestguess_nse"])
    assert_frame_equal(sol, df_exp, check_dtype=False)

    remove(MOCK_CACHE_PATH)
    remove(path2test_out_eval_all)

    #assert sound changes were written correctly
    #scdictbase must be false for this, else files too large to compare

    sol = eval_all(
    opt_param_path=MOCK_CACHE_PATH,
    formscsv=path2cog27,
    tgt_lg="EAH",
    src_lg="WOT",
    crossval=True,

    guesslist=[1, 5, 10, 100, 1000],
    max_struc=2,
    max_paths=3,
    writesc=PATH2OUT, # just a folder!
    vowelharmony=True,
    clusterised=True,
    sort_by_nse=True,
    struc_filter=True,
    show_workflow=False,

    mode="adapt",
    scdictbase=None, #rankclosest ranks equally similar sounds randomly!
    #causing the results in fp and best_guess to be random as well
    write_to=None,
    plot_to=None,
    plotldnld=False)

    for i in range(23):
        out = literal_eval(open(PATH2OUT / f"sc{i}isolated.txt").read())
        assert isinstance(out, list)
        assert len(out) == 6
        assert [isinstance(i, dict) for i in out]
        remove(PATH2OUT / f"sc{i}isolated.txt")

    remove(MOCK_CACHE_PATH)

    del scb

    #assert crossval=False works correctly by checking the written file
    path2noncrossval = PATH2OUT / "test_noncrossval_sc_eval_all_integration.txt"
    path2exp = Path(__file__).parent / "expected_files" / "\
test_noncrossval_sc_eval_all_integration.txt"

    sol = eval_all(
    opt_param_path=MOCK_CACHE_PATH,
    formscsv=path2cog27,
    tgt_lg="EAH",
    src_lg="WOT",
    crossval=False,

    guesslist=[1, 5, 10, 100, 1000],
    max_struc=2,
    max_paths=3,
    writesc=path2noncrossval,
    vowelharmony=True,
    clusterised=True,
    sort_by_nse=True,
    struc_filter=True,
    show_workflow=False,

    mode="adapt",
    scdictbase=None, #rankclosest ranks equally similar sounds randomly!
    #causing the results in fp and best_guess to be random as well
    write_to=None,
    plot_to=None,
    plotldnld=False)

    out = literal_eval(open(path2noncrossval).read())
    out3 = out.pop(3)
    exp = literal_eval(open(path2exp).read())
    exp3 = exp.pop(3)
    assert out == exp
    assert set(out3) == set(exp3) #phonotactic inventory has randomness


    remove(MOCK_CACHE_PATH)

    del out

def test_plot_roc():

    path2mockplot = Path(__file__).parent / "output_files" / "mockplot.jpg"

    df_forms_mock = DataFrame(
    {"Source_Form": ["kiki", "buba", "kaba"],
     "Target_Form": ["hehe", "pupa", "hapa"],
     "guesses": [1, None, 3]})

    plot_roc(df=df_forms_mock, fplist=[0,1,2],
    plot_to=path2mockplot, tpr_fpr_opt=(
    [0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
    [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
    (0.501, 0.6, 0.099)), opt_howmany=1,
    opt_tpr=0.6, len_df=3, mode="adapt")

    #verify manually that results in output_files and expected_files are same

def test_postprocess():
    adrc_obj = Adrc(PATH2FORMS, "WOT", "EAH")
    adrc_obj.dfety["best_guess"] = ["apple", "apple2", "apple3"]
    assert_frame_equal(postprocess(adrc_obj), DataFrame({
    'Target_Form': ["aɣat͡ʃi", "aldaɣ", "ajan"],
    'Source_Form': ["aɣat͡ʃːɯ", "aldaɣ", "ajan"],
    'Cognacy': [1, 2, 3],
    'best_guess': ["apple", "apple2", "apple3"],
    'LD_bestguess_TargetForm': [6, 5, 5],
    'NLD_bestguess_TargetForm': [0.86, 0.83, 0.83],
    'comment': [""]*3}))






























    #
