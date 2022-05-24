"""
Check how sane the model is by evaluating predictions.

"""

from collections import OrderedDict
from datetime import datetime
from math import ceil
from re import search
from time import gmtime, strftime, time

try: from matplotlib.pyplot import (legend, plot, savefig, scatter, text, title,
                                   xlabel, ylabel)
except KeyError: pass  # Sphinx needs this to generate the documentation
from pandas import DataFrame, concat, read_csv
from panphon.distance import Distance
from tqdm import tqdm

from loanpy.helpers import Etym, get_howmany
from loanpy.adrc import Adrc

BANNED = ["KeyError", "not old", "wrong phonotactics",
          "wrong vowel harmony", "wrong clusters"]

class ArgumentsAlreadyTested(Exception):
    """
    Raised by loanpy.sanity.check_cache if arguments for evaluation were \
already run once. Classical cache thing.
    """
    pass

def eval_all(
# Fllowing 9 params will go to loanpy.adrc.Adrc.__init__
formscsv, # name of df etymological data to evaluate (forms.csv in cldf)
tgt_lg,  # computational target language in forms.csv (cldf format)
src_lg,  # computational source language in forms.csv (cldf format)
mode="adapt",  # should we adapt or reconstruct?
struc_most_frequent=9999999,  # define the howmany most freq strucs to accept to phonotactic inventory
struc_inv=None,  # chance to hard-code phonotactic inventory
connector=None,  # by default "<" for adaptations and  "<*" for reconstr.
scdictbase=None, # cannot be bool! Provide path, or dict, else None
vfb=None,  # define placeholder vowels else None
# These 5 go to both loanpy.adrc.Adrc.adapt AND loanpy.adrc.Adrc.reconstruct
guesslist=[10, 50, 100, 500, 1000],  # list of input args for param "howmany"
clusterised=False,  # should wrong clusters be filtered (adapt) or input clusterised (reconstruct)?
vowelharmony=False,  # should vowelharmony be repaired (adapt) or filtered (reconstruct)?
struc_filter=False,  # should words with wrong phonotactics be filtred out?
sort_by_nse=False,  # sort results by likelihood?
# These 5 go only to loanpy.adrc.Adrc.adapt if mode=="adapt"
max_struc=1,  # ceiling for nr of repaired phonotactic structures in adapt
max_paths=1,  # ceiling for nr of paths to repaired phonotactic strucs in adapt
deletion_cost=100,
insertion_cost=49,
show_workflow=False,  # should workflow be displayed (adapt)
# These 6 are for internal use
opt_param_path=False,  # path to cache (contains optimal parameters)
crossval=True,  # boolean: Should word to predict be isolated from training data?
writesc=False,  # should sound corresp files be written? (for debugging)
write_to=None,  # path name if results should be written to csv-file
plot_to=None, # path name if results should be plotted with matplotlib else None
plotldnld=False):  # indicate if normalised levenshtein distance should be plotted
    """

        Trains crossvalidated models, evaluates and visualises predictions. \
25 args in total, only 3 positional ones.

    The first 9 args are passed on to \
    loanpy.adrc.Adrc.__init__, out of which the first 3 are positional:

    :param formscsv: The path to cldf's forms.csv of the etymological \
dictionary. Will be used to initiate loanpy.adrc.Adrc. For more details see \
loanpy.helpers.read_forms.
    :type formscsv: pathlib.PosixPath | str

    :param tgt_lg: The computational target language. Will be used to \
    initiate loanpy.adrc.Adrc. For more details see loanpy.helpers.Etym.
    :type tgt_lg: str (options are listed in column "ID" in \
    cldf / etc / languages.tsv), default=None

    :param src_lg: The computational source language. Will be used to \
    initiate loanpy.adrc.Adrc. For more details see loanpy.helpers.Etym.
    :type src_lg: str (options are listed in column "ID" in \
    cldf / etc / languages.tsv), default=None

    :param mode: Indicate whether predictions should be made with \
    loanpy.adrc.Adrc.reconstruct or loanpy.adrc.Adrc.adapt (also \
    sound correspondences \
    will be extracted with loanpy.qfysc.Qfy.get_sound_corresp in the given mode). \
    See also loanpy.qfysc.Qfy for more details.
    :type mode: "adapt" | "reconstruct", default="adapt"

    :param struc_most_frequent: The n most frequent structures \
    to accept into the target language's phonotactic inventory. For more details \
    see loanpy.helpers.Etym.read_strucinv.
    :type struc_most_frequent: int, default=9999999

    :param struc_inv: Chance to plug in phonotactic inventory manually. If None, \
    will be extracted automatically from data. For more details \
    see loanpy.helpers.Etym.read_strucinv.
    :type struc_inv: list, default=None

    :param connector: The string that connects the left side of an etymology \
    with the right side for adapting vs reconstructing. For more details see \
    loanpy.qfysc.read_connector
    :type connector: iterable of len 2

    :param scdictbase: Indicate whether sound correspondences based \
    on data should be combined with the (rather large) dictionary of \
    heuristic correspondences. For pitfalls see param <writesc>.
    :type scdictbase: None | pathlib.PosixPath | dict

    :param vfb: Indicate whether there should be placeholder vowels. \
    For more details see loanpy.qfysc.Qfy.

    The next 5 args are passed on to loanpy.adrc.Adrc.adapt if \
    mode="adapt" but to loanpy.adrc.Adrc.reconstruct if mode="reconstruct":

    :param guesslist: The list of number of guesses to be made. Will be \
    passed into loanpy.adrc.Adrc.adapt's or loanpy.adrc.Adrc.reconstruct's \
    parameter <howmany> in a loop. Loop breaks as soons as prediction was correct.
    :type guesslist: list of int

    :param clusterised: Will be passed to loanpy.adrc.Adrc.adapt's or \
    loanpy.adrc.Adrc.reconstruct's parameter <clusterised>. \
    Indicate whether predictions that contain consonant or \
    vowel clusters that are not documented in the target language should be \
    filtered out if passed on to adapt() or if the tokeniser should \
    clusterise the input-word and look in the sound correspondence dictionary \
    for clusters as keys to predictions if passed on to reconstruct().
    :type clusterised: bool, def=False

    :param vowelharmony: Will be passed to loanpy.adrc.Adrc.adapt's or \
    loanpy.adrc.Adrc.reconstruct's parameter <vowelharmony>. \
    Indicate whether vowelharmony should be repaired \
    if passed on to adapt() or if results violating the constraint "front-back \
    vowelharmony" should be filtered out if passed on to reconstruct().
    :type vowelharmony: bool, default=False

    :param struc_filter: Indicate if predictions made by \
    loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct \
    should be filtered out if they consist of a \
    phonotactic structure that is not contained in the language's \
    phonotactic inventory.
    :type struc_filter: bool, default=False

    :param sort_by_nse: Indicate if predictions should be sorted by likelihood
    :type sort_by_nse: bool, default=False

    The next 5 args are passed on to loanpy.adrc.Adrc.adapt if mode="adapt" but if \
    mode="reconstruct" they will not be used.

    :param max_struc: The maximum number of phonotactic strucutres into which \
    the original string should be transformed. Will be passed into \
    loanpy.adrc.Adrc.adapt's parameter <max_struc>
    :type max_struc: int, default=1

    :param max_paths: The maximum number of cheapest paths through which a \
    phonotactic structure can be repaired. Will be passed into \
    loanpy.adrc.Adrc.adapt's parameter <max_paths>
    :type max_paths: int, default=1

    :param deletion_cost: The cost of deleting a phoneme
    :type deletion_cost: int, float, default=100

    :param insertion_cost: The cost of inserting a phoneme
    :type insertion_cost: int, float, default=49

    :param show_workflow: Indicate whether the workflow should be \
    displayed in the output of loanpy.adrc.Adrc.adapt. Useful for debugging.
    :type show_workflow: bool, default=False

    The next 6 args are not passed on but will be used in this module.

    :param opt_param_path: The path to the csv-file in which information \
    about input-parameters and evaluated results is stored. If path points to \
    a non-existent file, the indicated file will be created. \
    If set to None, it will be written to cldf's folder etc (concluded from \
    the path provided in parameter <formscsv>) and will be called \
    f"opt_param_{tgt_lg}_{src_lg}". \
    If set to False, this file will be ignored and no data written to it. \
    For more information see loanpy.sanity.check_cache.
    :type opt_param_path: pathlib.PosixPath | str, None, False, default=False

    :param crossval: Indicate if results should be cross-validated. If true, \
    the model with which we predict an adaptation or a reconstruction \
    of a word will be trained without that word.
    :type crossval: bool

    :param writesc: Indicate if loanpy.qfysc.Qfy.get_sound_corresp should \
    write its output to a file. If yes and if crossval is True, \
    provide a path to a *folder* (!). If yes and crossval is False, \
    provide a path to a file. \
    This is useful for debugging. \
    Careful: Since one file will be written in every \
    round of the cross-validation loop (as many iterations as there are \
    predictions to evaluate), the total storage room taken up by the files can \
    get large. E.g. if we want to evaluate 500 words and scdictbase is \
    1.6MB, the entire folder will take up 500*1.6MB. There are two ways to \
    avoid this: If writesc=True, make sure to set either crossval=False, this will \
    only write one soundchange.txt file. Or set scdictbase=None, since this is \
    the part that takes up the most storage. Predictions will be blurred in both \
    cases but for debugging this is usually enough.
    :type writesc: bool, def=False

    :param write_to: Indicate whether results should be written to a \
    text file. If yes, provide the path. None means that no file will be written.
    :type write_to: None | pathlib.PosixPath | str, default=None

    :param plot_to: Indicate whether results should be plotted as an \
    ROC-curve to a jpg-ile. If yes, provide the path. \
    None means that no file will be written.
    :type plot_to: None | pathlib.PosixPath | str, default=None

    :param plotldnld: Indicate whether Levenshtein distances and normalised \
    Levenshtein distances should be plotted. DCurrently not supported.
    :type plotldnld: bool, default=False

    :returns: Adds two columns to the input data frame: "guesses", which indicate \
howmany guesses were necessary to make the correct prediction (if inf, \
all predictions were wrong) and best guess, which shows the closest guess (if \
combinatorics were applied, this is 1 word, in case of (a)(b)(c) type regexes \
the entire regex is just kept.
    :rtype: pandas.core.frame.DataFrame

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.sanity import eval_all, __file__
    >>> path2cog27 = Path(__file__).parent / "tests" / "integration" / "input_files" / "forms_27cogs.csv"
    >>> eval_all(formscsv=path2cog27, tgt_lg="EAH", src_lg="WOT", \
mode="reconstruct", clusterised=True, sort_by_nse=True)
    23it [00:00, 54.31it/s]
       Target_Form Source_Form  Cognacy  guesses      best_guess
    0      aɣat͡ʃi  aɣat͡ʃt͡ʃɯ        1      inf  ɣ, t͡ʃ not old
    1        aldaɣ       aldaɣ        2      inf      ld not old
    2         ajan        ajan        3      inf       j not old
    3          aːl          al        8      inf      l# not old
    4       alat͡ʃ     alat͡ʃɯ        9      inf          alat͡ʃ
    5       alat͡ʃ     alat͡ʃu       10      inf          alat͡ʃ
    6       alat͡ʃ     alat͡ʃo       11      inf          alat͡ʃ
    7       alat͡ʃ      ɒlɒt͡ʃ       12      inf          alat͡ʃ
    8         alma        alma       13      inf      lm not old
    9      altalaɡ     altɯlɯɡ       14      7.0         altɯlɯɡ
    10     altalaɡ     altɯlɯɡ       15      7.0         altɯlɯɡ
    11          op          op       16      inf      p# not old
    12       oporo       opura       17      inf   o, o# not old
    13      opuruɣ      opuruɣ       18      1.0          opuruɣ
    14        orat        orat       19      1.0            orat
    15        orat          or       20      inf            orat
    16         aːr          ar       21      inf      r# not old
    17      aːrtat       artat       22      inf      rt not old
    18       arkan       arkan       23      inf      rk not old
    19       aːruk        aruk       24      inf      k# not old
    20        arpa        arpa       25      inf      rp not old
    21      aritan      arɯtan       26      inf    i, t not old
    22        aski        askɯ       27      inf      sk not old


    """

    # don't check for the cache if param set to False
    if opt_param_path is not False:
        # if param set to None set default path in cldf folder "etc".
        if opt_param_path is None: opt_param_path = formscsv.parent.parent / "\
etc" / f"opt_params_{src_lg}_{tgt_lg}.csv"
        # capture all local vars. See help(locals)
        eval_all_args = locals()  # captures current scope's local variables
        # so it has to be run in the beginning before more local vars are added.
        check_cache(opt_param_path, eval_all_args)

    # The first 9 args are passed on here.
    adrc_obj = Adrc(
    formscsv=formscsv,
    srclg=src_lg,
    tgtlg=tgt_lg,
    mode=mode,
    struc_most_frequent=struc_most_frequent,
    struc_inv=struc_inv,
    connector=connector,
    scdictbase=scdictbase,
    vfb=vfb)

    # these will be the two new columns to attach to the input to create output
    step7_fp, best_guess = [], []

    # this will show the single steps of the process. For debugging.
    if show_workflow: workflow = OrderedDict({"tokenised": [],
    "adapted_struc": [], "adapted_vowelharmony": [],
    "before_combinatorics": [], "donor_struc": [],
    "pred_strucs": []})

    # if we don't crossvalidate we just get sound corresp from big file
    if crossval is False: (adrc_obj.scdict, adrc_obj.sedict, _,
    adrc_obj.scdict_struc, _, _) = adrc_obj.get_sound_corresp(writesc) #getsc

    # document how long the entire loop takes. Can take long if forms.csv is big.
    start = time()
    # loop through etymological data with a progressbar
    for idx,(srcwrd, tgtwrd) in tqdm(
    # pick only source form (as input) and target form (for evaluation)
    enumerate(zip(adrc_obj.dfety["Source_Form"], adrc_obj.dfety["Target_Form"]))):
        # drop the current word for training if crossval=True
        if crossval is True: adrc_obj = get_crossval_sc(adrc_obj, idx, writesc)
        # make prediction from source word, check if target was hit
        # solution is a dictionary with infos about the prediction
        solution = eval_one(adrc_obj, srcwrd, tgtwrd, guesslist,
        max_struc, max_paths, deletion_cost, insertion_cost, vowelharmony,
        clusterised, sort_by_nse, struc_filter, show_workflow, mode)

        # append idx of guess to new col for output df (inf if not predicted).
        step7_fp.append(solution["sol_idx_plus1"])
        # append the best guess to the other new col for output df
        best_guess.append(solution["best_guess"])
        # append contents of adapt's workflow-dict to new cols for output df.
        if show_workflow:  # better to do this explicitely than in loop!
            workflow["tokenised"].append(
            solution["workflow"].get("tokenised", None))
            workflow["adapted_struc"].append(
            solution["workflow"].get("adapted_struc", None))
            workflow["adapted_vowelharmony"].append(
            solution["workflow"].get("adapted_vowelharmony", None))
            workflow["before_combinatorics"].append(
            solution["workflow"].get("before_combinatorics", None))
            workflow["donor_struc"].append(
            solution["workflow"].get("donor_struc", None))
            workflow["pred_strucs"].append(
            # "" b/c None is not iterable & postprocess() will need to iterate
            solution["workflow"].get("pred_strucs", ""))

    # check how long the loop took. Will be written to cache.
    end = time()  # don't print this. tqdm does that already.

    # give the actual etymologies a likelihood score (nse)
    # COMMENT OUT THE NEXT THREE LINES WHEN TESTS READY
    adrc_obj.dfety["target_nse"] = [
    adrc_obj.get_nse(src, tgt) for src, tgt in zip(
    adrc_obj.dfety["Source_Form"], adrc_obj.dfety["Target_Form"])]

    #UNCOMMENT THIS BLOCK FOR TESTING
#    adrc_obj.dfety = concat([adrc_obj.dfety,
#    DataFrame([adrc_obj.get_nse(src, tgt) for src, tgt in zip(
#    adrc_obj.dfety["Source_Form"], adrc_obj.dfety["Target_Form"])],
#    columns=["nse_target", "se_target", "distr_target", "align_target"])])
    # if show_workflow is False: del (adc_obj.dfety["se_target"],
#    adc_obj.dfety["distr_target"], adc_obj.dfety["align_target"])

    # put the two output-list into cols of the out df.
    adrc_obj.dfety["guesses"], adrc_obj.dfety["best_guess"] = step7_fp, best_guess
    # give the predicted etymologies the same likelihood score (nse)
    adrc_obj.dfety["bestguess_nse"] = [adrc_obj.get_nse(src, pred)
    for src, pred in zip(adrc_obj.dfety["Source_Form"], adrc_obj.dfety["best_guess"])]
    # mandatorily postprocess the data
    adrc_obj.dfety = postprocess(adrc_obj)

    # add optional extra stuff if indicated in the last parameters

    # add workflow to ouput df if indicated
    if show_workflow:
        # insert both nse workflows here
        adrc_obj.dfety = concat([adrc_obj.dfety, DataFrame(workflow)], axis=1)
        etym = Etym()  # check if phonotactic structure was predicted
        workflow["struc_predicted"] = [True if etym.word2struc(actual) in pred
        else False for actual, pred in
        zip(adrc_obj.dfety["Target_Form"], adrc_obj.dfety["pred_strucs"])]
        del etym  # takes up much RAM
    #only calculate this if cache should be written or roc-curve plotted
    if opt_param_path is not False or plot_to:
        len_df = len(adrc_obj.dfety)  # don't calculate this everytime anew
        #get tuple of true positive rate, false positive rate and the optimum
        tpr_fpr_opt = gettprfpr(step7_fp, guesslist, len_df)
        # create statistics from optimum, max nr of guesses and len of input df.
        stat = make_stat(tpr_fpr_opt[2][2], tpr_fpr_opt[2][1], guesslist[-1], len_df)
    # write results to cache if param not set to False
    if opt_param_path is not False:
        write_to_cache(stat, eval_all_args, opt_param_path, start, end)
    # plot results and write them to a file if indicated
    if plot_to: plot_roc(adrc_obj.dfety, guesslist, plot_to,
    tpr_fpr_opt, stat[0], stat[2], len_df, mode)

    # write output to file if indicataed
    if write_to: adrc_obj.dfety.to_csv(write_to, encoding="utf_8", index=False)

    # plot the postprocessing - currently not supported
    if plotldnld: plot_ld_nld(adrc_obj.dfety)

    # return the input data frame with the new columns
    return adrc_obj.dfety

def check_cache(opt_param_path, init_args):
    """
    Called by loanpy.sanity.eval_all. \
Checks if cache-file exists, if not: empty file is created, \
if yes: checks whether init_args occur in one of its rows. If yes: \
Error is raised. If no, nothing happens.

    :param opt_param_path: The path to the csv-file in which information \
    about input-parameters and evaluated results is stored. If path points to \
    a non-existent file, the correct file will be created. \
    If set to None, it will be written to cldf's folder etc (concluded from \
    the path provided in parameter <formscsv>) and will be called \
    f"opt_param_{tgt_lg}_{src_lg}". \
    For more information see loanpy.sanity.check_cache.
    :type opt_param_path: pathlib.PosixPath | str

    :param init_args: Dictionary where keys are the arguments of \
loanpy.sanity.eval_all and vals their assigned value. Generated by locals(). \
See help(locals).
    :type init_args: dict

    :raises ArgumentsAlreadyTested: If these arguments were already passed \
to loanpy.sanity.eval_all once, they won't be calculated again. \
To solve this error, change the value of some input args, or provide a path \
to a different cache, or delete the current cache, or delete the particular \
row that was tested already from the current test (The error message \
explicitely mentions which row contains the identical args.)

    :returns: Just raises an Error or writes an empty cache under \
certain conditions, else no action
    :rtype: None
    """

    try:  # this means that the file already exists
        for idx, row in read_csv(opt_param_path,  # loop through file
            usecols=list(init_args)).fillna("").iterrows():
            # check whether given parameters were already run once
            if list(map(str, list(init_args.values()))
            ) == list(map(str, list(row))):
            # if yes, raise error and specify the row where they are stored
                raise ArgumentsAlreadyTested(f"These arguments were tested \
already, see {opt_param_path} line {idx+1}! (start counting at 1 in 1st row)")

    except FileNotFoundError: # if cache doesn't exist, create empty one
    # columns are the args with which eval_all was run
         DataFrame(columns=list(init_args)+[
         # as well as evaluation columns eval_all will create
         "opt_tpr", "optimal_howmany", "opt_tp", "timing", "date"
         # write empty cache to file at indicated location
         ]).to_csv(opt_param_path, index=False, encoding="utf-8")

def get_crossval_sc(adrc_obj, idx, writesc=None):
    """
    Called by loanpy.sanity.eval_all. \
Get sound changes by dropping the indicated row to isolate from the dataframe, \
and extracting sound correspondences from the data without the dropped row.

    :param adrc_obj: An instance of the loanpy.adrc.Adrc class. Contains \
self.dfety, which is the etymological data for training.
    :type adrc_obj: loanpy.adrc.Adrc

    :param idx: Index of the row to drop from the etymological data
    :type idx: int

    :param writesc: Indicate whether the sound correspondece files (results of \
training) should be written. If None, they will not be written. If they should \
be written, a path to a *folder* has to be provided since this function \
will be called for every round of the main loop in loanpy.sanity.eval_all and \
multiple files will be written. See param writesc in eval_all for more details.
    :type writesc: None | pathlib.PosixPath (to folder!) | str

    :returns: The same instance of the Adrc-class but with the cross-validated \
model passed into its attributes for looking up sound correspondences.
    :rtype: loanpy.adrc.Adrc

    """
    # memorise dropped row to plug back in later
    dropped_row = DataFrame(dict(adrc_obj.dfety.iloc[idx]), index=[idx])
    # drop the indicated row for training
    adrc_obj.dfety = adrc_obj.dfety.drop([adrc_obj.dfety.index[idx]]) #isolate
    # create filename for corssvalidated training results
    if writesc: writesc = writesc / f"sc{idx}isolated.txt"
    # train model on crossvalidated data (chosen row is isolated)
    adrc_obj.scdict, adrc_obj.sedict, _, adrc_obj.scdict_struc, _, _ = adrc_obj.get_sound_corresp(writesc) #getsc
    # plug the dropped row back in again so df can be reused in next loop round
    adrc_obj.dfety = concat([adrc_obj.dfety.head(idx),
                             dropped_row, #re-insert isolated row
                             adrc_obj.dfety.tail(len(adrc_obj.dfety)-idx)])
    return adrc_obj

def eval_one(adrc_obj, srcwrd, tgtwrd, guesslist, max_struc, max_paths,
deletion_cost, insertion_cost,
vowelharmony, clusterised, sort_by_nse, struc_filter, show_workflow, mode): #11 args b/c + tgtwrd!

    """
    Called by loanpy.sanity.eval_all. \
Makes a prediction with crossvalidated model and checks if any of the predicted \
words match the actual one. Indicates how many guesses were necessary to \
get the prediction right (inf means correct prediction was not made).
    """

    # create output dictionary
    sol_dict = {"sol_idx_plus1": float("inf"), "best_guess": ""}

    # two modes available, default is "adapt"
    if mode == "adapt":
        for guess in guesslist:  # guess will be passed to adrc's howmany param
            # turn guess to a tuple where the product approximates original guess
            guess = get_howmany(guess, max_struc, max_paths)
            # try to make a prediction
            try: adapted = adrc_obj.adapt(srcwrd, guess[0], guess[1], guess[2],
            deletion_cost, insertion_cost,
            vowelharmony, clusterised, sort_by_nse, struc_filter, show_workflow)
            # key errors can happen due to crossvalidation
            except KeyError:  # indicate in output that phoneme was not in model
                sol_dict["best_guess"] = "KeyError"
                break  # will throw us to end of function

            # output is a string separated by ", ". Turn it to list
            adapted = adapted.split(", ")
            # first element of predictions is added to out in any case.
            sol_dict["best_guess"] = adapted[0]  # best with sort_by_nse=True
            if tgtwrd in adapted:  # if a correct prediction was made
            # add its humna readable position to output
                sol_dict["sol_idx_plus1"] = adapted.index(tgtwrd)+1
                # best guess is equal to target word if prediction was right
                sol_dict["best_guess"] = tgtwrd
                break  # if correct prediction was made jump to bottom of func

    elif mode == "reconstruct":
        for guess in guesslist:
            srcwrd, tgtwrd = tgtwrd, srcwrd  #important
            # no KeyError possible b/c reconstruct catches it with "x not old"
            reconstructed = adrc_obj.reconstruct(srcwrd, guess, clusterised, struc_filter,
            vowelharmony, sort_by_nse)

            # set up default value for output
            sol_dict["best_guess"] = reconstructed
            # check if regex is of type (a)(b)(c)
            if "(" in reconstructed:
                # if yes check if the pred regex is in the actual word
                if bool(search(reconstructed, tgtwrd)):
                    # if yes, the current state of the loop is the nr of guesses needed
                    sol_dict["sol_idx_plus1"] = guess
                    break  # jump to bottom of func if correct pred was made

            # if combinatorics were applied
            elif "^" in reconstructed:  # if combinatorics were applied:
                reconstructed = reconstructed[1:-1].split("$|^")  # turn2list
                # best guess= 1st pred in list. sort_by_nse=True is useful here.
                sol_dict["best_guess"] = reconstructed[0]
                if tgtwrd in reconstructed:  # if prediction was right
                    # indicate the howmanieth guess was the correct one
                    sol_dict["sol_idx_plus1"] = reconstructed.index(tgtwrd)+1
                    # best guess is equal to target word if prediction was right
                    sol_dict["best_guess"] = tgtwrd
                    break  # jump to end if right prediction was made

            #else: all other cases are just error messages!

    # add workflow to output if indicated so in params
    if mode=="adapt" and show_workflow: sol_dict["workflow"] = adrc_obj.workflow
    # return dict of 2-3 keys: sol_idx_plus1, best guess, ( + workflow if indicated)
    return sol_dict

def make_stat(opt_fp, opt_tp, max_fp, len_df):
    opt_howmany = round(opt_fp*max_fp) + 1 #howmany = 1 more than fp (opt_fp is a % of max_guess)
    opt_tp_str = str(round(opt_tp*len_df)) + "/" + str(len_df) #how many did it find out of all, e.g. 10/100
    opt_tpr = str(round(opt_tp*100)) + "%" #how many percent is that, e.g. 10/100 would be 10%

    return opt_howmany, opt_tp_str, opt_tpr

def gettprfpr(step7_fp, fplist, len_step7):
    tpr, fpr = [], []
    for fp in fplist: #loop through fpr
        tpr.append(round(len([i for i in step7_fp if i and i <= fp])/len_step7, 3)) #keep only rows that are fp or lower = the correctly identified ones in the given round
        #divide that number by the number by the max amount of true positives -> e.g. 10/119 were correct if 100 guesses were made
        fpr.append(round(fp/(fplist[-1]), 3)) #how much of a fraction of the max amount of fp is our current fpr, e.g. 2K out out 10K would be 0.2 = 20%

    optimum = max([(tp-fp, tp, fp) for tp,fp in zip(tpr, fpr)])

    return tpr, fpr, optimum

def write_to_cache(stat, init_args, opt_param_path, start, end):
    """write to DIY cache"""

    for i in init_args: init_args[i] = str(init_args[i]) #to make df out of it

    #concat old and new cache, sort, write to csv
    concat([read_csv(opt_param_path), DataFrame(init_args, index=[0]).assign(
    optimal_howmany=[stat[0]], opt_tp=[stat[1]], opt_tpr=[stat[2]],
    timing=[strftime("%H:%M:%S",gmtime(end-start))],
    date=[datetime.now().strftime("%x %X")])]).sort_values(
    by=['opt_tpr'], ascending=False, ignore_index=True).to_csv(
    opt_param_path, index=False, encoding="utf_8_sig")

def plot_roc(df, fplist, plot_to, tpr_fpr_opt, opt_howmany, opt_tpr, len_df, mode, lev_dist=False, norm_lev_dist=False):
    xlabel('fpr')
    ylabel('tpr')

    if not lev_dist and not norm_lev_dist:
        plot(tpr_fpr_opt[1], tpr_fpr_opt[0], label=f'loanpy.adrc.Adrc.adapt')
        scatter(tpr_fpr_opt[2][2],tpr_fpr_opt[2][1], marker='x', c='blue', label=f"Optimum:\nhowmany={opt_howmany-1} -> tpr: {opt_tpr}")
        text(tpr_fpr_opt[1][-1]-0.3, tpr_fpr_opt[0][0], f"{mode}: 100%={fplist[-1]+1}")
        title('Predicting loanword adaptation with loanpy.adrc.Adrc.adapt')
        legend()

    if lev_dist:
        plot(tpr_fpr_opt[1], tpr_fpr_opt[0], label=f'Levenshtein Distance') # Plot some data on the axes.
        coord1 = min(tpr_fpr_opt[0], key = lambda x: abs(x-opt_tpr/100)) #find same tpr on y-axis
        coord2 = tpr_fpr_opt[1][tpr_fpr_opt[0].index(coord1)]
        scatter(coord2, coord1, marker='x', c='orange', label=f"tpr: {round(coord1*100)}% -> LD={ceil(coord2*10)}")
        text(0, tpr_fpr_opt[0][-1]-0.1, f"LD: 100%={fplist[-1]}")
        title('loanpy.adrc.Adrc.adapt vs Leveshtein Distance')
        legend()

    if norm_lev_dist:
        plot(tpr_fpr_opt[1], tpr_fpr_opt[0], label=f'Normalised Lev. Dist.') # Plot some data on the axes.
        coord1 = min(tpr_fpr_opt[0], key = lambda x: abs(x-opt_tpr/100)) #find same tpr on y-axis
        coord2 = tpr_fpr_opt[1][tpr_fpr_opt[0].index(coord1)]
        scatter(coord2, coord1, marker='x', c='green', label=f"tpr: {round(coord1*100)}% -> NLD={round(coord2, 2)}")
        text(0, tpr_fpr_opt[0][-1]-0.2, f"NLD: 100%=1")
        title('loanpy.adrc.Adrc.adapt vs Normalised Leveshtein Distance')
        legend()

    savefig(plot_to)

def postprocess(adrc_obj):

    dist, ld, nld = Distance(), [], []

    for idx, row in adrc_obj.dfety.iterrows():
        bg, tf = row["best_guess"], row["Target_Form"]
        if not any(ban in bg for ban in BANNED):
            ld.append(round(dist.fast_levenshtein_distance(bg, tf), 2))
            nld.append(round(dist.fast_levenshtein_distance_div_maxlen(bg, tf), 2))
        else:
            ld.append(float("inf"))
            nld.append(float("inf"))

    adrc_obj.dfety["LD_bestguess_TargetForm"] = ld
    adrc_obj.dfety["NLD_bestguess_TargetForm"] = nld
    adrc_obj.dfety["comment"] = ""

    return adrc_obj.dfety

def plot_ld_nld(df):
    gl = list(range(11)) #guesslist = Levenshteindistances
    tpr_fpr_opt = gettprfpr(df, gl, "LD")  # different input params!
    plot_roc(df, gl, f"{write_to}_Levenshtein", tpr_fpr_opt, LD=True)  # different input params!

    gl = [i/10 for i in range(11)] #guesslist = normalised Levenshteindistances
    tpr_fpr_opt = gettprfpr(df, gl, "NLD")  # different input params
    plot_roc(df, gl, f"{outname}_normalised_Levenshtein", tpr_fpr_opt, NLD=True)  #different input params!
