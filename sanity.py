"""
Make sure our model is not crazy.

"""

from collections import OrderedDict
from datetime import datetime
from math import ceil
from re import search
from time import gmtime, strftime, time

try: from matplotlib.pyplot import (legend, plot, savefig, scatter, text, title,
                                   xlabel, ylabel)
except KeyError: pass ##Sphinx needs this to generate the documentation
from pandas import DataFrame, concat, read_csv
from panphon.distance import Distance
from tqdm import tqdm

from loanpy.helpers import Etym, get_howmany
from loanpy.adrc import Adrc

BANNED = [None, "KeyError", "not old", "wrong phonotactics",
    "wrong vowel harmony", "wrong clusters"]

class ArgumentsAlreadyTested(Exception):
    pass

class Missing_dfargscsv(Exception):
    pass

def eval_all(
formscsv, # name of df to evaluate (forms.csv in cldf)
tgt_lg,  # target language in forms.csv (cldf format)
src_lg,  # source language in forms.csv (cldf format)
opt_param_path=None,  # path to DIY cache (contains optimal parameters)
crossval=True,  # boolean: Should word to predict be isolated from training data?
#the following 9 args go into adrc.adapt
guesslist=[10, 50, 100, 500, 1000],  # list of input args for param "howmany"
max_struc=1,  # ceiling for nr of repaired phonotactic structures
max_paths=1,  # ceiling for nr of paths to repaired phonotactic strucs
writesc=False,  # should soundchanges.txt be written to file (for debugging)
vowelharmony=False,  # should vowelharmony be adapted? (in adrc.adapt)
clusterised=False,  # goes into adrc.adapt
sort_by_nse=False,  # goes into adrc.adapt or reconstruct
struc_filter=False,  # goes into adrc.adapt or reconstruct
show_workflow=False,  # goes into adrc.adapt
#should scdictbase be generated?
scdictbase=None, # cannot be bool!
mode="adapt", #should we adapt or reconstruct (two funcs in adrc.py)
#the following args are part of post processing
write_to=None,  # path name if df should be written to csv-file
plot_to=None, # path name if curves should be drawn with matplotlib else None
plotldnld=False):  # indicate if normalised levenshtein distance should be calculated
    """

    Trains models, evaluates and visualises predictions.

    Positional arguments:

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

    Keyword arguments:

    :param opt_param_path: The path to the csv-file in which information \
about input-parameters and evaluated results is stored. If path points to \
a non-existent file, the correct file will be created. \
If set to None, it will be written to cldf's folder etc (concluded from \
the path provided in parameter <formscsv>) and will be called f"opt_param_\
{tgt_lg}_{src_lg}". \
For more information \
see loanpy.sanity.check_cache.
    :type opt_param_path: pathlib.PosixPath | str

    :param crossval: Indicate if results should be cross-validated. If true, \
the model with which we predict an adaptation or a reconstruction \
of a word will be trained without that word.
    :type crossval: bool

    :param guesslist: The list of number of guesses to be made. Will be \
passed into loanpy.adrc.Adrc.adapt's or loanpy.adrc.Adrc.reconstruct's \
parameter <howmany> in a loop. Loop breaks if prediction was correct.
    :type guesslist: list of int

    :param max_struc: The maximum number of phonotactic strucutres into which \
the original string should be transformed. Will be passed into \
loanpy.adrc.Adrc.adapt's parameter <max_struc>
    :type max_struc: int, default=1

    :param max_paths: The maximum number of cheapest paths through which a \
phonotactic structure can be repaired. Will be passed into \
loanpy.adrc.Adrc.adapt's parameter <max_paths>
    :type max_paths: int, default=1

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

    :param vowelharmony: Will be passed to loanpy.adrc.Adrc.adapt's or \
loanpy.adrc.Adrc.reconstruct's parameter <vowelharmony>. \
Indicate whether vowelharmony should be repaired \
if passed on to adapt() or if results violating the constraint "front-back \
vowelharmony" should be filtered out if passed on to reconstruct().
    :type vowelharmony: bool, default=False

    :param clusterised: Will be passed to loanpy.adrc.Adrc.adapt's or \
loanpy.adrc.Adrc.reconstruct's parameter <clusterised>. \
Indicate whether predictions that contain consonant or \
vowel clusters that are not documented in the target language should be \
filtered out if passed on to adapt() or if the tokeniser should \
clusterise the input-word and look in the sound correspondence dictionary \
for clusters as keys to predictions if passed on to reconstruct().
    :type clusterised: bool, def=False

    :param sort_by_nse: Indicate if predictions should be sorted by likelihood
    :type sort_by_nse: bool, default=False

    :param struc_filter: Indicate if predictions made by \
loanpy.adrc.Adrc.adapt should be filtered out if they consist of a \
phonotactic structure that is not contained in the language's \
phonotactic inventory.
    :type struc_filter: bool, default=False

    :param show_workflow: Indicate whether the workflow should be \
displayed in the output. Useful for debugging.
    :type show_workflow: bool, default=False

    :param scdictbase: Indicate whether sound correspondences based \
on data should be combined with the (rather large) dictionary of \
heuristic correspondences. For pitfalls see param <writesc>.
    :type scdictbase: None | pathlib.PosixPath | dict

    :param mode: Indicate whether predictions should be made with \
loanpy.adrc.Adrc.reconstruct or loanpy.adrc.Adrc.adapt.
    :type mode: "adapt" | "reconstruct", default="adapt"

    :param write_to: Indicate whether results should be written to a \
text file. If yes, provide the path. None means that no file will be written.
    :type write_to: None | pathlib.PosixPath | str, default=None

    :param plot_to: Indicate whether results should be plotted as an \
ROC-curve to a jpg-ile. If yes, provide the path. \
None means that no file will be written.
    :type plot_to: None | pathlib.PosixPath | str, default=None

    :param plotldnld: Indicate whether Levenshtein distances and normalised \
Levenshtein distances should be plotted. Does currently not work.
    :type plotldnld: bool, default=False

    """

    if opt_param_path is None: opt_param_path = formscsv.parent.parent / "\
etc" / f"opt_params_{src_lg}_{tgt_lg}.csv"

    eval_all_args = locals()
    check_cache(opt_param_path, eval_all_args)

    adrc_obj, step7_fp, best_guess, workflow = Adrc(
    src_lg, tgt_lg, formscsv=formscsv, mode=mode, scdictbase=scdictbase
    ), [], [], OrderedDict({"target": [],
    "source": [], "sol_idx_plus1": [], "tokenised": [], "adapted_struc": [],
    "adapted_vowelharmony": [], "before_combinatorics": [], "donor_struc": [],
    "pred_strucs": []})

    if crossval is False: adrc_obj.scdict, adrc_obj.sedict, _, adrc_obj.scdict_struc, _, _ = adrc_obj.get_sound_corresp(writesc) #getsc

    start = time()
    for idx,(srcwrd, tgtwrd) in tqdm(
    enumerate(zip(adrc_obj.dfety["Source_Form"], adrc_obj.dfety["Target_Form"]))):
        if crossval is True: adrc_obj = get_sc(adrc_obj, idx, writesc)
        solution = eval_one(adrc_obj, srcwrd, tgtwrd, guesslist,
        max_struc, max_paths, vowelharmony,
        clusterised, sort_by_nse, struc_filter, show_workflow, mode)
        #10 args b/c + tgtwrd (adapt takes 9 args)

        step7_fp.append(solution["sol_idx_plus1"])
        best_guess.append(solution["best_guess"])
        if show_workflow: #better to do this explicitely than in loop!
            workflow["target"].append(tgtwrd)
            workflow["source"].append(srcwrd)
            workflow["sol_idx_plus1"].append(solution["sol_idx_plus1"])
            workflow["tokenised"].append(solution["workflow"].get("tokenised", None))
            workflow["adapted_struc"].append(solution["workflow"].get("adapted_struc", None))
            workflow["adapted_vowelharmony"].append(solution["workflow"].get("adapted_vowelharmony", None))
            workflow["before_combinatorics"].append(solution["workflow"].get("before_combinatorics", None))
            workflow["donor_struc"].append(solution["workflow"].get("donor_struc", None))
            workflow["pred_strucs"].append(solution["workflow"].get("pred_strucs", "")) #for write workflow! None is not iterable
    end = time()

    adrc_obj.dfety["guesses"], adrc_obj.dfety["best_guess"] = step7_fp, best_guess
    fplist, len_df = [i-1 for i in guesslist], len(adrc_obj.dfety)
    tpr_fpr_opt = gettprfpr(step7_fp, fplist, len_df)
    stat = make_stat(tpr_fpr_opt[2][2], tpr_fpr_opt[2][1], fplist[-1], len_df)
    write_to_cache(stat, eval_all_args, opt_param_path, start, end)
    if write_to: adrc_obj.dfety.to_csv(write_to, encoding="utf_8", index=False)
    if plot_to: plot_roc(adrc_obj.dfety, fplist, plot_to, tpr_fpr_opt, stat[0], stat[2], len_df, mode)
    if show_workflow:
        wf = write_workflow(workflow, adrc_obj.dfety["best_guess"], opt_param_path)  # writes AND returns
        if plotldnld: plot_ld_nld(wf)

    return adrc_obj.dfety

def check_cache(opt_param_path, init_args):
    """create DIY cache"""

    try:
        for idx, row in read_csv(opt_param_path,
            usecols=list(init_args)).fillna("").iterrows():
            if list(map(str, list(init_args.values()))
            ) == list(map(str, list(row))):
            #check whether these parameters were run already
                raise ArgumentsAlreadyTested(f"These arguments were tested \
already, see {opt_param_path} line {idx+1}! (start counting at 1 in 1st row)")

    except FileNotFoundError: #if cache doesn't exist, create empty one
         DataFrame(columns=list(init_args)+[
         "opt_tpr",
         "optimal_howmany",
         "opt_tp",
         "timing",
         "date"]).to_csv(opt_param_path, index=False, encoding="utf-8")

def get_sc(adrc_obj, idx, writesc=None):
    dropped_row = DataFrame(dict(adrc_obj.dfety.iloc[idx]), index=[idx])
    adrc_obj.dfety = adrc_obj.dfety.drop([adrc_obj.dfety.index[idx]]) #isolate
    if writesc: writesc = writesc / f"sc{idx}isolated.txt"
    adrc_obj.scdict, adrc_obj.sedict, _, adrc_obj.scdict_struc, _, _ = adrc_obj.get_sound_corresp(writesc) #getsc
    adrc_obj.dfety = concat([adrc_obj.dfety.head(idx),
                             dropped_row, #re-insert isolated row
                             adrc_obj.dfety.tail(len(adrc_obj.dfety)-idx)])
    return adrc_obj

def eval_one(adrc_obj, srcwrd, tgtwrd, guesslist, max_struc, max_paths,
vowelharmony, clusterised, sort_by_nse, struc_filter, show_workflow, mode): #11 args b/c + tgtwrd!

    sol_dict = {"sol_idx_plus1": float("inf"), "best_guess": ""}
    if mode == "adapt":
        for guess in guesslist:
            guess = get_howmany(guess, max_struc, max_paths)
            try: adapted = adrc_obj.adapt(srcwrd, guess[0], guess[1], guess[2], #9 args b/c no tgtwrd!
            vowelharmony, clusterised, sort_by_nse, struc_filter, show_workflow)
            except KeyError:
                sol_dict["best_guess"] = "KeyError"
                break

            adapted = adapted.split(", ")
            sol_dict["best_guess"] = adapted[0]
            if tgtwrd in adapted:
                sol_dict["sol_idx_plus1"] = adapted.index(tgtwrd)+1
                sol_dict["best_guess"] = tgtwrd
                break

    elif mode == "reconstruct":
        for guess in guesslist:
            srcwrd, tgtwrd = tgtwrd, srcwrd
            try: reconstructed = adrc_obj.reconstruct(srcwrd, guess, clusterised, struc_filter,
            vowelharmony, sort_by_nse)
            except KeyError:
                sol_dict["best_guess"] = "KeyError"
                break

            if "(" in reconstructed: #if combinatorics were not applied
                if bool(search(reconstructed, tgtwrd)): #check if predicted
                    sol_dict["sol_idx_plus1"] = guess #howmany guesses needed
                    sol_dict["best_guess"] = reconstructed
                    break

            elif "|" in reconstructed: #if combinatorics were applied:
                reconstructed = reconstructed[1:-1].split("$|^") #turn2list
                sol_dict["best_guess"] = reconstructed[0]
                if tgtwrd in reconstructed:
                    sol_dict["sol_idx_plus1"] = reconstructed.index(tgtwrd)+1
                    sol_dict["best_guess"] = tgtwrd
                    break

            sol_dict["best_guess"] = reconstructed

    if mode=="adapt" and show_workflow: sol_dict["workflow"] = adrc_obj.workflow
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

def write_workflow(workflow, best_guess, opt_param_path):
    wf = DataFrame(workflow)
    wf["comment"], etym, dist = "", Etym(), Distance()
    wf["struc_predicted"] = [True if etym.word2struc(actual) in pred else False for actual, pred in zip(wf["target"], wf["pred_strucs"])]
    wf["LD_bestguess_target"] = [dist.fast_levenshtein_distance(b,t) if b not in BANNED else float("inf") for b,t in zip(best_guess, wf["target"])]
    wf["NLD_bestguess_target"] = [dist.fast_levenshtein_distance_div_maxlen(b,t) if b not in BANNED else float("inf") for b,t in zip(best_guess, wf["target"])]

    workflow_path = opt_param_path.parent / "workflow.csv"
    wf.to_csv(f"{workflow_path}",
    encoding="utf_8", index=False)
    return wf

def plot_ld_nld(wf):
    gl = list(range(11)) #guesslist = Levenshteindistances
    tpr_fpr_opt = gettprfpr(wf, gl, "LD")  # different input params!
    plot_roc(wf, gl, f"{write_to}_Levenshtein", tpr_fpr_opt, LD=True)  # different input params!

    gl = [i/10 for i in range(11)] #guesslist = normalised Levenshteindistances
    tpr_fpr_opt = gettprfpr(wf, gl, "NLD")  # different input params
    plot_roc(wf, gl, f"{outname}_normalised_Levenshtein", tpr_fpr_opt, NLD=True)  #different input params!
