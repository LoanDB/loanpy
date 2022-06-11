"""Check how sane the model is by evaluating predictions."""

from collections import OrderedDict
from datetime import datetime
from functools import wraps
from math import ceil
from pathlib import Path
from re import search
from time import gmtime, strftime, time

try:
    from matplotlib.pyplot import (
                                   clf,
                                   legend,
                                   plot,
                                   savefig,
                                   scatter,
                                   text,
                                   title,
                                   xlabel,
                                   ylabel)
except KeyError:
    pass  # Sphinx needs this to generate the documentation
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


def cache(method):
    """
    Simple decorator function to check if function was already run with \
given arguments and to store the results in a CSV-file. For more details \
see loanpy.sanity.check_cache and loanpy.sanity.write_to_cache. Intended \
to decorate loanpy.sanity.eval_all.

    :param method: The function or method to decorate. \
    Designed for loanpy.sanity.eval_all.
    :type method: func

    :raises ArgumentsAlreadyTested: for more details see \
    loanpy.sanity.check_cache

    :returns: Raises an exception or writes a file, but return value \
    itself is always None. Uncomment "# return result" to change this \
    behaviour and return the return value of the decorated function instead.
    :rtype: None

    :Example:

    >>> # copy lines one by one, try-except throws indentation error \
when bulk-copying this entire paragraph
    >>> from pathlib import Path
    >>> from os import remove
    >>> from loanpy.sanity import cache, __file__
    >>> mockpath2cache = Path(__file__).parent / \
"tests" / "output_files" / "mock_cache.csv"
    >>> try:
    >>>     remove(mockpath2cache)  # delete leftovers from last time
    >>> except FileNotFoundError:
    >>>     pass
    >>> def mockfunc(*args, **kwargs):
    >>>     return "bla", (1, 2, 3), 4, 5
    >>> mockfunc = cache(mockfunc)  # decorate
    >>> mockfunc(path2cache=mockpath2cache, a="hi", b="bye")
    [Inspect results in mock_cache.csv in folder tests/output_files/\
"mock_cache.csv"]


    """
    @wraps(method)
    def wrapped(*args, **kwargs):
        path2cache, init_args = {**kwargs}["path2cache"], {**kwargs} | dict(
            zip(["forms_csv", "tgt_lg", "src_lg"], [*args]))
        check_cache(path2cache, init_args)
        result = method(*args, **kwargs)
        write_to_cache(path2cache, init_args, result[1], result[2], result[3])
        # return result
    return wrapped


def eval_all(
                # Fllowing 9 params will go to loanpy.adrc.Adrc.__init__
                forms_csv,  # etymological data to evaluate (cldf's forms.csv)
                target_language,  # computational target language in forms.csv
                source_language,  # computational source language in forms.csv
                mode="adapt",  # adapt or reconstruct?
                most_frequent_phonotactics=9999999,  # howmany most freq
                phonotactic_inventory=None,  # chance to hard-code inventory
                connector=None,  # by default "<" for adapt &  "<*" 4 reconstr.
                scdictbase=None,  # not bool! Provide path, or dict, else None
                vfb=None,  # define placeholder vowels else None
                # These 5 go to both loanpy.adrc.Adrc.adapt AND .reconstruct
                guesslist=[10, 50, 100, 500, 1000],  # list of args 4 "howmany"
                clusters=False,  # filter (adapt) / slice (reconstr.) clusters
                phonotactics_filter=False,  # filter out wrong phonotactics?
                vowelharmony=False,  # repaire (adapt) / filter (reconstr)?
                sort_by_nse=False,  # sort results by likelihood?
                # These 5 go only to loanpy.adrc.Adrc.adapt if mode=="adapt"
                max_repaired_phonotactics=1,  # howmany phtct replacements max
                max_paths2repaired_phonotactics=1,  # CCV->CV: delete 1st/2nd C
                deletion_cost=100,  # cost for deleting a phoneme (TCRS)
                insertion_cost=49,  # cost for inserting a phoneme (TCRS)
                show_workflow=False,  # should workflow be displayed (adapt)
                # These 4 are for internal use
                path2cache=None,  # path to cache (optimal parameters)
                crossval=True,  # bool: Isolate word from training data?
                writesc=False,  # write sound corresp files? (for debugging)
                write_to=None  # "pathname.csv" to write and plot results
                ):
    """

    Trains cross-validated models, evaluates and visualises predictions. \
    23 arguments in total, only 3 positional. Best to run this from a loop \
    with different parameter settings and decorated with @loanpy.sanity.cache

    The first 9 args are passed on to \
    loanpy.adrc.Adrc.__init__, out of which the first 3 are positional:

    :param forms_csv: The path to CLDF's forms.csv of the etymological \
    dictionary. Will be used to initiate loanpy.adrc.Adrc. For more details \
    see \
    loanpy.helpers.read_forms.
    :type forms_csv: pathlib.PosixPath | str

    :param target_language: The computational target language. \
    Will be used to \
    initiate loanpy.adrc.Adrc. For more details see loanpy.helpers.Etym.
    :type target_language: str (options are listed in column "ID" in \
    cldf / etc / languages.tsv)

    :param source_language: The computational source language. \
    Will be used to \
    initiate loanpy.adrc.Adrc. For more details see loanpy.helpers.Etym.
    :type source_language: str (options are listed in column "ID" in \
    cldf / etc / languages.tsv)

    :param mode: Indicate whether predictions should be made with \
    loanpy.adrc.Adrc.reconstruct or loanpy.adrc.Adrc.adapt (also \
    sound correspondences \
    will be extracted with loanpy.qfysc.Qfy.get_sound_corresp in \
    the given mode). \
    See also loanpy.qfysc.Qfy for more details.
    :type mode: "adapt" | "reconstruct", default="adapt"

    :param most_frequent_phonotactics: The n most frequent structures \
    to accept into the target language's phonotactic inventory. \
    For more details \
    see loanpy.helpers.Etym.read_phonotactic_inv.
    :type most_frequent_phonotactics: int, default=9999999

    :param phonotactic_inventory: Chance to plug in phonotactic \
    inventory manually. If None, \
    will be extracted automatically from target language. For more details \
    see loanpy.helpers.Etym.read_phonotactic_inv.
    :type phonotactic_inventory: list, default=None

    :param connector: The string that connects the left side of an etymology \
    with the right side for adapting vs. reconstructing. For more details see \
    loanpy.qfysc.read_connector
    :type connector: iterable of len 2 | None, default=None

    :param scdictbase: Indicate whether sound correspondences based \
    on data should be combined with the (rather large) dictionary of \
    heuristic correspondences. For pitfalls see param <writesc>. Don't \
    pass a boolean. None means no, pathlib.PosixPath, dict, or str means yes.
    :type scdictbase: None | pathlib.PosixPath | dict, default=None

    :param vfb: Indicate whether there should be placeholder vowels. \
    For more details see loanpy.qfysc.Qfy.
    :type vfb: iterable of len 3 | None, default=None

    The next 5 args are passed on to loanpy.adrc.Adrc.adapt if \
    mode="adapt" but to loanpy.adrc.Adrc.reconstruct if mode="reconstruct":

    :param guesslist: The list of number of guesses to be made. Will be \
    passed into loanpy.adrc.Adrc.adapt's or loanpy.adrc.Adrc.reconstruct's \
    parameter <howmany> in a loop. Loop breaks as soon as \
    prediction was correct. \
    If results are plotted, this will serve as the x-axis, representing \
    the false positive rate. Each element of the list will be displayed \
    as a percentage of the last, i.e. highest element. For the default \
    setting this means [1%, 5%, 10%, 50%, 100%]. These values will also \
    be used to calculate the optimum of the ROC-curve.
    :type guesslist: list of int, default=[10, 50, 100, 500, 1000]

    :param clusters: Will be passed to loanpy.adrc.Adrc.adapt's parameter \
    <cluster_filter> or loanpy.adrc.Adrc.reconstruct's parameter \
    <clusterised>. \
    Indicate whether predictions that contain consonant or \
    vowel clusters that are not documented in the target language should be \
    filtered out if passed on to loanpy.adrc.Adrc.adapt or \
    if the tokeniser should \
    clusterise the input-word and look in the sound correspondence dictionary \
    for clusters as keys to predictions if passed on to \
    loanpy.adrc.Adrc.reconstruct.
    :type clusters: bool, default=False

    :param phonotactics_filter: Indicate if predictions made by \
    loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct \
    should be filtered out if they consist of a \
    phonotactic structure that is not contained in the language's \
    phonotactic inventory. For more details see \
    loanpy.helpers.Etym.read_phonotactic_inv.
    :type phonotactics_filter: bool, default=False

    :param vowelharmony: Will be passed to loanpy.adrc.Adrc.adapt's \
    parameter <repair_vowelharmony> or \
    loanpy.adrc.Adrc.reconstruct's parameter <vowelharmony_filter>. \
    Indicate whether vowel harmony should be repaired \
    if passed on to loanpy.adrc.Adrc.adapt or if results \
    violating the constraint \
    "front-back \
    vowelharmony" should be filtered out if passed \
    on to loanpy.adrc.Adrc.reconstruct.
    :type vowelharmony: bool, default=False

    :param sort_by_nse: Indicate if or how many predictions \
    should be sorted by likelihood. True and False will sort all or none, \
    just like float("inf") and 0. Passing an integer will not sort the entire \
    output. Instead it will pick as many \
    of the words with the highest NSE (likelihood) as indicated.
    :type sort_by_nse: bool | int, default=False

    The next 5 args are passed on to loanpy.adrc.Adrc.adapt if \
    mode="adapt" but if \
    mode="reconstruct" they will not be used.

    :param max_repaired_phonotactics: The maximum number of phonotactic \
    profiles into which \
    the original string should be transformed. Will be passed on to \
    loanpy.adrc.Adrc.adapt's parameter <max_repaired_phonotactics>
    :type max_repaired_phonotactics: int, default=1

    :param max_paths2repaired_phonotactics: The maximum number of \
    cheapest paths through which a \
    phonotactic structure can be repaired. Will be passed on to \
    loanpy.adrc.Adrc.adapt's parameter <max_paths2repaired_phonotactics>
    :type max_paths2repaired_phonotactics: int, default=1

    :param deletion_cost: The cost of deleting a phoneme
    :type deletion_cost: int, float, default=100

    :param insertion_cost: The cost of inserting a phoneme
    :type insertion_cost: int, float, default=49

    :param show_workflow: Indicate whether the workflow should be \
    displayed in the output of loanpy.sanity.eval_adapt. Useful for debugging.
    :type show_workflow: bool, default=False

    The next 4 args are not passed on but will be used in this module.

    :param path2cache: The path to the CSV-file in which information \
    about input-parameters and evaluated results is stored. If path points to \
    a non-existent file, the indicated file will be created. \
    If set to None, it will be written to CLDF's folder "etc" (concluded from \
    the path provided in parameter <forms_csv>) and will be called \
    f"opt_param_{target_language}_{source_language}". \
    If loanpy.sanity.cache is written as a decorator around this function, \
    it will receive this keyword argument, else it will be ignored. \
    For more information see loanpy.sanity.cache.
    :type path2cache: pathlib.PosixPath | str | None, default=None

    :param crossval: Indicate whether results should \
    be cross-validated. If true, \
    the model with which to predict an adaptation or a reconstruction \
    of a word will be trained without that word. That means that all 3 \
    inventories (phoneme, clusters, phonotactic) are also concluded \
    from data that excludes the specific row in the data frame, \
    where the word in question occurs. \
    Pitfall: If a word occurs multiple times in one data set, \
    this might blu the cross-validation.
    :type crossval: bool, default=True

    :param writesc: Indicate if loanpy.qfysc.Qfy.get_sound_corresp should \
    write its output to a file. If yes and if crossval is True, \
    provide a path to a **folder** (!). If yes and crossval is False, \
    provide a path to a file. \
    This is useful for debugging. \
    Careful: Since one file will be written in every \
    round of the cross-validation loop (as many iterations as there are \
    predictions to evaluate), the total storage room taken up by \
    the files can \
    get large. E.g. if 500 words should be evaluated and scdictbase is \
    1.6MB, the entire folder will take up 500*1.6MB. There are two ways to \
    avoid this: If writesc=True, make sure to set either crossval=False, \
    this will \
    only write one soundchange.txt file. Or set scdictbase=None, \
    since this is \
    the part that takes up the most storage. Predictions will be blurred \
    in both \
    cases but for debugging this is usually enough.
    :type writesc: bool, default=False

    :param write_to: Indicate whether results should be written to a \
    CSV-file and plotted to a jpg. If yes, provide the path to the file \
    including the ".csv"-extension. \
    None means that no file \
    will be written.
    :type write_to: None | pathlib.PosixPath | str, default=None

    :returns: Adds two columns to the input data frame: "guesses", \
    which indicate \
    how many guesses were necessary to make the correct prediction (if inf, \
    all predictions were wrong) \
and best guess, which shows the closest guess.
    :rtype: pandas.core.frame.DataFrame

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.sanity import eval_all, __file__
    >>> path2cog27 = Path(__file__).parent / "tests" \
    / "input_files" / "forms_27cogs.csv"
    >>> path2out = Path(__file__).parent / "tests" / "output_files" / \
    "eval_27cogs.csv"
    >>> eval_all(forms_csv=path2cog27, target_language="EAH", \
source_language="WOT", \
mode="reconstruct", clusters=True, sort_by_nse=True, write_to=path2out)
    [inspect result in folder tests/output_files/"eval_27cogs.csv"]

    """
    start = time()  # start timer

    # first 9 args
    adrc_obj = Adrc(
                    forms_csv=forms_csv,
                    source_language=source_language,
                    target_language=target_language,
                    mode=mode,
                    most_frequent_phonotactics=most_frequent_phonotactics,
                    phonotactic_inventory=phonotactic_inventory,
                    connector=connector,
                    scdictbase=scdictbase,
                    vfb=vfb)

    # next 10 args to adapt/reconstruct
    adrc_obj = loop_thru_data(
                                # first 7 go to adapt *and* reconstruct
                                adrc_obj,                   # 0  "self"
                                # X(1)  srcwrd inserted here in loop_thru_data
                                # X(2)  howmany inserted here in eval_one
                                clusters,                   # 1 (3)
                                phonotactics_filter,        # 2 (4)
                                vowelharmony,               # 3 (5)
                                sort_by_nse,                # 4 (6)
                                # next 5 only to loanpy.adrc.Adrc.adapt
                                max_repaired_phonotactics,        # 5 (7)
                                max_paths2repaired_phonotactics,  # 6 (8)
                                deletion_cost,              # 7 (9)
                                insertion_cost,             # 8 (10)
                                show_workflow,              # 9 (11)

                                guesslist,  # 10  # pop in eval_one
                                mode,  # 11 pop in eval_one, picks ad vs rc
                                writesc,  # 12 pop in loop_thru_data
                                crossval)  # 13 pop in loop_thru_data

    adrc_obj = postprocess(adrc_obj)
    stat = postprocess2(adrc_obj, guesslist, mode, write_to)

    end = time()

    return adrc_obj.dfety, stat, start, end


def loop_thru_data(*args):  # args range(14)
    """
    Called by loanpy.sanity.eval_all. Loops through the input \
data frame. cross-validates \
and writes down sound correspondences and inventories if indicated so. \
Calls loanpy.sanity.eval_one after \
concluding inventories and sound correspondences. \
If cross-validation is chosen, the row for prediction is isolated \
from the training data from which sound correspondences and inventories are \
calculated. Isolation of that row is \
reversed at the end of the loop latest - i.e. the taken out elements \
are all plugged in again so the loop can continue. Last two args (12, 13) \
will be popped, so only 0-12 are passed on to eval_one. The first argument \
must be an instance of loanpy.adrc.Adrc. This will also be the return value. \
The sourcre word is inserted at index 1 of the list of arguments. This is \
necessary so that all positional arguments are in the right place when \
passing them to loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct in \
loanpy.sanity.eval_adapt and loanpy.santiy.eval_recon.

    :param args: Arguments 0-14 that were passed to loanpy.sanity.eval_all \
(14 not included)
    :type args: mixed, see details below

    :param args[0]: This instance is created in \
loanpy.sanity.eval_all with its \
first 9 args
    :type args[0]: loanpy.adrc.Adrc

    :param args[1:14]: See documentation of params \
from <clusters> until <show_workflow> \
in loanpy.sanity.eval_all.
    :type args[1:14]: bool, int

    :returns: The (cross-)validated predictions. Serves as input for \
loanpy.sanity.postprocess.
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import loop_thru_data, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH")
    >>> loop_thru_data(adrc_obj, 1, 1, 100, 49, \
False, False, False, False, False, \
[10, 50, 100, 500, 1000], 'adapt', False, True).dfety
    Target_Form Source_Form  Cognacy  guesses best_guess
    0     aɣat͡ʃi    aɣat͡ʃːɯ        1      inf   KeyError
    1       aldaɣ       aldaɣ        2      inf   KeyError
    2        ajan        ajan        3      inf   KeyError
    """

    args, out = [*args], {}
    crossval, writesc, = args.pop(), args.pop()
    args.insert(1, 0)  # insert empty place for source word!
    # get non-crossvalidated sound correspondences if indicated
    if crossval is False:
        args[0] = get_noncrossval_sc(args[0], writesc)

    for idx2isolate, (srcwrd, tgtwrd) in tqdm(enumerate(zip(
            args[0].dfety.Source_Form, args[0].dfety.Target_Form))):
        # get crossvalidated sound correspondences if indicated
        if crossval is True:
            args[0] = get_crossval_data(args[0], idx2isolate, writesc)
        # make prediction from source word, check if target was hit
        args[1] = srcwrd  # insert sourceword into empty space created for it
        result_eval_one = eval_one(tgtwrd, *args)  # args 0-11 (incl)
        if out == {}:
            out = {k: [result_eval_one[k]] for k in result_eval_one}
        else:
            # update dict
            for key in out:
                out[key].append(result_eval_one[key])
        # plug dropped word back in to forms if necessary
        if crossval is True:
            args[0].forms_target_language.insert(
                args[0].idx_of_popped_word, args[0].popped_word)
    args[0].dfety = concat([args[0].dfety, DataFrame(out)], axis=1)
    return args[0]  # the adrc_obj


def eval_one(tgtwrd, *args):  # in: args 0-11 (incl). out: dict
    """
    Called by loanpy.sanity.loop_thru_data.
    Loops through the guesslist and adapts/reconstructs (depending on mode) \
until either the correct prediction was made or the guesslist has reached \
its end. Arguments <guesslist> and <mode> are removed from list of args, \
<howmany> is inserted at index 2.

    :param tgtwrd: The target word. This needs to be hit by the predictions. \
    *Hit* means prediction and target must be identical, \
    i.e. with an edit distance of zero.
    :type tgtwrd: str

    :param args[0]: This instance is created in \
    loanpy.sanity.eval_all with its \
    first 9 args and passed on to this function via \
    loanpy.sanity.loop_thru_data. It will be passed on to \
    loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct as param <self> \
    via \
    loanpy.sanity.eval_adapt and loanpy.santiy.eval_recon.
    :type args[0]: loanpy.adrc.Adrc

    :param args[1]: The input/source word from which predictions will \
    be made. \
    This was inserted into the list of args in lonapy.sanity.loop_thru_data \
    and will be passed on to param <ipastr> in \
    loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct via \
    loanpy.sanity.eval_adapt and loanpy.sanity.eval_recon
    :type args[1]: str

    :param args[2:6]: These 4 args will go to both, \
    loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct via \
    loanpy.sanity.eval_adapt and loanpy.sanity.eval_recon. For more details \
    see params from <clusters> to <sort_by_nse> \
    in loanpy.sanity.eval_all.
    :type args[2:6]: [bool, bool, bool, (bool | int)]

    :param args[6:11]: These 5 args will go only to \
    loanpy.adrc.Adrc.adapt via \
    loanpy.sanity.eval_adapt if param <mode> was set to \
    "adapt" in loanpy.santiy.eval_all (the default setting). \
    If mode was set to "reconstruct", these args will not be passed on. \
    For more details \
    see params <max_repaired_phonotactics> until <show_workflow> \
    in loanpy.sanity.eval_all.
    :type args[6:11]: [int, int, int, int, bool]

    :returns: A dictionary with at least two keys: Key "guesses" tells \
    either the index of the target word in the list of predictions, or, \
    if target was hit through a regular expression, it tells the integer \
    that was passed on to pararm <howmany> \
    in loanpy.adrc.Adrc.adapt and loanpy.adrc.Adrc.reconstruct \
    to make the correct prediction. Infinity means the target \
    was not hit, either due to a KeyError, or because of wrong predictions. \
    KeyErrors are more common when param <crossvalidate> = True, since \
    certain phonemes, clusters, or phonotactic profiles \
    occur only in the word that \
    is being isolated, i.e. they are missing from the training data and \
    therefore their keys are missing from the trained model. \
    Key "best_guess" shows the target word \
    if target word was hit, else it shows the first guess in the list of \
    predictions if predictions were made but target missed, \
    else it shows "KeyError" in case \
    param  <mode> was set to "adapt" (default) and a KeyError occurred \
    (This commonly happens if param  <scdictbase> in loanpy.sanity.eval_all \
    is set to None or {} (default), else IPA characters missing from the \
    trained model would be caught by the heuristics in scdictbase, which \
    is generated separately with loanpy.helpers.Etym.get_scdictbase) else \
    if mode was set to "reconstruct" it shows the characters missing from \
    the trained model together with the string "not old." \
    The best setting for param <sort_by_nse> = 1: The \
    actual best guess goes to the beginning of the list and will show \
    up in this dictionary, the rest of the \
    guesses remain unsorted to save time and energy. \
    If param <show_workflow> was set to "True" the dictionary will contain \
    additional keys that show the single steps through which the input word \
    was transformed. The number of workflow-keys varies \
    between 2-5, depending on the \
    settings passed on to loanpy.adrc.Adrc.adapt.
    :rtype: dict

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import eval_one, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> path2sc_ad = Path(__file__).parent / "tests" / \
"input_files" / "sc_ad_3cogs.txt"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH", scdictlist=path2sc_ad)
    >>> eval_one("dada", adrc_obj, "dada", False, \
False, False, False, 0, 1, 100, 49, True, [1], "adapt")
    {'guesses': 1,
    'best_guess': 'dada',
    'tokenised': "['d', 'a', 'd', 'a']",
    'adapted_phonotactics': "[['d', 'a', 'd', 'a']]",
    'before_combinatorics': "[[['d'], ['a'], ['d'], ['a']]]"}

    """
    args = [*args]
    eval_func = eval_adapt if args.pop() == "adapt" else eval_recon
    for guess in args.pop():  # guesslist
        out = eval_func(tgtwrd, *args[:2], guess, *args[2:])
        if out["guesses"] != float("inf"):
            break
    return out  # dict (for df), from eval_adapt or eval_recon


def eval_adapt(tgtwrd, *args):  # in: args 0+howmany+ args1-9(incl)
    """
    Called by loanpy.sanity.eval_one.
    Checks if target was hit by predictions made by loanpy.adrc.Adrc.adapt

    :param tgtwrd: See loanpy.sanity.eval_one
    :type tgtwrd: str

    :param args[:2]: See loanpy.sanity.eval_one <args[0]>, <args[1]>
    :type args[:2]: loanpy.adrc.Adrc, str

    :param args[2]: The number of guesses to be made. Will be passed on to \
    loanpy.adrc.Adrc.adapt's param \
    <howmany>. This number was concluded from param <guesslist> in \
    loanpy.sanity.eval_one.
    :type args[2]: list of int

    :param args[3:]: Remaining args that need to be passed on to \
    loanpy.adrc.Adrc.adapt.
    :type args[3:]: str, int, bool, bool, bool, (bool | int), int, int, \
    (int | float), (int | float), bool

    :returns: A dictionary with at least two keys: Key "guesses" tells \
    the index of the target word in the list of predictions. \
    Infinity means the target \
    was not hit, either due to a KeyError, or because of wrong predictions. \
    KeyErrors are more common when param <crossvalidate> = True, since \
    certain phonemes, clusters, or structures occur only in the word that \
    is being isolated, i.e. they are missing from the training data and \
    therefore their keys are missing from the trained model. \
    Key "best_guess" shows the target word \
    if target word was hit, else it shows the first guess in the list of \
    predictions if predictions were made but target missed, \
    else it shows the string "KeyError" in case \
    a KeyError occurred \
    (This commonly happens if param  <scdictbase> in loanpy.sanity.eval_all \
    is set to None or {} (default), else IPA characters missing from the \
    trained model would be caught by the heuristics in scdictbase, which \
    is generated separately with loanpy.helpers.Etym.get_scdictbase). \
    The best setting for param <sort_by_nse> = 1: The \
    actual best guess goes to the beginning of the list and will show \
    up in this dictionary, the rest of the \
    guesses remain unsorted to save time and energy. \
    If param <show_workflow> was set to "True" the dictionary will contain \
    additional keys that show the single steps through which the input word \
    was transformed. The number of workflow-keys varies between 2-5, \
    depending on the \
    settings passed on to loanpy.adrc.Adrc.adapt, whose documentation of \
    param <show_workflow> provides more information.
    :rtype: dict

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import eval_adapt, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> path2sc_ad = Path(__file__).parent / "tests" / \
"input_files" / "sc_ad_3cogs.txt"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH", scdictlist=path2sc_ad)
    >>> eval_adapt("daʃa", adrc_obj, "aldajd", \
10, False, False, False, False, 1, 1, 100, 49, True)
    {'guesses': inf,
    'best_guess': 'aldad',
    'tokenised': "['a', 'l', 'd', 'a', 'j', 'd']",
    'donor_phonotactics': 'VCCVCC',
    'predicted_phonotactics': "['VCCVC']",
    'adapted_phonotactics': "[['a', 'l', 'd', 'a', 'd']]",
    'before_combinatorics': "[[['a'], ['l'], ['d'], ['a'], ['d']]]"}

    """
    args, pred = [*args], []
    args[2], args[3], args[4] = get_howmany(args[2], args[3], args[4])
    try:
        pred = Adrc.adapt(*args).split(", ")
        # best guess = target+1 if hit (e.g index 0 means 1 guess needed)
        try:
            out = {"guesses": pred.index(tgtwrd)+1, "best_guess": tgtwrd}
        except ValueError:
            # first prediction if not hit
            out = {"guesses": float("inf"), "best_guess": pred[0]}
    except KeyError:
        # this if predictions not made
        out = {"guesses": float("inf"), "best_guess": "KeyError"}
    if args[-1] is True:
        out |= args[0].workflow  # merge
    return out  # dict


def eval_recon(tgtwrd, *args):
    """
    Called by loanpy.sanity.eval_one.
    Checks if target was hit by predictions made by \
    loanpy.adrc.Adrc.reconstruct

    :param tgtwrd: See loanpy.sanity.eval_one
    :type tgtwrd: str

    :param args[:2]: See loanpy.sanity.eval_one <args[0]>, <args[1]>
    :type args[:2]: loanpy.adrc.Adrc, str

    :param args[2]: The number of guesses to be made. Will be passed on to \
    loanpy.adrc.Adrc.reconstruct's param \
    <howmany>. This number was concluded from param <guesslist> in \
    loanpy.sanity.eval_one.
    :type args[2]: list of int

    :param args[3:7]: Remaining args that need to be passed on to \
    loanpy.adrc.Adrc.reconstruct.
    :type args[3:7]: str, int, bool, bool, bool, (bool | int)

    :param args[7:]: These do nothing. It's just slicker to pass \
    on all args to loanpy.adrc.Adrc.reconstruct \
    than having to slice them first.
    :type args[7:]: irrelevant

    :returns: A dictionary with exactly two keys: Key "guesses" tells \
    either the index of the target word in the list of predictions, or, \
    if target was hit through a regular expression, the integer \
    that was passed on to pararm <howmany> \
    in loanpy.adrc.Adrc.reconstruct \
    to make the correct prediction. Infinity means the target \
    was not hit, either due to a KeyError, or because of wrong predictions. \
    KeyErrors are more common when param <crossvalidate> = True, since \
    certain phonemes, clusters, or structures occur only in the word that \
    is being isolated, i.e. they are missing from the training data and \
    therefore their keys are missing from the trained model. \
    Key "best_guess" shows the target word \
    if target word was hit, else it shows the first guess in the list of \
    predictions if predictions were made but target missed, \
    else if a KeyError occurred, it shows the characters missing from \
    the trained model together with the string "not old." \
    If target was hit through a reg-ex of the type ^(a|b)(c|d)$, \
    the entire reg-ex will be returned as "best_guess". \
    The best setting for param <sort_by_nse> = 1: The \
    actual best guess goes to the beginning of the list and will show \
    up in this dictionary, the rest of the \
    guesses remain unsorted to save time and energy.
    :rtype: dict

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import eval_recon, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> path2sc_rc = Path(__file__).parent / "tests" / \
"input_files" / "sc_rc_3cogs.txt"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="H", \
target_language="EAH", scdictlist=path2sc_rc)
    >>> eval_recon("anaat͡ʃi", adrc_obj, \
"aːruː", 1, True, False, False, True)
    {'guesses': 1, 'best_guess': 'anaat͡ʃi'}

    """
    # make prediction
    pred = Adrc.reconstruct(*args)
    # define two conditions with which the output will be evaluated
    # is pred-reg-ex IN tgtwrd?
    short_regex, target_hit = "(" in pred, bool(search(pred, tgtwrd))
    # return output based on those 2 conditions
    if short_regex and target_hit:
        out = args[2], pred
    elif "not old" in pred:
        out = float("inf"), pred
    elif short_regex and not target_hit:
        out = float("inf"), pred
    elif not short_regex and target_hit:
        out = pred[1:-1].split("$|^").index(tgtwrd)+1, tgtwrd
    elif not short_regex and not target_hit:
        out = float("inf"), pred[1:-1].split("$|^")[0]
    else:
        out = float("inf"), pred
    return {"guesses": out[0], "best_guess": out[1]}


def get_noncrossval_sc(adrc_obj, writesc):
    """
    Called by loanpy.sanity.loop_thru_data. \
    Get non-crossvalidated sound correspondences.

    :param adrc_obj: This instance is created \
in loanpy.sanity.eval_all with its \
    first 9 args
    :type adrc_obj: loanpy.adrc.Adrc

    :param writesc: Should sound correspondences be written? \
    Provide a path to a *folder* (!) if yes. For pitfalls see \
    sanity.eval_all param <writesc>.

    :returns: a loanpy.adrc.Adrc object with sound correspondences \
    assigned to its attribute \
    <scdict>, sum of examples to <sedict> and phonotactic correspondences \
    to <scdict_phonotactics>.
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import get_noncrossval_sc, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH")
    >>> adrc_obj = get_noncrossval_sc(adrc_obj, None)
    >>> adrc_obj.scdict
    {'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'],
    'n': ['n'], 't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']}
    >>> adrc_obj.sedict
    {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1,
    'l<l': 1, 'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2}
    >>> adrc_obj.scdict_phonotactics
    {'VCCVC': ['VCCVC', 'VCVC', 'VCVCV'],
     'VCVC': ['VCVC', 'VCCVC', 'VCVCV'],
     'VCVCV': ['VCVCV', 'VCVC', 'VCCVC']}
    """
    # if not crossvalidate just get sound corresp from big file
    (adrc_obj.scdict, adrc_obj.sedict, _,  # get sound correspondences
     adrc_obj.scdict_phonotactics, _, _) = adrc_obj.get_sound_corresp(writesc)
    return adrc_obj


def get_crossval_data(adrc_obj, idx, writesc=None):
    """
    Called by loanpy.sanity.loop_thru_data. \
Get sound correspondences by dropping the indicated row to isolate from \
the data frame, \
and extracting sound correspondences and inventories \
from the data without the dropped row. \
Pitfall: If same row occurs multiple times, cross-validation is blurred.

    :param adrc_obj: loanpy.adrc.Adrc. Contains \
attribute <dfety>, which is the etymological data for training \
(type: pandas.core.frame.DataFrame).
    :type adrc_obj: loanpy.adrc.Adrc

    :param idx: Index of the row to drop from the etymological data
    :type idx: int

    :param writesc: Indicate whether the sound correspondence \
    files (results of \
training) should be written. If None, they will not be written. \
If they should \
be written, a path to a *folder* has to be provided since this function \
will be called for every round of the main loop in \
loanpy.sanity.eval_all and \
multiple files will be written. See param writesc in eval_all for pitfalls.
    :type writesc: None | pathlib.PosixPath (to folder!) | str, default=None

    :returns: The same instance of loanpy.adrc.Adrc that was passed \
to param <adrc_obj> but \
with the cross-validated \
model passed into its attributes for looking up sound correspondences \
(i.e. <scdict>, <sedict>, <scdict_struc>).
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import get_crossval_data, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH")
    >>> adrc_obj = get_crossval_data(adrc_obj, 0, None)
    >>> # first cog isolated, missing sc: a ɣ a t͡ʃ i - a ɣ a t͡ʃː ɯ
    >>> adrc_obj.scdict
    {'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'], 'n': ['n'], 'ɣ': ['ɣ']}
    >>> adrc_obj.sedict
    {'a<a': 4, 'd<d': 1, 'j<j': 1, 'l<l': 1, 'n<n': 1, 'ɣ<ɣ': 1}
    >>> adrc_obj.scdict_phonotactics
    {'VCCVC': ['VCCVC', 'VCVC'], 'VCVC': ['VCVC', 'VCCVC']}

    """
    # memorise dropped row to plug back in at end of this function
    dropped_row = DataFrame(dict(adrc_obj.dfety.iloc[idx]), index=[idx])
    # drop the indicated row for training
    # isolate
    adrc_obj.dfety = adrc_obj.dfety.drop([adrc_obj.dfety.index[idx]])
    # create filename for corssvalidated training results
    if writesc:
        writesc = writesc / f"sc{idx}isolated.txt"
    # get index of popped word in forms
    adrc_obj.idx_of_popped_word = adrc_obj.forms_target_language.index(
        dropped_row.at[idx, "Target_Form"])
    # pop isolated word from forms, from which inventories are concluded
    adrc_obj.popped_word = adrc_obj.forms_target_language.pop(
        adrc_obj.idx_of_popped_word)
    # get crossvalidated inventories from crossvalidated forms
    (adrc_obj.phoneme_inventory, adrc_obj.cluster_inventory,
     adrc_obj.phonotactic_inventory) = adrc_obj.get_inventories()
    # popped word will be plugged in in loop_thru_data at end of loop
    # because adapt & reconstruct use inventories for filters etc.
    # train model on crossvalidated data (chosen row is isolated)
    (adrc_obj.scdict, adrc_obj.sedict, _,  # get sound correspondences
     adrc_obj.scdict_phonotactics, _, _) = adrc_obj.get_sound_corresp(writesc)
    # dropped row plugged in again so df can be reused in next round of loop
    # can be done here b/c only qfysc uses dfety, not used in adapt/reconstruct
    adrc_obj.dfety = concat([adrc_obj.dfety.head(idx),
                             dropped_row,  # re-insert isolated row
                             adrc_obj.dfety.tail(len(adrc_obj.dfety)-idx)])
    return adrc_obj


def postprocess(adrc_obj):
    """
    Called by loanpy.sanity.eval_all. \
    Evaluates the consistency and quality of both, \
    the etymological data, and the predictions. \
    Takes the return value of sanity.loop_thru_data as input. Needs columns \
    "guesses" and "best guess" to make calculations, which are: \
    a) Calculates the NSEs between source and target words, as well as \
    source words and best guesses. This results in 2*4=8 columns, since \
    each of the 4 elements of the tuple returned by loanpy.adrc.Adrc.get_nse. \
    gets its own column. b) Checks if the correct phonotactics \
    were predicted or not, \
    in case param <show_workflow> is True \
    and <max_repaired_phonotactics> is greater than zero in \
    loanpy.sanity.eval_all. \
    c) Calculates the Levenshtein Distance and its \
    normalised version between target words and best guesses. \
    This is an alternative way of measuring by how much the target \
    was missed, than the false positive rate. \
    Useful when comparing the results of the rule-based model with an AI,
    since AIs usually only make 1 guess, so there is no false positive rate \
    for evaluation there. Currently, there is no formula of how to convert \
    the Levenshtein Distance into a number of false positives. The keyword \
    here is "Levenshtein Distance neighbourhood problem".


    :param adrc_obj: This is returned by sanity.loop_thru_data
    :type adrc_obj: loanpy.adrc.Adrc

    :returns: An adrc_obj with 10+ additional columns: 2*4=8 about NSE. \
    1 optional one called "phonotactics_predicted" \
and n (default: 2) about the phonological distances by which the \
predictions missed the target.
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import postprocess, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> path2sc_ad = Path(__file__).parent / "tests" / \
"input_files" / "sc_ad_3cogs.txt"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH", scdictlist=path2sc_ad)
    >>> # pretend guesses are already made
    >>> adrc_obj.dfety["best_guess"] = ["aɣa", "bla", "ajan"]
    >>> df = postprocess(adrc_obj).dfety
    >>> for col in df.columns:
    >>>     print(df[col])
    0    aɣat͡ʃi
    1      aldaɣ
    2       ajan
    Name: Target_Form, dtype: object
    0    aɣat͡ʃːɯ
    1       aldaɣ
    2        ajan
    Name: Source_Form, dtype: object
    0    1
    1    2
    2    3
    Name: Cognacy, dtype: int64
    0     aɣa
    1     bla
    2    ajan
    Name: best_guess, dtype: object
    0    3.2
    1    3.2
    2    3.5
    Name: NSE_Source_Target_Form, dtype: float64
    0    16
    1    16
    2    14
    Name: SE_Source_Target_Form, dtype: int64
    0    [6, 2, 6, 1, 1]
    1    [6, 1, 1, 6, 2]
    2       [6, 1, 6, 1]
    Name: E_distr_Source_Target_Form, dtype: object
    0    ['a<a', 'ɣ<ɣ', 'a<a', 't͡ʃ<t͡ʃː', 'i<ɯ']
    1         ['a<a', 'l<l', 'd<d', 'a<a', 'ɣ<ɣ']
    2                ['a<a', 'j<j', 'a<a', 'n<n']
    Name: align_Source_Target_Form, dtype: object
    0    2.80
    1    1.17
    2    3.50
    Name: NSE_Source_best_guess, dtype: float64
    0    14
    1     7
    2    14
    Name: SE_Source_best_guess, dtype: int64
    0       [6, 2, 6, 0, 0]
    1    [0, 0, 1, 0, 6, 0]
    2          [6, 1, 6, 1]
    Name: E_distr_Source_best_guess, dtype: object
    0        ['a<a', 'ɣ<ɣ', 'a<a', 'C<t͡ʃː', 'V<ɯ']
    1    ['V<a', 'b<C', 'l<l', 'C<d', 'a<a', 'C<ɣ']
    2                  ['a<a', 'j<j', 'a<a', 'n<n']
    Name: align_Source_best_guess, dtype: object
    0    4
    1    3
    2    0
    Name: fast_levenshtein_distance_best_guess_Target_Form, dtype: int64
    0    0.57
    1    0.60
    2    0.00
    Name: fast_levenshtein_distance_div_maxlen_best_guess_Target_Form, \
dtype: float64

    """
    adrc_obj = get_nse4df(adrc_obj, "Target_Form")
    adrc_obj = get_nse4df(adrc_obj, "best_guess")
    adrc_obj = phonotactics_predicted(adrc_obj)
    adrc_obj = get_dist(adrc_obj, "best_guess")
    return adrc_obj


def postprocess2(adrc_obj, guesslist, mode, write_to=None):
    """
    Called by loanpy.sanity.eval_all.
    2nd post-processing. Calls loanpy.sanity.get_tpr_fpr_opt to get the \
    true positive rate, false positive rate and the optimum. Calculates \
    statistics from that, writes output files as .csv and .jpg if \
    indicated so.

    :param adrc_obj: The return value of loanpy.sanity.postprocess.
    :type adrc_obj: loanpy.adrc.Adrc

    :param guesslist: The list of number of guesses that are \
    looped through in loanpy.sanity.eval_one and passed on to \
    param  <howmany> in loanpy.adrc.Adrc.adapt and \
    loanpy.adrc.Adrc.reconstruct via loanpy.sanity.eval_adapt and \
    loanpy.sanity.eval_recon. Defined in loanpy.sanity.eval_all.
    :type guesslist: list of int

    :param mode: Defined in loanpy.sanity.eval_all.
    :type mode: "adapt" | "reconstruct", default="adapt"

    :param write_to: None, if no output should be written - this is the \
    preferred setting if loanpy.sanity.eval_all is run from a loop. \
    Else a path to a file including the ".csv" extension in the file name \
    should be provided. There will be another file written with the same \
    name but with a .jpg ending instead.
    :type write_to: pathlib.PosixPath | str | None

    :returns: The return value of loanpy.sanity.make_stat
    :rtype: tuple of int, str, str

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import postprocess2, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> path2sc_ad = Path(__file__).parent / "tests" / \
"input_files" / "sc_ad_3cogs.txt"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH", scdictlist=path2sc_ad)
    >>> # pretend guesses are already made
    >>> adrc_obj.dfety["guesses"] = [1, 2, 3]
    >>> postprocess2(adrc_obj, [4, 5, 6], "adapt")
    (5, '3/3', '100%')

    """

    tpr_fpr_opt = get_tpr_fpr_opt(
                                    # howmany guesses were NEEDED
                                    adrc_obj.dfety.guesses,
                                    guesslist,   # how many guesses were MADE
                                    len(adrc_obj.dfety)
                                    )

    stat = make_stat(
                        tpr_fpr_opt[2][2],
                        tpr_fpr_opt[2][1],
                        guesslist[-1],
                        len(adrc_obj.dfety)
                        )

    if write_to:
        plot_roc(
                    guesslist,
                    Path(str(write_to)[:-4]+".jpg"),
                    tpr_fpr_opt,
                    stat[0],
                    stat[2],
                    len(adrc_obj.dfety),
                    mode
                    )

        adrc_obj.dfety.to_csv(
                                write_to,
                                encoding="utf_8",
                                index=False
                                )

    return stat


def get_nse4df(adrc_obj, tgt_col):
    """
    Called by loanpy.sanity.postprocess. \
    Calcuclates NSE between column Source_Form and given target column

    :param adrc_obj: The return value of loanpy.sanity.loop_thru_data
    :type adrc_obj: loanpy.adrc.Adrc

    :param tgt_col: The name of the target column \
    in adrc_obj.dfety, to which to compare its column \
    "Source_Form" to, to establish the normalised sum of examples (NSE).
    :type tgt_col: str

    :returns: The same adrc object as the input was but with 4 columns added: \
    normalised sum of examples (nse), sum of examples (se), examples (e), \
    and the alignment.
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import get_nse4df, __file__
    >>> path2forms = Path(__file__).parent / "tests" / \
"input_files" / "forms_3cogs_wot.csv"
    >>> path2sc_ad = Path(__file__).parent / "tests" / \
"input_files" / "sc_ad_3cogs.txt"
    >>> adrc_obj = Adrc(forms_csv=path2forms, source_language="WOT", \
target_language="EAH", scdictlist=path2sc_ad)
    >>> get_nse4df(adrc_obj, "Target_Form").dfety
    [Similar large output as in the example for loanpy.sanity.postprocess,
     just last two columns missing, [3 rows x 7 columns]]

    """

    col1, col2 = adrc_obj.dfety[tgt_col], adrc_obj.dfety["Source_Form"]
    if adrc_obj.mode == "reconstruct":
        col1, col2 = col2, col1  # flip it!

    adrc_obj.dfety = concat(
        [adrc_obj.dfety, DataFrame([adrc_obj.get_nse(tgt, src)
         for tgt, src in zip(col1, col2)],
            columns=[f"NSE_Source_{tgt_col}", f"SE_Source_{tgt_col}",
                     f"E_distr_Source_{tgt_col}",
                     f"align_Source_{tgt_col}"])], axis=1)

    return adrc_obj


def phonotactics_predicted(adrc_obj):
    """
    Called by loanpy.sanity.postprocess. \
    Checks if phonotactic profile of target word was predicted or not \
    and adds that information to a new column called \
    "phonotactics_predicted". If \
    column "predicted_phonotactics" is missing from \
    adrc_obj.dfety a KeyError is \
    triggered and the input object is returned without any changes. \
    One way to avoid a KeyError is setting param <show_workflow> to True \
    and <max_repaired_phonotactics> > 0 in \
    loanpy.sanity.eval_all because this will create the column \
    "predicted_phonotactics".

    :param adrc_obj: The output value of loanpy.sanity.get_nse4df
    :type adrc_obj: loanpy.adrc.Adrc

    :returns: The same object as inputted, but optionally with an \
    extra column "phonotactics_predicted".
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from pandas import DataFrame
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import phonotactics_predicted, __file__
    >>> adrc_obj = Adrc()
    >>> adrc_obj.dfety = DataFrame({"Target_Form": ["abc", "def", "ghi"], \
"predicted_phonotactics": [["CCC", "VVV"], ["CVC"], ["CCV", "CCC"]]})
    >>> phonotactics_predicted(adrc_obj).dfety
      Target_Form predicted_phonotactics  phonotactics_predicted
    0         abc             [CCC, VVV]                   False
    1         def                  [CVC]                    True
    2         ghi             [CCV, CCC]                    True


    """

    try:
        adrc_obj.dfety[
            "phonotactics_predicted"] = [True if
                                         adrc_obj.word2phonotactics(actual)
                                         in pred else False for actual, pred
                                         in zip(adrc_obj.dfety["Target_Form"],
                                                adrc_obj.dfety[
                                                    "predicted_phonotactics"])]
    except KeyError:
        pass
    return adrc_obj


def get_dist(adrc_obj, col, dst_msrs=["fast_levenshtein_distance",
                                      "fast_levenshtein_distance_div_maxlen"]):
    """
    Called by loanpy.sanity.postprocess.
    Calculates the Levenshtein Distance and the normalised Levenshtein \
    Distance between Target and best guess. This is the best way to \
    compare the performance of the rule-based model with that of an AI.

    :param adrc_obj: The output value of loanpy.sanity.phonotactics_pred
    :type adrc_obj: loanpy.adrc.Adrc

    :param col: The name of the column with the words to which the \
    distance to the target \
    words should be calculated.
    :type col: str

    :param dst_msrs: The list of distance measures that should be calculated. \
    For an exhaustive list of input options see param \
    <phondist_msr> in loanpy.loanfinder.Search.
    :type dst_msrs: list of str, default=["fast_levenshtein_distance", \
    "fast_levenshtein_distance_div_maxlen"]

    :returns: The same object as inputted but with the same amount of \
    extra columns as the list provided in param <dst_msrs> is long.
    :rtype: loanpy.adrc.Adrc

    :Example:

    >>> from pathlib import Path
    >>> from pandas import DataFrame
    >>> from loanpy.adrc import Adrc
    >>> from loanpy.sanity import get_dist, __file__
    >>> adrc_obj = Adrc()
    >>> adrc_obj.dfety = DataFrame({\
"best_guess": ["will not buy", "record", "scratched"], \
"Target_Form": ["won't buy", "tobacconists", "scratched"]})
    >>> df = get_dist(adrc_obj, "best_guess").dfety
    >>> for col in df.columns:
    >>>     print(df[col])
    0    will not buy
    1          record
    2       scratched
    Name: best_guess, dtype: object
    0       won't buy
    1    tobacconists
    2       scratched
    Name: Target_Form, dtype: object
    0     5
    1    10
    2     0
    Name: fast_levenshtein_distance_best_guess_Target_Form, dtype: int64
    0    0.42
    1    0.83
    2    0.00
    Name: fast_levenshtein_distance_div_maxlen_best_guess_Target_Form, \
dtype: float64


    """
    dist = Distance()  # PanPhon
    for dst_msr in dst_msrs:  # all chosen distance measures get a col
        new_col = []
        msr = getattr(dist, dst_msr)  # turn str into method of Distance obj
        for idx, row in adrc_obj.dfety.iterrows():  # loop thru df
            bg, tf = row[col], row["Target_Form"]  # best guess, target form
            if any(ban in bg for ban in BANNED):  # red flags to ignore
                new_col.append(float("inf"))  # means no distance calculable
            else:
                new_col.append(round(msr(bg, tf), 2))  # calculate distance
        adrc_obj.dfety[f"{dst_msr}_{col}_Target_Form"] = new_col  # add to col

    return adrc_obj  # same as input obj but with new cols in adrc_obj.dfety


def make_stat(opt_fp, opt_tp, max_fp, len_df):
    """
    Called by loanpy.sanity.postprocess2.
    Calculates  statistics from optimum, max nr of guesses and length \
    of input data frame.

    :param opt_fp: The optimal false positive rate as a fraction of the \
    maximal false positive rate, i.e. last (=highest) element of \
    list passed to param <guesslist> in loanpy.sanity.eval_all.
    :type opt_fp: float

    :param opt_tp: The optimal true positive rate as a fraction of the \
    total number of input words for predictions, i.e. length of data frame.
    :type opt_tp: float

    :param max_fp: The maximal false positive rate is the \
    highest number of possible guesses, i.e. the last element of the list \
    passed to param <guesslist> in loanpy.sanity.eval_all.
    :type max_fp: int | float

    :param len_df: The total number of input words for predictions.
    :type len_df: int

    :returns: The optimal setting for param <howmany> in \
    loanpy.adrc.Adrc.adapt or loanpy.adrc.Adrc.reconstruct.
    :rtype: tuple of int, str, str

    :Example:

    >>> from loanpy.sanity import make_stat
    >>> make_stat(opt_fp=0.099, opt_tp=0.6, max_fp=1000, len_df=10)
    (100, "6/10", "60%")
    """
    # howmany = 1 more than fp (opt_fp is a % of max_guess)
    opt_howmany = round(opt_fp*max_fp) + 1
    # how many did it find out of all, e.g. 10/100
    opt_tp_str = str(round(opt_tp*len_df)) + "/" + str(len_df)
    # how many percent is that, e.g. 10/100 would be 10%
    opt_tpr = str(round(opt_tp*100)) + "%"

    return opt_howmany, opt_tp_str, opt_tpr


def get_tpr_fpr_opt(guesses_needed, guesses_made, len_df):
    """
    Called by loanpy.sanity.postprocess2. \
    Get the true positive rate, the false positive rate and the optimum.

    :param guesses_needed: This is the column "guesses" in adrc_obj.dfety, \
    as it is calculated by loanpy.sanity.loop_thru_data.
    :type guesses_needed: pandas.core.series.Series | list | iterable \
    of float | int

    :param guesses_made: Value passed on to param \
    <guesslist> in loanpy.sanity.eval_all.
    :type guesses_made: pandas.core.series.Series | list | iterable \
    of float | int

    :param len_df: The length of the first two params passed.
    :type len_df: int

    :returns: A tuple of the true positive rate, as a list, \
    the false positive rate as a list (i.e. "if this many guesses were \
    made, this many words were correctly predicted in total"), and the \
    optimum, which in turn is again a tuple of the \
    lowest possible difference between \
    each point of the true positive rate and the false positive rate \
    (they are both percentages, so subtraction makes sense), the given \
    point of the true positive rate and the given point of the false positive \
    rate.
    :rtype: tuple of (list of int, list of int, tuple of (int, int, int))

    :Example:

    >>> from loanpy.sanity import get_tpr_fpr_opt
    >>> gn = [10, None, 20, 4, 17, None, None, 8, 9, 120]
    >>> gm = [1, 3, 5, 7, 9, 99, 999]
    >>> get_tpr_fpr_opt(\
guesses_needed=gn, \
guesses_made=gm, \
len_df=10)
    ([0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
    [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
    (0.501, 0.6, 0.099))
    """
# keep only rows that are guess or lower = the correctly
# identified ones in the given round
# divide that number by the max amount of true positives
# -> e.g. 10/119 were correct if 100 guesses were made
    tpr, fpr = [], []
    for guess in guesses_made:  # loop through fpr
        tpr.append(round(len([
                                i for i in guesses_needed
                                if i and i <= guess])/len_df, 3))
        # how much of a fraction of the max amount of guess is our current fpr
        # e.g. 2K out out 10K would be 0.2 = 20%
        fpr.append(round(guess/(guesses_made[-1]), 3))

    # I'm gleb to see you
    # https://www.youtube.com/watch?v=jl7bMLf-f9A
    optimum = max([(tp-guess, tp, guess) for tp, guess in zip(tpr, fpr)])

    return tpr, fpr, optimum


def plot_roc(guesslist, plot_to, tpr_fpr_opt, opt_howmany, opt_tpr,
             len_df, mode):
    """
    Plots an ROC-curve: True positive rate goes on the x-axis, false \
    positive rate on the y-axis. Optimum is marked with an "x" on the curve.

    :param guesslist: Value passed on to param \
    <guesslist> in loanpy.sanity.eval_all.
    :type guesslist: list of int

    :param plot_to: Concluded from param \
    <write_to> in loanpy.sanity.eval_all, by clipping off ".csv" and \
    replacing it by ".jpg".
    :type plot_to: pathlib.PosixPath | str

    :param tpr_fpr_opt: Return value of loanpy.sanity.get_tpr_fpr_opt
    :type tpr_fpr_opt: tuple of \
    (list of int, list of int, tuple of (int, int, int))

    :param opt_howmany: The optimal setting of parameter <howmany> in \
    loanpy.adrc.Adrc.adapt or loanpy.adrc.Adrc.reconstruct. The first \
    element of the tuple returned by loanpy.sanity.make_stat.
    :type opt_howmany: int

    :param opt_tpr: The optimal true positive rate. The last element of \
    the tuple returned by loanpy.sanity.make_stat. Shows the true positive \
    rate when param <howmany> in loanpy.adrc.Adrc.adapt or \
    loanpy.adrc.Adrc.reconstruct is set to the optimum.
    :type opt_tpr: float

    :param len_df: The length of the input data frame, i.e. how many words \
    there were to make predictions for
    :type len_df: int

    :param mode: If reconstructing or adapting (horizontal vs. \
    vertical transfers)
    :type mode: "adapt" | "reconstruct", default="adapt"

    :returns: Plots an ROC-curve and writes it to a file
    :rtype: None.


    :Example:

    >>> from pathlib import Path
    >>> from loanpy.sanity import plot_roc
    >>> path2mockplot = Path(__file__).parent \
/ "tests" / "output_files" / "mockplot.jpg"
    >>> plot_roc(guesslist=[1,2,3],
    >>> plot_to=path2mockplot,
    >>> tpr_fpr_opt=([0.0, 0.0, 0.1, 0.1, 0.3, 0.6, 0.7],
    >>> [0.001, 0.003, 0.005, 0.007, 0.009, 0.099, 1.0],
    >>> (0.501, 0.6, 0.099)),
    >>> opt_howmany=1,
    >>> opt_tpr=0.6, len_df=3, mode="adapt")
    [inspect result in tests/output_file / "mockplot.jpg"]

    .. image:: ../../tests/output_files/mockplot.jpg
    """
    xlabel(f'fpr (100%={guesslist[-1]})')
    ylabel(f'tpr (100%={len_df})')
    plot(tpr_fpr_opt[1], tpr_fpr_opt[0], label=f'ROC-curve')
    scatter(tpr_fpr_opt[2][2], tpr_fpr_opt[2][1], marker='x', c='blue',
            label=f"Optimum:\nhowmany={opt_howmany} -> tpr: {opt_tpr}")
    title(f'Predicting with loanpy.adrc.Adrc.{mode}')
    legend()

    savefig(plot_to)
    clf()  # important, else old plot keeps getting overwritten


def check_cache(path2cache, init_args):
    """
    Called by loanpy.sanity.eval_all. \
Checks if cache-file exists, if not: empty file is created, \
if yes: checks whether init_args occur in one of its rows. If yes: \
Error is raised. If no, nothing happens. Input args are the output args of \
loanpy.sanity.eval_all.

    :param path2cache: The path to the CSV-file in which information \
    about input-parameters and evaluated results is stored. If path points to \
    a non-existent file, the correct file will be created. \
    If set to None, it will be written to CLDF's folder etc (concluded from \
    the path provided in parameter <forms_csv>) and will be called \
    f"opt_param_{tgt_lg}_{src_lg}". \
    For more information see loanpy.sanity.check_cache.
    :type path2cache: pathlib.PosixPath | str

    :param init_args: Dictionary where keys are the arguments of \
loanpy.sanity.eval_all and values their \
assigned value. Generated by locals(). \
See help(locals).
    :type init_args: dict

    :raises ArgumentsAlreadyTested: If these arguments were already passed \
to loanpy.sanity.eval_all once, they won't be calculated again. \
To solve this error, simply don't decorate the function in question with \
loanpy.sanity.cache

    :returns: Just raises an Error or writes an empty cache under \
certain conditions, else no action
    :rtype: None
    """

    try:  # this means that the file already exists
        # loop through file
        for idx, row in read_csv(
                path2cache, usecols=list(init_args)).fillna("").iterrows():
            # check whether given parameters were already run once
            if list(map(str, list(init_args.values()))
                    ) == list(map(str, list(row))):
                # if yes, raise error and specify the row where they are stored
                raise ArgumentsAlreadyTested(f"These arguments were tested \
already, see {path2cache} line {idx+1}! (start counting at 1 in 1st row)")

    except FileNotFoundError:  # if cache doesn't exist, create empty one
        # columns are the args with which eval_all was run
        DataFrame(columns=list(
            init_args)+["opt_tpr", "optimal_howmany", "opt_tp", "timing",
                        "date"]).to_csv(path2cache, index=False,
                                        encoding="utf-8")
        # as well as evaluation columns eval_all will create
        # write empty cache to file at indicated location


def write_to_cache(path2cache, init_args, stat, start, end):
    """
    Writes the results of loanpy.sanity.eval_all to a CSV and sorts it \
    by column "opt_tpr". So that on the top of the .csv will be \
    the parameter settings that give the most accurate predictions.

    :param path2cache: The path to which the cache should be written.
    :type path2cache: pathlib.PosixPath | str

    :param init_args: The arguments that were passed on to \
    loanpy.sanity.eval_all to generate the results.
    :type init_args: list

    :param stat: The return value of loanpy.sanity.make_stat.
    :type stat: tuple of int, str, str

    :param start: The start time when loanpy.sanity.eval_all started.
    :type start: float

    :param end: The end time when loanpy.sanity.eval_all ended.
    :type end: float

    :returns: Stores the args with which loanpy.sanity_eval_all was run \
    and its results in a CSV-file.
    :rtype: None
    """

    for i in init_args:
        init_args[i] = str(init_args[i])  # to make df out of it

    # concat old and new cache, sort, write to csv
    concat([read_csv(path2cache),
            DataFrame(
                init_args, index=[0]).assign(
                        optimal_howmany=[stat[0]],
                        opt_tp=[stat[1]], opt_tpr=[stat[2]],
                        timing=[strftime("%H:%M:%S",
                                         gmtime(end-start))],
                        date=[datetime.now().strftime("%x %X")]
                        )]).sort_values(
                                        by=['opt_tpr'],
                                        ascending=False,
                                        ignore_index=True).to_csv(
                                            path2cache, index=False,
                                            encoding="utf_8_sig")
