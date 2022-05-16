from collections import OrderedDict
from inspect import ismethod
from os import remove
from pathlib import Path

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from loanpy.adrc import Adrc, read_scdictlist

PATH2FORMS = Path(__file__).parent / "input_files" / "forms_3cogs_wot.csv"
PATH2SC_TEST = Path(__file__).parent / "input_files" / "sc_ad_3cogs.txt"
PATH2SC_HANDMADE = Path(__file__).parent / "input_files" / "sc_ad_handmade.txt"
PATH2SC_HANDMADE2 = Path(__file__).parent / "input_files" / "sc_ad_handmade2.txt"
PATH2SC_HANDMADE3 = Path(__file__).parent / "input_files" / "sc_rc_handmade.txt"

def test_read_scdictlist():
    """test if list of sound correspondence dictionaries is read correctly"""

    # set up: creat a mock list of dicts and write it to file
    dict0 = {"dict0": "szia"}
    dict1 = {"dict1": "cső"}
    out = [dict0, dict1, dict0, dict1]
    PATH2SC_TEST = Path(__file__).parent / "test_read_scdictlist.txt"
    with open(PATH2SC_TEST, "w") as f:
        f.write(str(out))

    # assert mock dict list is read in correctly
    assert read_scdictlist(PATH2SC_TEST) == out

    # tear down
    remove(PATH2SC_TEST)
    del dict0, dict1, out, PATH2SC_TEST

def test_move_sc():
    pass #unit == integration test (no patches)

def test_init():
    """test if the Adrc-class is initiated properly"""

    #check if initiation without args works fine
    adrc_inst = Adrc()
    assert len(adrc_inst.__dict__) == 16

    # 6 attributes initiated in Adrc, rest inherited
    assert adrc_inst.scdict is None
    assert adrc_inst.sedict is None
    assert adrc_inst.edict is None
    assert adrc_inst.scdict_struc is None
    assert adrc_inst.workflow == OrderedDict()

    #5 attributes inherited from Qfy
    assert adrc_inst.mode == "adapt"
    assert adrc_inst.connector == "<"
    #assert adrc_inst.nsedict == {}
    assert adrc_inst.scdictbase == {}
    assert adrc_inst.vfb is None

    #7 attributes inherited from Etym via Qfy
    assert isinstance(adrc_inst.phon2cv, dict)
    assert len(adrc_inst.phon2cv) == 6358
    assert isinstance(adrc_inst.vow2fb, dict)
    assert len(adrc_inst.vow2fb) == 1240
    assert adrc_inst.dfety is None
    assert adrc_inst.phoneme_inventory is None
    assert adrc_inst.clusters is None
    assert adrc_inst.struc_inv is None
    ismethod(adrc_inst.distance_measure)

    #assert initiation runs correctly with non-empty params as well

    #set up fake sounndchange.txt file
    d0, d1, d2, d3 = [
    {'a': ['a'], 'd': ['d'], 'j': ['j'], 'l': ['l'], 'n': ['n'], 't͡ʃː': ['t͡ʃ'], 'ɣ': ['ɣ'], 'ɯ': ['i']},
    {'a<a': 6, 'd<d': 1, 'i<ɯ': 1, 'j<j': 1, 'l<l': 1, 'n<n': 1, 't͡ʃ<t͡ʃː': 1, 'ɣ<ɣ': 2},
    {'a<a': [1, 2, 3], 'd<d': [2], 'i<ɯ': [1], 'j<j': [3], 'l<l': [2], 'n<n': [3], 't͡ʃ<t͡ʃː': [1], 'ɣ<ɣ': [1, 2]},
    {'VCCVC': ['VCCVC', ''], 'VCVC': ['VCVC', ''], 'VCVCV': ['VCVCV', '']}]

    adrc_inst = Adrc(
        scdictlist=PATH2SC_TEST,
        formscsv=PATH2FORMS,
        srclg="WOT", tgtlg="EAH",
        mode="reconstruct",
        struc_most_frequent=2)

    assert len(adrc_inst.__dict__) == 16

    # assert initiation went correctly
    assert adrc_inst.scdict == d0
    assert adrc_inst.sedict == d1
    assert adrc_inst.edict == d2
    assert adrc_inst.scdict_struc == d3
    assert adrc_inst.workflow == OrderedDict()

    #5 attributes inherited from Qfy
    assert adrc_inst.mode == "reconstruct"
    assert adrc_inst.connector == "<*"
    #assert adrc_inst.nsedict == {}
    assert adrc_inst.scdictbase == {}
    assert adrc_inst.vfb is None

    #7 attributes inherited from Etym via Qfy
    assert isinstance(adrc_inst.phon2cv, dict)
    assert len(adrc_inst.phon2cv) == 6358
    assert isinstance(adrc_inst.vow2fb, dict)
    assert len(adrc_inst.vow2fb) == 1240
    assert_frame_equal(adrc_inst.dfety, DataFrame({"Target_Form": ["aɣat͡ʃi", "aldaɣ", "ajan"],
    "Source_Form": ["aɣat͡ʃːɯ", "aldaɣ", "ajan"],  "Cognacy": [1, 2, 3]}))
    assert adrc_inst.phoneme_inventory == {'a', 'd', 'i', 'j', 'l', 'n', 't͡ʃ', 'ɣ'}
    assert adrc_inst.clusters == {'a', 'ia', 'j', 'ld', 'n', 't͡ʃ', 'ɣ'}
    assert adrc_inst.struc_inv == ['VCVCV', 'VCCVC']
    ismethod(adrc_inst.distance_measure)

    #don't remove yet,
    #remove("test_soundchanges.txt")

def test_get_diff():
    """test if the difference is calculated correctly
    between the first two sound of a sound correspondence list"""

    # create instance
    adrc_inst = Adrc(
        scdictlist=PATH2SC_TEST,
        formscsv=PATH2FORMS,
        srclg="WOT", tgtlg="EAH",
        mode="adapt",
        struc_most_frequent=2)

    # assert
    assert adrc_inst.get_diff(
        sclistlist=[["d", "x", "$"], ["a", "x", "$"], ["d", "x", "$"], ["a", "x", "$"]],
        ipa=["d", "a", "d", "a"]) == [1, 6, 1, 6]

    assert adrc_inst.get_diff(
        sclistlist=[["d", "x", "$"], ["a", "$"], ["d", "x", "$"], ["a", "$"]],
        ipa=["d", "a", "d", "a"]) == [1, float("inf"), 1, float("inf")]

    assert adrc_inst.get_diff( #test if second exception works
        sclistlist=[["x", "x", "$"], ["a", "x", "$"], ["x", "x", "$"], ["a", "x", "$"]],
        ipa=["k", "a", "k", "a"]) == [9999999, 6, 9999999, 6]

    assert adrc_inst.get_diff(
        sclistlist=[["x", "x", "$"], ["x", "x", "$"], ["x", "x", "$"], ["x", "x", "$"]],
        ipa=["k", "i", "k", "i"]) == [9999999]*4

    del adrc_inst

def test_read_sc():
    """test if sound correspondences are read correctly"""

    # set up mock class, plug in mock scdict, mock tokenise, mock math.prod
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE)

    # assert
    assert adrc_inst.read_sc(ipa="dade", howmany=1) == [["d"], ["a"], ["d"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=2) == [["d", "tʰ"], ["a"], ["d"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=3) == [["d", "tʰ"], ["a", "e"], ["d"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=4) == [["d", "tʰ"], ["a", "e"], ["d"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=5) == [["d", "tʰ"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=6) == [["d", "tʰ"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=7) == [["d", "tʰ"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=12) == [["d", "tʰ", "t"], ["a", "e"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=18) == [["d", "tʰ", "t"], ["a", "e", "i"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=24) == [["d", "tʰ", "t", "tː"], ["a", "e", "i"], ["d", "tʰ"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=36) == [["d", "tʰ", "t", "tː"], ["a", "e", "i"], ["d", "tʰ", "t"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=48) == [["d", "tʰ", "t", "tː"], ["a", "e", "i"], ["d", "tʰ", "t", "tː"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=64) == [["d", "tʰ", "t", "tː"], ["a", "e", "i", "o"], ["d", "tʰ", "t", "tː"], ["y"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=128) == [["d", "tʰ", "t", "tː"], ["a", "e", "i", "o"], ["d", "tʰ", "t", "tː"], ["y", "u"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=160) == [["d", "tʰ", "t", "tː"], ["a", "e", "i", "o", "u"], ["d", "tʰ", "t", "tː"], ["y", "u"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=240) == [["d", "tʰ", "t", "tː"], ["a", "e", "i", "o", "u"], ["d", "tʰ", "t", "tː"], ["y", "u", "e"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=99999999) == [["d", "tʰ", "t", "tː"], ["a", "e", "i", "o", "u"], ["d", "tʰ", "t", "tː"], ["y", "u", "e"]]
    assert adrc_inst.read_sc(ipa="dade", howmany=float("inf")) == [["d", "tʰ", "t", "tː"], ["a", "e", "i", "o", "u"], ["d", "tʰ", "t", "tː"], ["y", "u", "e"]]

    #tear down
    del adrc_inst

def test_reconstruct():
    """test if reconstructions based on sound correspondences work"""

    # test first break: some sounds are not in scdict

    # set up adrc instance
    adrc_inst = Adrc(mode="reconstruct",
        scdictlist=Path(__file__).parent / "input_files" / "sc_rc_3cogs.txt")
#
#    # assert reconstruct works when sound changes are missing from data
    assert adrc_inst.reconstruct(
        ipastring="kiki") == "#k, i, k, i# not old"
#
#    # test 2nd break: struc and vowelharmony are False
    assert adrc_inst.reconstruct(
        ipastring="aːruː", clusterised=False) == "^(a)(n)(a)(at͡ʃi)$"
#
#    # test same break again but param <howmany> is greater than 1 now
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=False,
    howmany=2) == "^(a)(n)(a)(at͡ʃi|ɣ)$"

    # overwrite adrc instance, now with formscsv, so struc_inv is read.
    adrc_inst = Adrc(mode="reconstruct", formscsv=PATH2FORMS,
    srclg="H", tgtlg="EAH",
        scdictlist=Path(__file__).parent / "input_files" / "sc_rc_3cogs.txt")

#    #test with struc=True (a filter)
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=False,
    howmany=2, struc=True) == "^anaɣ$"
#
#    # assert reconstruct works with struc=True and result being empty
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=False,
    howmany=1, struc=True) == "wrong phonotactics"

#    # same as previous one but with clusterised=True
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=True,
    howmany=1, struc=True) == "wrong phonotactics"

    # assert filter vowelharmony works
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=True,
    howmany=2, struc=False, vowelharmony=True) == "^anaat͡ʃi$|^anaɣ$"

    #let's assume "i" was a back vowel
    adrc_inst.vow2fb["i"] = "B"
    #then it would would filter ^anaat͡ʃi$ out, since it has back and front vow
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=True,
    howmany=2, struc=False, vowelharmony=True) == "^anaɣ$"
    # test vowelharmony=True, result is empty
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=True,
    howmany=1, struc=False, vowelharmony=True) == "wrong vowel harmony"
    #tear down
    adrc_inst.vow2fb["i"] = "F"

#    # test sort_by_nse=True
#    # assert reconstruct works and sorts the result by nse
    assert adrc_inst.reconstruct(ipastring="aːruː", clusterised=True,
    howmany=2, struc=False, vowelharmony=False,
    sort_by_nse=True) == "^anaɣ$|^anaat͡ʃi$"

    del adrc_inst

def test_adapt_struc():
    """test if phonotactic structures are adapted correctly"""

    # test first break

    # create instance
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE,
        formscsv=PATH2FORMS,
        srclg="WOT", tgtlg="EAH")

    # assert adapt_struc works with max_struc=1
    assert adrc_inst.adapt_struc(ipa_in="kiki", max_struc=1) == [
    ['k', 'i', 'k', 'i']]
    #but also with max_struc=2
    assert adrc_inst.adapt_struc(ipa_in="kiki", max_struc=2) == [['k', 'i', 'k'], ['k', 'i', 'C', 'k', 'i']]
#
#    # test struc missing from dict and rank_closest instead, test show_workflow
     # pretend scdict_struc is empty:
    adrc_inst.scdict_struc = {}
    assert adrc_inst.adapt_struc(ipa_in="kiki", max_struc=2,
                        show_workflow=True) == [['V', 'k', 'i', 'k', 'i'], ['i', 'k', 'i', 'C']]
    assert adrc_inst.workflow == OrderedDict([('donor_struc', ['CVCV']), ('pred_strucs', [['VCVCV', 'VCVC']])])

def test_adapt():
    """test if words are adapted correctly with sound correspondence data"""

    # basic settings
    # create instance
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE,
        formscsv=PATH2FORMS,
        srclg="WOT", tgtlg="EAH")

    #assert adapt is working
    assert adrc_inst.adapt(ipa_in="dade",
    howmany=5) == "dady, datʰy, dedy, detʰy, tʰady"

    # change max_struc to 2 from 1.
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=6,
                    max_struc=2) == "dad, ded, tʰad, tʰed, dajdy, dejdy"

    # change max_paths to 2 from 1.
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=6,
                    max_struc=2,
                    max_paths=2) == "dad, tʰad, dajdy, tʰajdy, dadjy, tʰadjy"

    # let's assume e is a back vowel and repair vowel harmony
    adrc_inst.vow2fb["e"] = "B"
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=6,
                    max_struc=2,
                    max_paths=2,
                    vowelharmony=True) == "dad, tʰad, dajdæ, tʰajdæ, dujdy, tʰujdy"

    #let's assume 'CVCCV' was an allowed structure
    adrc_inst.struc_inv.add('CVCCV')
    #apply filter where unallowed structures are filtered out
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=6,
                    max_struc=2,
                    max_paths=2,
                    vowelharmony=True,
                    struc_filter=True) == "dajdæ, tʰajdæ, dujdy, tʰujdy, dadjæ, tʰadjæ"

    #test with clusterised=True
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE2, #different scdictlist
        formscsv=PATH2FORMS,
        srclg="WOT", tgtlg="EAH")
    adrc_inst.struc_inv.add('CVCCV') #add this so struc_filter won't be empty
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=1000,
                    max_struc=2,
                    max_paths=2,
                    vowelharmony=True,
                    struc_filter=True,
                    clusterised=True) == "t͡ʃalda"

    #let more things go through filter clusterised:
    adrc_inst.clusters.add("d")
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=1000,
                    max_struc=2,
                    max_paths=2,
                    vowelharmony=True,
                    struc_filter=True,
                    clusterised=True,) == "dalda, t͡ʃalda"

    #sort result by nse (likelihood of reconstruction)
    adrc_inst.clusters.add("d")
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=1000,
                    max_struc=2,
                    max_paths=2,
                    vowelharmony=True,
                    struc_filter=True,
                    clusterised=True,
                    sort_by_nse=True) == "t͡ʃalda, dalda"

    #test show_workflow - run adapt first, then check workflow
    assert adrc_inst.adapt(
                    ipa_in="dade",
                    howmany=1000,
                    max_struc=2,
                    max_paths=2,
                    vowelharmony=True,
                    struc_filter=True,
                    clusterised=True,
                    sort_by_nse=True,
                    show_workflow=True) == "t͡ʃalda, dalda"

    assert adrc_inst.workflow == OrderedDict([('tokenised',
[['d', 'a', 'd', 'e']]), ('donor_struc', ['CVCV']), ('pred_strucs',
[['CVC', 'CVCCV']]), ('adapted_struc', [[['d', 'a', 'd'],
['d', 'a', 'C', 'd', 'e'], ['d', 'a', 'd', 'C', 'e']]]),
('adapted_vowelharmony',
[[['d', 'a', 'd'], ['d', 'a', 'C', 'd', 'e'],
  ['d', 'a', 'd', 'C', 'e']]]),
('before_combinatorics',
[[[['d', 't͡ʃ'], ['a', 'e'], ['d', 't͡ʃ']],
[['d', 't͡ʃ'], ['a', 'e'], ['l'], ['d', 't͡ʃ'], ['e', 'a']],
  [['d', 't͡ʃ'], ['a', 'e'], ['d', 't͡ʃ'], ['l'], ['e', 'a']]]])])

    #tear down
    del adrc_inst

def test_get_nse():
    #assert with mode=="adapt" (default)
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE,
        formscsv=PATH2FORMS,
        srclg="WOT", tgtlg="EAH")

    #assert with basic setting
    assert adrc_inst.get_nse("dade", "dady") == 33.25
    #assert with se=False
    assert adrc_inst.get_nse("dade", "dady", se=False) == [[1, 2], [0], [1, 2], [3]]
    #assert with show_workflow=True
    assert adrc_inst.get_nse("dade", "dady", show_workflow=True) == (
    33.25, 133, [1, 6, 1, 125]
    )

    #assert with mode=="reconstruct"
    adrc_inst = Adrc(
        scdictlist=PATH2SC_HANDMADE3,
        formscsv=PATH2FORMS,
        srclg="H", tgtlg="EAH",
        mode="reconstruct")

    #assert with basic setting
    assert adrc_inst.get_nse("ɟɒloɡ", "jɑlkɑ") == 6.666666666666667
    #assert with se=False
    assert adrc_inst.get_nse("ɟɒloɡ", "jɑlkɑ", se=False) == [[1, 2], [3, 4], [5], [6, 7, 8], [9], 0]
    #assert with show_workflow=True
    assert adrc_inst.get_nse("ɟɒloɡ", "jɑlkɑ", show_workflow=True) == (
    6.666666666666667, 40, [10, 9, 8, 7, 6, 0])

    #tear down
    del adrc_inst













                #
