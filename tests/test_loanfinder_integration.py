"""integration test for loanpy.loanfinder.py (2.0 BETA) for pytest 7.1.1"""

from inspect import ismethod
from os import remove
from pathlib import Path

from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
from pandas import DataFrame, Series, read_csv
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import raises

from loanpy.adrc import Adrc
from loanpy.helpers import plug_in_model
from loanpy.loanfinder import NoPhonMatch, Search, gen, read_data

PATH2READ_DATA = Path(__file__).parent / "input_files" / "ad_read_data.csv"
PATH2SC_AD = Path(__file__).parent / "input_files" / "sc_ad_likeliest.txt"
PATH2SC_RC = Path(__file__).parent / "input_files" / "sc_rc_likeliest.txt"


def test_read_data():
    """test if data is being read correctly"""

    # setup expected outcome, path, input-dataframe, mock pandas.read_csv
    srsexp = Series(["a", "blub", "club"], name="col1", index=[0, 1, 1])

    # assert that the actual outcome equals the expected outcome for adapted
    assert_series_equal(read_data(PATH2READ_DATA, "col1"), srsexp)

    # overwrite data for reading reconstructed stuff
    srsexp = Series(["(a)?", "(b|c)"], name="col1", index=[1, 3])
    path = Path(__file__).parent / "input_files" / "rc_read_data.csv"
    # assert expected and actual outcome are the same pandas Series for
    # reconstr
    assert_series_equal(read_data(path, "col1"), srsexp)

    del path, srsexp


def test_gen():
    assert list(gen([1, 2, 3], [4, 5, 6], lambda x, y: x + y)) == [5, 7, 9]
    # make sure *args after "prefix" are passed into the function (z=1)
    assert list(gen([1, 2, 3], [4, 5, 6],
                    lambda x, y, z: x + y + z, "", 1)) == [6, 8, 10]


def test_init():
    """test if class Search is initiated correctly"""

    # check whether initiation runs without providing args
    search_inst = Search()
    assert len(search_inst.__dict__) == 12

    # assert initiation went properly
    search_inst.search_in is None  # 1
    assert search_inst.search_for is None  # 2
    assert search_inst.phondist == 0  # 3
    assert ismethod(search_inst.phondist_msr)  # 4s
    assert search_inst.donpath is None  # 5
    assert search_inst.recpath is None  # 6
    assert search_inst.doncol == "ad"  # 7
    assert search_inst.reccol == "rc"  # 8
    assert search_inst.semsim == 1  # 9
    assert search_inst.semsim_msr.__name__ == "gensim_multiword"  # 10
    assert ismethod(search_inst.get_nse_ad)  # 11
    assert ismethod(search_inst.get_nse_rc)  # 12

    # assert initiation runs with advanced parameter settings
    path2don = Path(__file__).parent / "input_files" / "got.csv"
    path2rec = Path(__file__).parent / "input_files" / "hun.csv"
    # initiate Search() with real parameters
    search_inst = Search(
        path2donordf=path2don,
        path2recipdf=path2rec,
        donorcol="ad",
        recipcol="rc",
        scdictlist_ad=Path(__file__
                           ).parent / "input_files" / "sc_ad_3cogs.txt",
        scdictlist_rc=Path(__file__
                           ).parent / "input_files" / "sc_rc_3cogs.txt")

    assert len(search_inst.__dict__) == 12

    # assert initiation went properly
    assert_series_equal(
        search_inst.search_in, Series(["z"], name="ad", index=[0]))  # 1
    assert_series_equal(
        search_inst.search_for, Series(["z"], name="rc", index=[0]))  # 2
    assert search_inst.phondist == 0  # 3
    assert ismethod(search_inst.phondist_msr)  # 4s
    assert search_inst.donpath == path2don  # 5
    assert search_inst.recpath == path2rec  # 6
    assert search_inst.doncol == "ad"  # 7
    assert search_inst.reccol == "rc"  # 8
    assert search_inst.semsim == 1  # 9
    assert search_inst.semsim_msr.__name__ == "gensim_multiword"  # 10
    assert ismethod(search_inst.get_nse_ad)  # 11
    assert ismethod(search_inst.get_nse_rc)  # 12

    # tear down
    del search_inst, path2don, path2rec


def test_phonmatch_small():
    """test if most similar/identical words are found in wordlist,
    while words in which to search have to be passed through param"""

    # set pandas Series in which to search for words later on
    srs_srch_in = Series(["a", "blub", "club"], name="ad", index=[0, 1, 1])

    # set up mock instancec of Search class
    search_inst = Search()

    # run test if param dropduplicates == False
    assert_frame_equal(
        search_inst.phonmatch_small(
            search_in=srs_srch_in,
            search_for="(b|c)?lub",
            index=99,
            dropduplicates=False),
        DataFrame(
            {
                "match": [
                    "blub",
                    "club"],
                "recipdf_idx": [
                    99,
                    99]},
            index=[
                1,
                1]))

    # run test if param dropduplicates == True
    assert_frame_equal(
        search_inst.phonmatch_small(
            search_in=srs_srch_in,
            search_for="(b|c)?lub",
            index=99,
            dropduplicates=True),
        DataFrame(
            {
                "match": ["blub"],
                "recipdf_idx": [99]},
            index=[1]))

    # test if parameters allow matches to have a higher phon. dist. than 0
    # dropduplicates=True (default setting)
    search_inst = Search(phondist=0.5)
    # if search for regex, phondist must = 0!!
    assert_frame_equal(search_inst.phonmatch_small(
        search_in=srs_srch_in, search_for="blub", index=99),
                       DataFrame({"match": ["blub"], "recipdf_idx": [99]},
                                 index=[1]))

    # tear down
    del srs_srch_in, search_inst


def test_phonmatch():
    """same as test_phonmatch_small but words to searchin are passed through
    class initiation rather than function arg"""

    # set up mock instancec of Search class
    search_inst = Search(path2donordf=PATH2READ_DATA, donorcol="col1")

    # run test if param dropduplicates == False
    assert_frame_equal(search_inst.phonmatch(
        search_for="(b|c)?lub", index=99, dropduplicates=False),
        DataFrame({"match": ["blub", "club"], "recipdf_idx": [99, 99]},
                  index=[1, 1]))

    # run test if param dropduplicates == True
    assert_frame_equal(search_inst.phonmatch(
        search_for="(b|c)?lub", index=99, dropduplicates=True),
        DataFrame({"match": ["blub"], "recipdf_idx": [99]},
                  index=[1]))

    # test if parameters allow matches to have a higher phon. distance than 0
    # dropduplicates=True (default setting)
    search_inst = Search(path2donordf=PATH2READ_DATA,
                         donorcol="col1", phondist=0.5)
    assert_frame_equal(search_inst.phonmatch(
        # if search for regex, phondist must = 0!!
        search_for="blub", index=99),
        DataFrame({"match": ["blub"], "recipdf_idx": [99]},
                  index=[1]))

    # tear down
    del search_inst


def test_likeliestphonmatch():

    search_inst = Search(path2donordf=PATH2READ_DATA, donorcol="col1",
                         scdictlist_ad=PATH2SC_AD, scdictlist_rc=PATH2SC_RC)

    assert_frame_equal(search_inst.likeliestphonmatch(
        donor_ad="a, blub, club", recip_rc="(b|c)?lub",
        donor_segment="elub", recip_segment="dlub"),
        DataFrame({"match": ["blub"],
                   "nse_rc": [10],
                   "se_rc": [50],
                   "distr_rc": str([10] * 5),
                   "align_rc": "['#-<*-', '#dl<*bl', \
'u<*u', 'b#<*b', '-#<*-']",
                   "nse_ad": [4],
                   "se_ad": [20],
                   "distr_ad": "[0, 0, 10, 10, 0]",
                   "align_ad": "['e<V', 'C<b', 'l<l', 'u<u', 'b<b']",
                   "nse_combined": [14]}),
        check_dtype=False)

    # tear down
    del search_inst


def test_loans():
    # create instance of class
    search_inst = Search(
        path2donordf=Path(__file__).parent /
        "input_files" /
        "loans_got.csv",
        path2recipdf=Path(__file__).parent /
        "input_files" /
        "loans_hun_NoPhonMatch.csv")

    # test first break
    # assert that error is raised correctly
    # b/c no phonological matches were found
    with raises(NoPhonMatch) as nophonmatch_mock:
        search_inst.loans()
    assert str(nophonmatch_mock.value) == "no phonological matches found"

    # test with merge_with_rest=False, postprocess=False, write=True
    # these would be just some bonus parameters,
    # we're testing the core functionality now

    # create instance of class
    search_inst = Search(
        path2donordf=Path(__file__).parent / "input_files" / "loans_got.csv",
        path2recipdf=Path(__file__).parent / "input_files" / "loans_hun.csv",
        semsim=0.1)

    # plug in dummy vectors, api would need internet + a minute to load
    plug_in_model(word2vec.Word2Vec(common_texts, min_count=1).wv)

    search_inst.loans()
    assert_frame_equal(search_inst.loans(),
                       DataFrame({'match': ["blub"],
                                  'recipdf_idx': [0],
                                  'Meaning_x': ["computer, interface"],
                                  'Meaning_y': ["human"],
                                  'gensim_multiword': [0.10940766]}),
                       check_dtype=False)

    # test with semsim==0.2: no matches found bc all meanings are less similar

    # create instance of class
    search_inst = Search(
        path2donordf=Path(__file__).parent / "input_files" / "loans_got.csv",
        path2recipdf=Path(__file__).parent / "input_files" / "loans_hun.csv",
        semsim=0.2)

    assert_frame_equal(search_inst.loans(), DataFrame({
        'match': [], 'recipdf_idx': [], 'Meaning_x': [],
        'Meaning_y': [], 'gensim_multiword': []}), check_dtype=False)

    # test with advanced settings.
    path2dummy_results = Path(__file__
                              ).parent / "loans_result_integration_test.csv"

    # create instance of class. semsim=0.1 again so matches are found
    search_inst = Search(
        path2donordf=Path(__file__).parent / "input_files" / "loans_got.csv",
        path2recipdf=Path(__file__).parent / "input_files" / "loans_hun.csv",
        semsim=0.1, scdictlist_ad=PATH2SC_AD, scdictlist_rc=PATH2SC_RC)

    search_inst.loans(
        postprocess=True,
        merge_with_rest=True,
        write_to=path2dummy_results)

    assert_frame_equal(read_csv(path2dummy_results),
                       DataFrame({
                           'recipdf_idx': [0],
                           'gensim_multiword': [0.10940766],
                           'match': ["blub"],
                           'nse_rc': [10],
                           'se_rc': [50],
                           'distr_rc': [str([10] * 5)],
                           'align_rc': "['#-<*-', '#dl<*bl', \
'u<*u', 'b#<*b', '-#<*-']",
                           'nse_ad': [5],
                           'se_ad': [20],
                           'distr_ad': [str([0, 10, 10, 0])],
                           'align_ad': "['b<b', 'l<l', 'u<u', 'b<b']",
                           'nse_combined': [15],
                           "Segments_x": ["b l u b"],
                           "Meaning_x": ["human"],
                           "ad": ["blub, club"],
                           "bla_x": ["xyz"],
                           "Segments_y": ["d l u b"],
                           "Meaning_y": ["computer, interface"],
                           "rc": ["(b|c)?lub"],
                           "bla_y": ["xyz"]}), check_dtype=False)

    # tear down
    plug_in_model(None)
    remove(path2dummy_results)
    del search_inst, path2dummy_results


def test_postprocess():
    """test if postprocessing works fine with test input data"""
    # set up: define input data frame
    dfin = DataFrame({"match": ["blub"], "recipdf_idx": [0], "Meaning_x": [
                     "computer, interface"],
        "Meaning_y": ["human"], "semsim_msr": [0.10940766]})

    # set up: define expected output data frame
    dfexp = DataFrame(
        {
            "recipdf_idx": [0],
            "Meaning_x": ["computer, interface"],
            "Meaning_y": ["human"],
            "semsim_msr": [0.10940766],
            "match": ["blub"],
            "nse_rc": [10],
            "se_rc": [50],
            "distr_rc": str(
                [10] * 5),
            "align_rc": "['#-<*-', '#dl<*bl', 'u<*u', 'b#<*b', '-#<*-']",
            "nse_ad": [5],
            "se_ad": [20],
            "distr_ad": "[0, 10, 10, 0]",
            "align_ad": "['b<b', 'l<l', 'u<u', 'b<b']",
            "nse_combined": [15]})

    # create instance of class
    search_inst = Search(
        path2donordf=Path(__file__).parent / "input_files" / "loans_got.csv",
        path2recipdf=Path(__file__).parent / "input_files" / "loans_hun.csv",
        scdictlist_ad=PATH2SC_AD, scdictlist_rc=PATH2SC_RC,
        semsim=0.2)
    assert_frame_equal(search_inst.postprocess(dfin), dfexp, check_dtype=False)

    # tear down
    del dfin, dfexp, search_inst


def test_merge_with_rest():
    """test if postprocessed output is merged with original cols"""
    search_inst = Search(
        path2donordf=Path(__file__).parent / "input_files" / "loans_got.csv",
        path2recipdf=Path(__file__).parent / "input_files" / "loans_hun.csv",
        scdictlist_ad=PATH2SC_AD, scdictlist_rc=PATH2SC_RC)
    # mock input data frame
    dfin = DataFrame({"a": ["pi", "pa", "po"],
                      "b": ["bi", "ba", "bo"],
                      "Meaning_x": ["mi", "ma", "mo"],
                      "Meaning_y": ["ni", "na", "no"],
                      "recipdf_idx": [0, 0, 0],
                      "mocksemsim": [98, 99, 100],
                      "gensim_multiword": [0.4, 0.2, 0.9]})

    dfexp = DataFrame({"a": ["pi"],
                       "b": ["bi"],
                       "recipdf_idx": [0],
                       "mocksemsim": [98],
                       "gensim_multiword": [0.4],
                       "Segments_x": ["b l u b"],
                       "Meaning_x": ["human"],
                       "ad": ["blub, club"],
                       "bla_x": ["xyz"],
                       "Segments_y": ["d l u b"],
                       "Meaning_y": ["computer, interface"],
                       "rc": ["(b|c)?lub"],
                       "bla_y": ["xyz"]})

    assert_frame_equal(search_inst.merge_with_rest(dfin), dfexp)

    del dfin, dfexp
