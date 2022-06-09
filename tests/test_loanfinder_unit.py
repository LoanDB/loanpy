"""unit test for loanpy.loanfinder.py (2.0 BETA) for pytest 7.1.1"""

from inspect import ismethod
from os import remove
from pathlib import Path
from unittest.mock import patch, call

from pandas import DataFrame, RangeIndex, Series, read_csv
from pandas.testing import (assert_frame_equal, assert_index_equal,
                            assert_series_equal)
from pytest import raises

from loanpy.loanfinder import Search, gen, read_data, NoPhonMatch
from loanpy import loanfinder as lf


def test_read_data():
    """test if data is being read correctly"""

    # setup expected outcome, path, input-dataframe, mock pandas.read_csv
    srsexp = Series(["a", "b", "c"], name="col1", index=[0, 1, 1])
    path = Path(__file__).parent / "test_read_data.csv"
    dfin = DataFrame({"col1": ["a", "b, c", "wrong clusters",
                      "wrong phonotactics"], "col2": [1, 2, 3, 4]})
    with patch("loanpy.loanfinder.read_csv") as read_csv_mock:
        read_csv_mock.return_value = dfin

        # assert that the actual outcome equals the expected outcome
        assert_series_equal(read_data(path, "col1"), srsexp)

    # assert mock call to read_csv_mock was correct
    assert read_csv_mock.call_args_list[0] == call(
        path, encoding="utf-8", usecols=["col1"])

    # test read recip

    # setup: overwrite expected outcome and input-dataframe, mock
    # pandas.read_csv
    srsexp = Series(["(a)?", "(b|c)"], name="col1", index=[1, 3])
    dfin = DataFrame({"col1": ["wrong vowel harmony", "(a)?",
                     "wrong phonotactics", "(b|c)"], "col2": [1, 2, 3, 4]})
    with patch("loanpy.loanfinder.read_csv") as read_csv_mock:
        read_csv_mock.return_value = dfin

        # assert expected and actual outcome are the same pandas Series
        assert_series_equal(read_data(path, "col1"), srsexp)

    # assert mock was called with correct input
    assert read_csv_mock.call_args_list[0] == call(
        path, encoding="utf-8", usecols=["col1"])

    # tear down
    del path, dfin, srsexp


def test_gen():
    """test if generator yields the right things"""

    # set up mock-tqdm (which is a progress bar)
    def tqdm_mock(iterable, prefix):
        """this just returns the input and remembers it"""
        tqdm_mock.called_with = (iterable, prefix)
        return iterable
    tqdm = lf.tqdm  # remember the original tqdm to plug back in later
    lf.tqdm = tqdm_mock  # overwrite real tqdm with mock-tqdm function

    # set up: create custom class
    class SomeMonkeyClass:
        def __init__(self):
            self.somefunc_called_with = []

        def somefunc(self, *args):
            arglist = [*args]
            self.somefunc_called_with.append(arglist)
            return arglist[0] + arglist[1]

    # set up: create instance of mock class
    somemockclass = SomeMonkeyClass()

    # assert generator yields/returns the expected outcome
    assert list(gen([2, 3, 4], [4, 5, 6],
                somemockclass.somefunc, "lol", "rofl")) == [6, 8, 10]

    # assert 2 mock calls: tqdm and somefunc in SomeMonkeyClass
    assert tqdm_mock.called_with == ([2, 3, 4], "lol")
    assert somemockclass.somefunc_called_with == [
        [2, 4, "rofl"], [3, 5, "rofl"], [4, 6, "rofl"]]

    # tear down
    lf.tqdm = tqdm  # plug back in the original tqdm
    del tqdm, somemockclass, tqdm_mock, SomeMonkeyClass


def test_init():
    """test if class Search is initiated correctly"""

    # set up mock panphon class with mock edit distance
    class DistanceMonkey:
        def hamming_feature_edit_distance(): pass

    # set up mock Adrc class for get_nse
    class AdrcMonkey:
        def get_nse(self, *args): pass

    # set up mock function for semantic distance measure
    def mock_gensim_mw():
        return "sthsth"

    # set up vars 4 exped outcome, set up mock instance of DistanceMonkey class
    srsad = Series(["a", "b", "c"], name="adapted", index=[0, 1, 1])
    srsrc = Series(["a", "b", "c"], name="adapted", index=[0, 1, 1])
    dist_mockinstance = DistanceMonkey()
    # set up: mock read_data, mock panphon.Distance mock loanpy.adrc.Adrc
    with patch("loanpy.loanfinder.read_data", side_effect=[
            srsad, srsrc]) as read_data_mock:
        with patch("loanpy.loanfinder.Distance") as Distance_mock:
            Distance_mock.return_value = dist_mockinstance
            with patch("loanpy.loanfinder.Adrc") as Adrc_mock:
                Adrc_mock.return_value = AdrcMonkey

                # initiate Search() with mock parameters
                mocksearch = Search(
                    path2donordf="got.csv",
                    path2recipdf="hun.csv",
                    donorcol="adapted",
                    recipcol="reconstructed",
                    scdictlist_ad="scad.txt",
                    scdictlist_rc="scrc.txt",
                    semsim_msr=mock_gensim_mw)

                # assert initiation went properly
                assert_series_equal(mocksearch.search_in, srsad)
                assert_series_equal(mocksearch.search_for, srsrc)
                assert mocksearch.phondist == 0
                assert ismethod(mocksearch.phondist_msr)
                assert mocksearch.donpath == "got.csv"
                assert mocksearch.recpath == "hun.csv"
                assert mocksearch.doncol == "adapted"
                assert mocksearch.reccol == "reconstructed"
                assert mocksearch.semsim == 1
                assert mocksearch.semsim_msr.__name__ == "mock_gensim_mw"
                assert mocksearch.get_nse_ad == AdrcMonkey.get_nse
                assert mocksearch.get_nse_rc == AdrcMonkey.get_nse

                # double check with __dict__
                msdict = mocksearch.__dict__
                assert len(msdict) == 12
                for i in msdict:
                    if i in zip(["search_in", "search_for"], [srsad, srsrc]):
                        assert_series_equal(msdict[i], expsrs)
                    if i == "doncol":
                        assert msdict[i] == "adapted"
                    if i == "donpath":
                        assert msdict[i] == "got.csv"
                    if i == "get_nse_ad":
                        assert msdict[i
                                      ] == AdrcMonkey.get_nse
                    if i == "get_nse_rc":
                        assert msdict[i
                                      ] == AdrcMonkey.get_nse
                    if i == "phondist":
                        assert msdict[i] == 0
                    if i == "phondist_msr":
                        hmng = dist_mockinstance.hamming_feature_edit_distance
                        assert msdict[i] == hmng
                    if i == "reccol" == "reconstructed":
                        assert msdict[
                            i] == "reconstructed"
                    if i == "recpath":
                        assert msdict[i] == "hun.csv"
                    if i == "semsim":
                        assert msdict[i] == 1
                    if i == "semsim_msr":
                        assert msdict[i] == mock_gensim_mw

    # assert calls
    read_data_mock.assert_has_calls(
        [call("got.csv", "adapted"),
         call("hun.csv", "reconstructed")])
    Distance_mock.assert_called_with()
    Adrc_mock.assert_has_calls(
        [call(scdictlist="scad.txt", mode='adapt'),
         call(scdictlist="scrc.txt", mode='reconstruct')])

    # assert init runs correctly without entering parameters as well

    # set up: mock read_data, mock panphon.Distance mock loanpy.adrc.Adrc
    with patch("loanpy.loanfinder.read_data", side_effect=[
            srsad, srsrc]) as read_data_mock:
        with patch("loanpy.loanfinder.Distance") as Distance_mock:
            Distance_mock.return_value = dist_mockinstance
            with patch("loanpy.loanfinder.Adrc") as Adrc_mock:
                Adrc_mock.return_value = AdrcMonkey

                # initiate Search() without any parameters (default params)
                mocksearch = Search()

                # assert initiation went properly
                assert_series_equal(mocksearch.search_in, srsad)
                assert_series_equal(mocksearch.search_for, srsrc)
                assert mocksearch.phondist == 0
                assert ismethod(mocksearch.phondist_msr)
                assert mocksearch.donpath is None
                assert mocksearch.recpath is None
                assert mocksearch.doncol == "ad"
                assert mocksearch.reccol == "rc"
                assert mocksearch.semsim == 1
                # sic!
                assert mocksearch.semsim_msr.__name__ == "gensim_multiword"
                # not "mock_gensim_mw" even though mock func is plugged in!
                assert mocksearch.get_nse_ad == AdrcMonkey.get_nse
                assert mocksearch.get_nse_rc == AdrcMonkey.get_nse

            # double check with __dict__
            msdict = mocksearch.__dict__
            assert len(msdict) == 12
            for i in msdict:
                if i in zip(["search_in", "search_for"], [srsad, srsrc]):
                    assert_series_equal(msdict[i], expsrs)
                if i == "doncol":
                    assert msdict[i] == "ad"
                if i == "donpath":
                    assert msdict[i] is None
                if i == "get_nse_ad":
                    assert msdict[i] == AdrcMonkey.get_nse
                if i == "get_nse_rc":
                    assert msdict[i] == AdrcMonkey.get_nse
                if i == "phondist":
                    assert msdict[i] == 0
                if i == "phondist_msr":
                    assert msdict[
                        i] == dist_mockinstance.hamming_feature_edit_distance
                if i == "reccol" == "reconstructed":
                    assert msdict[i] == "rc"
                if i == "recpath":
                    assert msdict[i] is None
                if i == "semsim":
                    assert msdict[i] == 1
                # cant mock gensim. If mock plugged in not passed on in arg.
                if i == "semsim_msr":
                    assert msdict[i] == lf.gensim_multiword

    # assert calls
    read_data_mock.assert_has_calls([
        call(None, 'ad'), call(None, 'rc')])
    Distance_mock.assert_called_with()
    Adrc_mock.assert_has_calls([call(scdictlist=None, mode='adapt'),
                                call(scdictlist=None, mode='reconstruct')])

    # tear down
    del (srsad, srsrc, mocksearch, msdict, DistanceMonkey,
         AdrcMonkey, mock_gensim_mw)


def test_phonmatch_small():
    """test if closest phonemes are picked from inventory,
    while words in whcih to search have to be passed through param"""

    # set up expected outcome
    srs = Series(["a", "blub", "plub"], name="ad", index=[0, 1, 1])

    # set up mock class of loanfinder.Search
    class SearchMonkey:
        def __init__(self):
            self.dm_returns = iter([0.8, 0.4, 0.5])
            self.phondist = 0

        def dist_msr(self, *args, target): return next(self.dm_returns)

    # run test if param dropduplicates == False

    # set up mock instancec of mock class,
    # mock pandas data frame, define its return value as var
    class_phon = SearchMonkey()
    with patch("loanpy.loanfinder.DataFrame") as DataFrame_mock:
        df = DataFrame({"match": ["blub", "plub"],
                       "recipdf_idx": [99, 99]}, index=[1, 1])
        DataFrame_mock.return_value = df

        # assert actual and expected outcome are the same pandas data frames
        assert_frame_equal(Search.phonmatch_small(
            class_phon, srs, "(b|p)?lub", 99, dropduplicates=False), df)

    # assert mock data frame was called correctly:
    # with a disctionary of len 2, where keys were "match" and recipdf_idx
    # and vals were (a pandas series of 2 elements) and 99.
    assert isinstance(DataFrame_mock.call_args_list[0][0][0], dict)
    assert len(DataFrame_mock.call_args_list[0][0][0]) == 2
    assert_series_equal(DataFrame_mock.call_args_list[0][0][0]["match"],
                        Series(["blub", "plub"], name="ad", index=[1, 1]))
    assert DataFrame_mock.call_args_list[0][0][0]["recipdf_idx"] == 99

    # test with param settings dropduplicates == True
    # set up: overwrite mock instance, mock pandas data frame,
    # define its outcome as var
    class_phon = SearchMonkey()
    with patch("loanpy.loanfinder.DataFrame") as DataFrame_mock:
        df = DataFrame({"match": ["blub"], "donor_idx": [99]}, index=[1])
        DataFrame_mock.return_value = df

        # assert actual and expected outcome are identical pandas data frames
        assert_frame_equal(
            Search.phonmatch_small(class_phon, srs, "(b|p)?lub", 99), df)

    # assert mock Data frame was called correctly:
    # with a disctionary of len 2, where keys were "match" and recipdf_idx
    # and vals were (a pandas series of 2 elements) and 99
    assert len(DataFrame_mock.call_args_list[0][0][0]) == 2
    assert_series_equal(DataFrame_mock.call_args_list[0][0][0]["match"],
                        Series(["blub", "plub"], name="ad", index=[1, 1]))
    assert DataFrame_mock.call_args_list[0][0][0]["recipdf_idx"] == 99

    # test if parameters allow matches to have a higher phonetic distance than
    # 0

    # set up: overwrite mock instance,
    # plug in new maximal phonetic distance of a match
    class_phon = SearchMonkey()
    class_phon.dist = 0.5
    # set up: mock pandas dataframe, define its return value as var
    with patch("loanpy.loanfinder.DataFrame") as DataFrame_mock:
        df = DataFrame({"match": ["blub"], "donor_idx": [99]}, index=[1])
        DataFrame_mock.return_value = df

        # assert actual and expected output are identical pandas data frames
        assert_frame_equal(
            Search.phonmatch_small(class_phon, srs, "(b|p)?lub", 99), df)

    # assert mock pandas data frame was called correctly:
    # with a disctionary of len 2, where keys were "match" and recipdf_idx
    # and vals were (a pandas series of 2 elements) and 99
    assert len(DataFrame_mock.call_args_list[0][0][0]) == 2
    assert_series_equal(DataFrame_mock.call_args_list[0][0][0]["match"],
                        Series(["blub", "plub"], name="ad", index=[1, 1]))
    assert DataFrame_mock.call_args_list[0][0][0]["recipdf_idx"] == 99

    # tear down all variables
    del srs, df, class_phon, SearchMonkey


def test_phonmatch():
    """test if phonetic matching works
    if words to search in were defined through __init__"""

    # set up mock class, used throughout this test
    class SearchMonkey:
        def __init__(self):
            self.dm_returns = iter([0.8, 0.4, 0.5])
            self.phondist = 0
            self.search_in = Series(["a", "blub", "plub"],
                                    name="ad", index=[0, 1, 1])

        def dist_msr(self, *args, target): return next(self.dm_returns)

    # test with param dropduplicates == False

    # set up mock instance of mock class,
    # mock pandas dataframe, define its return value as var
    class_phon = SearchMonkey()
    with patch("loanpy.loanfinder.DataFrame") as DataFrame_mock:
        df = DataFrame({"match": ["blub", "plub"],
                       "recipdf_idx": [99, 99]}, index=[1, 1])
        DataFrame_mock.return_value = df

        # assert expected and actual results are identical pandas data frames
        assert_frame_equal(Search.phonmatch(
            class_phon, "(b|p)?lub", 99, dropduplicates=False), df)

    # assert mock pandas data frame was called correctly:
    # with a disctionary of len 2, where keys were "match" and recipdf_idx
    # and vals were (a pandas series of 2 elements) and 99
    assert isinstance(DataFrame_mock.call_args_list[0][0][0], dict)
    assert len(DataFrame_mock.call_args_list[0][0][0]) == 2
    assert_series_equal(DataFrame_mock.call_args_list[0][0][0]["match"],
                        Series(["blub", "plub"], name="ad", index=[1, 1]))
    assert DataFrame_mock.call_args_list[0][0][0]["recipdf_idx"] == 99

    # test with param dropduplicates = True

    # set up: overwrite mock instance of mock class
    # mock pandas dataframe and define its return value as var named "df"
    class_phon = SearchMonkey()
    with patch("loanpy.loanfinder.DataFrame") as DataFrame_mock:
        df = DataFrame({"match": ["blub"], "donor_idx": [99]}, index=[1])
        DataFrame_mock.return_value = df

        # assert expected and actual result are identical pandas data frames
        assert_frame_equal(Search.phonmatch(class_phon, "(b|p)?lub", 99), df)

    # assert mock pandas data frame was called correctly:
    # with a disctionary of len 2, where keys were "match" and recipdf_idx
    # and vals were (a pandas series of 2 elements) and 99
    assert len(DataFrame_mock.call_args_list[0][0][0]) == 2
    assert_series_equal(DataFrame_mock.call_args_list[0][0][0]["match"],
                        Series(["blub", "plub"], name="ad", index=[1, 1]))
    assert DataFrame_mock.call_args_list[0][0][0]["recipdf_idx"] == 99

    # test matching where phonetic distance between words can be greater than 0

    # set up: overwrite mock instance of mock class,
    # plug in new phonetic distance
    class_phon = SearchMonkey()
    class_phon.dist = 0.5
    # set up: mock pandas dadta frame, define its return value as var named
    # "df"
    with patch("loanpy.loanfinder.DataFrame") as DataFrame_mock:
        df = DataFrame({"match": ["blub"], "donor_idx": [99]}, index=[1])
        DataFrame_mock.return_value = df
        # assert expected and actual result are identical pandas data frames
        assert_frame_equal(Search.phonmatch(class_phon, "(b|p)?lub", 99), df)

    # assert mock pandas data frame was called correctly:
    # with a disctionary of len 2, where keys were "match" and recipdf_idx
    # and vals were (a pandas series of 2 elements) and 99
    assert len(DataFrame_mock.call_args_list[0][0][0]) == 2
    assert_series_equal(DataFrame_mock.call_args_list[0][0][0]["match"],
                        Series(["blub", "plub"], name="ad", index=[1, 1]))
    assert DataFrame_mock.call_args_list[0][0][0]["recipdf_idx"] == 99

    # tear down all vars used in this test
    del df, class_phon, SearchMonkey


def test_likeliestphonmatch():
    """test if likeliest phonetic match is picked correctly"""

    # sest up mock class, used throughout this test and nowhere else
    class SearchMonkey:
        def __init__(self):
            self.phonmatch_small_called_with = []
            self.get_nse_ad_called_with = []
            self.get_nse_rc_called_with = []
            self.get_nse_ad_returns = iter(
                [(8, 32, "[7, 9, 15, 2]", "['bla']"),
                 (7, 21, [9, 3, 4, 5], "['bli']")])
            self.get_nse_rc_returns = iter(
                [(10, 40, "[10, 10, 10, 10]", "['blo']"),
                 (5, 20, "[4, 4, 4, 4]", "['blu']")])
            self.doncol = "ad"

        def phonmatch_small(self, *args, dropduplicates):
            self.phonmatch_small_called_with.append([*args, dropduplicates])
            return DataFrame(
                {"match": ["blub", "plub"],
                 "recipdf_idx": [99, 99]}, index=[1, 1])

        def get_nse_ad(self, *args):
            self.get_nse_ad_called_with.append([*args])
            return next(self.get_nse_ad_returns)

        def get_nse_rc(self, *args):
            self.get_nse_rc_called_with.append([*args])
            return next(self.get_nse_rc_returns)

    # set up: define return value of mocked pandas Series class
    srs = Series(["a", "blub", "plub"], name="ph")
    # set up: define expected result
    dfexp = DataFrame({"match": ["blub"],
                       "nse_rc": [10],
                       "se_rc": [40],
                       "distr_rc": str([10] * 4),
                       "align_rc": "['blo']",
                       "nse_ad": [8],
                       "se_ad": [32],
                       "distr_ad": "[7, 9, 15, 2]",
                       "align_ad": "['bla']",
                       "nse_combined": [18]})

    # set up: initiate instance of mock class
    mocksearch = SearchMonkey()
    # set up: mock pandas Series object
    with patch("loanpy.loanfinder.Series") as Series_mock:
        Series_mock.return_value = srs

        # assert that the expected and actual results
        # are identical pandas data frames
    #    print(Search.likeliestphonmatch(
    #        mocksearch, "a, blub, plub", "(b|p)?lub", "glub", "flub"))
        assert_frame_equal(Search.likeliestphonmatch(
            mocksearch, "a, blub, plub", "(b|p)?lub", "glub", "flub"), dfexp)

    # assert mock Series class was initiated with correct args
    Series_mock.assert_called_with(["a", "blub", "plub"], name="match")

    # assert 3 calls. phonmatch_small, get_nse_rc, get_nse_ad

    # assert phonmatch_small was called once with 3 args:
    # a pandas series, a regex string and boolean "False"
    assert len(mocksearch.phonmatch_small_called_with[0]) == 3
    assert_series_equal(mocksearch.phonmatch_small_called_with[0][0], srs)
    assert mocksearch.phonmatch_small_called_with[0][1] == "(b|p)?lub"
    assert mocksearch.phonmatch_small_called_with[0][2] is False
    # assert the other 2 functions where called with the corrects args
    assert mocksearch.get_nse_rc_called_with == [
        ["flub", "blub"], ["flub", "plub"]]
    assert mocksearch.get_nse_ad_called_with == [
        ["glub", "blub"], ["glub", "plub"]]

    # tear down
    del dfexp, mocksearch, srs, SearchMonkey


def test_loans():
    """test if the main function is working:
    matching phonetically and semanitcally"""

    # set up mock function for semantic similarity calculation
    def mock_semsim(*args): pass

    # set up mock class
    class SearchMonkeyLoans:
        def __init__(self, srsin=None):
            self.search_for = srsin
            self.donpath = "got.csv"
            self.recpath = "hun.csv"
            self.semsim = 0.8
            self.postprocess_called_with = []
            self.mwr_called_with = []

        def phonmatch(self): pass

        def semsim_msr(self): return mock_semsim

        def postprocess(self, df):
            self.postprocess_called_with.append(df)
            return df.assign(postprocessed=["postprobla"])

        def merge_with_rest(self, df):
            self.mwr_called_with.append(df)
            return df.assign(mwr=["mwrbla"])

    # set up mock generator function
    # this is only to convert list objects to generator objects!
    # there is a shorter way, but forgot how
    def mockgenerator(dflist):
        for df in dflist:
            yield df  # sic

    # test first break: if no matches are found return error message

    # set up empty pandas data frame
    # in which we search for matches and find nothing
    dfempty = DataFrame({})
    # set up search_for, which is a pandas Series,
    # with words we are supposedly searching for
    srsin = Series(["word1", "word2"], name="in1")
    # set up: create instance of mock class with mock input-Series
    mocksearch = SearchMonkeyLoans(srsin)
    # set up generator object
    gen_obj = mockgenerator([dfempty, dfempty, dfempty])
    # set up: mock generator function:
    # returns a generator object of 3 empty dfs (=no matches each)
    with patch("loanpy.loanfinder.gen") as gen_mock:
        gen_mock.return_value = gen_obj
        # set up: mock concat: takes the 3 empty dfs
        # from generator and concats them to 1 empty df
        with patch("loanpy.loanfinder.concat") as concat_mock:
            concat_mock.return_value = DataFrame({})

            # assert that error is raised correctly
            # b/c no phonological matches were found
            with raises(NoPhonMatch) as nophonmatch_mock:
                Search.loans(self=mocksearch)
            assert str(nophonmatch_mock.value
                       ) == "no phonological matches found"

    # assert that the mock generator was called with the correct args:
    # the input data frame, list of values to search for,
    # and the function "phonmatch" which is a method of our mock class
    assert_series_equal(gen_mock.call_args_list[0][0][0], srsin)
    # assert search_for consisted of 2 elements with correct index
    assert_index_equal(gen_mock.call_args_list[0][0][1],
                       RangeIndex(0, 2))  # start=0, stop=2, step=1
    assert gen_mock.call_args_list[0][0][2] == mocksearch.phonmatch
    # assert concat was called with generator object of 3 empty dfs
    concat_mock.assert_called_with(gen_obj)

    # test with merge_with_rest=False, postprocess=False, write=True
    # these are just some bonus parameters,
    # we're testing the core functionality now

    # set up pandas data frame with english translations of recipient language
    dfenrec = DataFrame({"Meaning": ["edge", "ball", "x"]}, index=[99, 98, 97])
    # set up pandas data frame with english translations of donor language
    dfendon = DataFrame({"Meaning": ["sharp", "soft", "y"]}, index=[1, 2, 3])
    # set up: create new instance of mock class with mock input data frame
    # could theoretically use the previous instance, but just to be sure
    mocksearch = SearchMonkeyLoans(srsin)
    # set up mock results of phonetic and semantic matches as generator objects
    dfblub = DataFrame({"ph_match": ["kiki"], "recipdf_idx": [99]}, index=[1])
    dfplub = DataFrame({"ph_match": ["buba"], "recipdf_idx": [98]}, index=[2])

    # set up the mock phonetic matches that the mock generator will return
    # have to define it twice, can't reuse same generator object!
    phon_match_res = mockgenerator([dfblub, dfplub])
    sem_match_res = mockgenerator([0.8, 0.7])
    phon_match_res2 = mockgenerator([dfblub, dfplub])
    sem_match_res2 = mockgenerator([0.8, 0.7])

    # set up mock return value of mock pandas.concat
    dfconcat = DataFrame({"ph_match": ["kiki", "buba"],
                          "recipdf_idx": [99, 98]}, index=[1, 2])

    # set up var with expected result of return
    exp = DataFrame({"ph_match": ["kiki"], "recipdf_idx": [99],
                     "Meaning_x": ["edge"], "Meaning_y": ["sharp"],
                     "semsim_msr": [0.8]}, index=[1])
    # set up expected output after running with non-default settings
    exp2 = exp.assign(postprocessed=["postprobla"], mwr=["mwrbla"])

    # set up path to test if param "write_to" works
    path = Path(__file__).parent / "loans_test.csv"
    # set up var with expected written result (index is different from return)
    exp_read = 'ph_match,recipdf_idx,Meaning_x,Meaning_y,semsim_msr,\
postprocessed,mwr\nkiki,99,edge,sharp,0.8,postprobla,mwrbla\n'

    # set up: mock generator function,
    # first side eff. for phon. match, 2nd for semantic
    # double the side-effect for second assertion
    with patch("loanpy.loanfinder.gen", side_effect=[
            phon_match_res, sem_match_res,
            phon_match_res2, sem_match_res2]) as gen_mock:
        # set up: mock pandas concat function - input comes from mock generator
        with patch("loanpy.loanfinder.concat") as concat_mock:
            concat_mock.return_value = dfconcat
            # set up: mock pandas read_csv, double side-effect for 2nd
            # assertion
            with patch("loanpy.loanfinder.read_csv", side_effect=[
                    dfenrec, dfendon] * 2) as read_csv_mock:
                # assert expected and actual
                # return values are identical padas data frames
                # postprocess, write_to, merge_with_rest all
                # ==False (default settings)
                assert_frame_equal(Search.loans(mocksearch), exp)
                # assert written result and expected written
                # result are identical pandas data frames
                try:  # make sure no file was written
                    remove(path)
                    assert 1 == 2  # this line should never be activated
                except FileNotFoundError:  # because an error should be raised
                    pass  # because the file should not exist

                # 2nd assertion: with non-default args
                # (postprocess, write_to, merge_with_rest all == True)
                assert_frame_equal(
                    Search.loans(
                        mocksearch,
                        write_to=path,
                        postprocess=True,
                        merge_with_rest=True),
                    exp2)
                # assert file was written correctly
                with open(path, "r") as f:
                    assert f.read() == exp_read

    # assert mocked pandas.read_csv was called with the right args
    assert read_csv_mock.call_args_list == [
        call(mocksearch.recpath, encoding="utf-8", usecols=["Meaning"]),
        call(mocksearch.donpath, encoding="utf-8", usecols=["Meaning"])] * 2

    # assert mocked pandas.concat was called with right args
    # called twice, by the 2nd and 4tth output of mock generator
    concat_mock.assert_has_calls([call(phon_match_res),
                                 call(phon_match_res2)])

    # assert mock generator was called with the right args, namely:
    # the input Series with the words to search for
    # the values with which to replace the results (=the index of
    # the Series with the word we search for)
    # and the phonetic similarity measuring function
    # assert the first call
    assert_series_equal(gen_mock.call_args_list[0][0][0], srsin)
    assert_index_equal(gen_mock.call_args_list[0][0][1],
                       RangeIndex(0, 2))
    assert gen_mock.call_args_list[0][0][2] == mocksearch.phonmatch
    # assert 2nd call, namely when semantic similarity is calculated
    # the two input Series with english translations
    # function is the semantic similarity measure of the mock class
    assert_series_equal(gen_mock.call_args_list[1][0][0], Series(
        ["edge", "ball"], index=[1, 2], name="Meaning_x"))
    assert_series_equal(gen_mock.call_args_list[1][0][1], Series(
        ["sharp", "soft"], index=[1, 2], name="Meaning_y"))
    assert gen_mock.call_args_list[1][0][2] == mocksearch.semsim_msr
    # assert third call of mock generator
    assert_series_equal(gen_mock.call_args_list[2][0][0], srsin)
    # assert fourth call of mock generator
    assert_series_equal(gen_mock.call_args_list[3][0][0], Series(
        ["edge", "ball"], index=[1, 2], name="Meaning_x"))
    assert_series_equal(gen_mock.call_args_list[3][0][1], Series(
        ["sharp", "soft"], index=[1, 2], name="Meaning_y"))

    # assert mocked postprocessing was called correctly
    assert_frame_equal(mocksearch.postprocess_called_with[0], exp)
    # assert merge with rest was called correctly
    assert_frame_equal(mocksearch.mwr_called_with[0],
                       exp.assign(postprocessed="postprobla"))

    # tear down
    remove(path)
    del (SearchMonkeyLoans,
         srsin,
         dfenrec,
         dfendon,
         mocksearch,
         dfblub,
         dfplub,
         path,
         exp,
         dfempty,
         mockgenerator)


def test_postprocess():
    """test if phon. matches are replaced with their likeliest version
    and nse+other stat. data is displayed correctly"""

    # set up: define input data frame
    dfin = DataFrame({"match": ["kiki"], "recipdf_idx": [99], "Meaning_x": [
                     "edge"], "Meaning_y": ["sharp"], "semsim_msr": [0.8]},
                     index=[1])

    # set up: define expected output data frame
    dfexp = DataFrame({"recipdf_idx": [99],
                       "Meaning_x": ["edge"],
                       "Meaning_y": ["sharp"],
                       "semsim_msr": [0.8],
                       "match": ["hihi"],
                       "nse_rc": [10],
                       "se_rc": [40],
                       "lst_rc": [[10] * 4],
                       "nse_ad": [8],
                       "se_ad": [32],
                       "lst_ad": [[7, 9, 15, 2]],
                       "nse_combined": [18],
                       "e_rc": [[[123], [456],
                                 [789], [908]]],
                       "e_ad": [[[123], [456], [789], [908]]]},
                      index=[1])

    # set up: define return value of mocked pandas concat function
    # autopep8 is responsible for this kind of formatting
    dfcc = DataFrame({"recipdf_idx": [99],
                      "Meaning_x": ["edge"],
                      "Meaning_y": ["sharp"],
                      "semsim_msr": [0.8],
                      "match": ["hihi"],
                      "nse_rc": [10],
                      "se_rc": [40],
                      "lst_rc": [[10] * 4],
                      "nse_ad": [8],
                      "se_ad": [32],
                      "lst_ad": [[7, 9, 15, 2]],
                      "nse_combined": [18],
                      "e_rc": [[[123], [456], [789], [908]]],
                      "e_ad": [[[123], [456], [789], [908]]],
                      "Segments_x": ["k i k i"],
                      "rc": ["(k|h)(i)(k|h)(i)"],
                      "Segments_y": ["k i k i"],
                      "ad": ["kiki, hihi"]},
                     index=[1])

    # set up expected call of mock concat
    # (not really concating anything to keep simple)
    dfcc_calledwith = DataFrame({"recipdf_idx": [99],
                                 "Meaning_x": ["edge"],
                                 "Meaning_y": ["sharp"],
                                 "semsim_msr": [0.8],
                                 "Segments_x": ["kiki"],
                                 "rc": ["(k|h)(i)(k|h)(i)"],
                                 "Segments_y": ["kiki"],
                                 "ad": ["kiki, hihi"]},
                                index=[1])

    # set up: mock a search_for data frame (=reconstructed)
    dfrc = DataFrame({"Segments": ["k i k i", "b u b a"], "rc": [
                     "(k|h)(i)(k|h)(i)", "(b|p)(u)(b|p)(a)"]}, index=[99, 100])
    # set up: mock a search_in data frame (=adapted)
    dfad = DataFrame({"Segments": ["k i k i", "b u b a"],
                      "ad": ["kiki, hihi", "buba, pupa"]}, index=[1, 2])

    # set up: mock the result of likeliestphonmatch
    out_likeliest = DataFrame({"match": ["hihi"],
                               "nse_rc": [10],
                               "se_rc": [40],
                               "lst_rc": [[10] * 4],
                               "nse_ad": [8],
                               "se_ad": [32],
                               "lst_ad": [[7, 9, 15, 2]],
                               "nse_combined": [18],
                               "e_rc": [[[123], [456], [789], [908]]],
                               "e_ad": [[[123], [456], [789], [908]]]})

    # set up: create a mock class, only used in this test
    class MonkeyPostprocess:
        def __init__(self):
            self.likeliestphonmatch_called_with = []
            self.donpath = "got.csv"
            self.recpath = "hun.csv"
            self.doncol = "ad"
            self.reccol = "rc"

        def likeliestphonmatch(self, *args):
            self.likeliestphonmatch_called_with.append([*args])
            return out_likeliest

    # initiate instance of mock class
    mocksearch = MonkeyPostprocess()

    # set up: mock pandas read_csv
    with patch("loanpy.loanfinder.read_csv", side_effect=[
            dfrc, dfad]) as read_csv_mock:
        # set up: mock pandas concat
        with patch("loanpy.loanfinder.concat", side_effect=[
                out_likeliest, dfcc]) as concat_mock:

            # assert that actual and expected outcome is same
            # note that self=mocksearch!
            assert_frame_equal(Search.postprocess(mocksearch, dfin), dfexp)

    # assert 3 calls: pandas.read_csv, mocksearch.likeliestphonmatch
    # and concat_mock
    read_csv_mock.assert_has_calls([
        call("hun.csv", encoding="utf-8", usecols=["Segments", "rc"]),
        call("got.csv", encoding="utf-8", usecols=["Segments", "ad"])])
    assert mocksearch.likeliestphonmatch_called_with == [
        ["kiki, hihi", "(k|h)(i)(k|h)(i)", "kiki", "kiki"]]
    # assert first concat calls:
    # vertical concatenation of likeliestphonmatch' outcome
    assert_frame_equal(
        concat_mock.call_args_list[0][0][0][0],
        out_likeliest)
    # first [0]=first call, 2nd [0] convert call to tuple
    # 3rd [0] pick first ele of tup (a list in this case),
    # 4th [0]: pick the first element of the list (dfmatches)
    assert_frame_equal(
        concat_mock.call_args_list[1][0][0][0], dfcc_calledwith)
    # üick the second element of the list (dfnewcols)
    assert_frame_equal(
        concat_mock.call_args_list[1][0][0][1],
        out_likeliest)

    # tear down
    del (mocksearch, MonkeyPostprocess, out_likeliest, dfad, dfrc,
         dfcc_calledwith, dfcc, dfexp, dfin)


def test_merge_with_rest():
    """test if output is correctly merged with redundant info from input"""

    # set up mock function for semantic similarity
    def mocksemsim(): pass

    # set up mock class, only used in this test
    class SearchMonkey:
        donpath = "got.csv"
        recpath = "hun.csv"
        semsim_msr = mocksemsim

    # initiated mock class
    mocksearch = SearchMonkey()

    # mock input data frame
    dfin = DataFrame({"a": ["pi", "pa", "po"],
                      "b": ["bi", "ba", "bo"],
                      "Meaning_x": ["mi", "ma", "mo"],
                      "Meaning_y": ["ni", "na", "no"],
                      "recipdf_idx": [7, 7, 8],
                      "mocksemsim": [98, 99, 100]},
                     index=[4, 5, 5])

    # set up return values of mocked pandas.read_csv
    dfmockread1 = DataFrame(
        {"c": ["ci", "ca", "co", "cu"]}, index=[4, 5, 6, 7])
    dfmockread2 = DataFrame(
        {"d": ["di", "da", "do", "du"]}, index=[7, 8, 9, 10])

    # set expected outcome, a pandas DataFrame object
    dfexp = DataFrame({"a": ["po", "pa", "pi"],
                       "b": ["bo", "ba", "bi"],
                       "recipdf_idx": [8, 7, 7],
                       "mocksemsim": [100, 99, 98],
                       "c": ["ca", "ca", "ci"],
                       "d": ["da", "di", "di"]},
                      index=[5, 5, 4])

    # mock pandas read_csv
    with patch("loanpy.loanfinder.read_csv", side_effect=[
            dfmockread1, dfmockread2]) as read_csv_mock:

        # assert that expected and actual return values
        # are identical pandas data frames
        assert_frame_equal(Search.merge_with_rest(mocksearch, dfin), dfexp)

    # assert that pandas read_csv was called with the correct values
    read_csv_mock.assert_has_calls([call("got.csv", encoding="utf-8"),
                                    call("hun.csv", encoding="utf-8")])

    # tear down
    del (dfexp, dfmockread2, dfmockread1, dfin,
         mocksearch, SearchMonkey, mocksemsim)
