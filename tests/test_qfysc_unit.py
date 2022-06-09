"""unit test for loanpy.qfysc.py (2.0 BETA) for pytest 7.1.1"""

from ast import literal_eval
from os import remove
from pathlib import Path
from unittest.mock import patch, call

from loanpy import qfysc as qs
from loanpy.qfysc import (
    Qfy,
    WrongModeError,
    read_mode,
    read_connector,
    # read_nsedict,
    read_scdictbase)

from pytest import raises
from pandas import DataFrame
from pandas.testing import assert_frame_equal

# used throughout the module


class QfyMonkey:
    pass

# used in multiple places


class PairwiseMonkey:
    def __init__(self, *args):
        self.align_called_with = []

    def align(self, *args):
        self.align_called_with.append([*args])

    def __str__(self): return 'k\ta\tl\ta\nh\ta\tl\t-\n13.7'

# used in multiple places


class QfyMonkeyGetSoundCorresp:
    def __init__(
            self,
            mode,
            connector,
            alignreturns1,
            alignreturns2,
            vfb=None):
        self.df1 = alignreturns1
        self.df2 = alignreturns2
        self.align_returns = iter([self.df1, self.df2])
        self.align_called_with = []
        self.word2phonotactics_called_with = []
        self.rank_closest_phonotactics_called_with = []
        self.dfety = DataFrame({"Target_Form": ["kiki", "buba"],
                                "Source_Form": ["hehe", "pupa"],
                                "Cognacy": [12, 13]})
        self.left = "Target_Form"
        self.right = "Source_Form"
        self.mode = mode
        self.connector = connector
        self.scdictbase = {"k": ["h", "c"], "i": ["i", "e"], "b": ["v"],
                           "u": ["o", "u"], "a": ["y", "ü"]}
        self.vfb = vfb
        self.phon2cv = {
            "k": "C",
            "i": "V",
            "b": "C",
            "u": "V",
            "a": "V",
            "e": "V"}
        self.vow2fb = {"i": "F", "e": "F", "u": "B", "a": "B"}

    def align(self, left, right):
        self.align_called_with.append((left, right))
        return next(self.align_returns)

    def word2phonotactics(self, word):
        self.word2phonotactics_called_with.append(word)
        return "CVCV"

    def rank_closest_phonotactics(self, struc):
        self.rank_closest_phonotactics_called_with.append(struc)
        return "CVC, CVCCC"

    def get_phonotactics_corresp(self, *args):
        return [{"d1": 0}, {"d2": 0}, {"d3": 0}]


def test_read_mode():
    """test if mode is read correctly"""

    # no setup or teardown needed for these assertions

    # assert exception and exception message
    with raises(WrongModeError) as wrongmodeerror_mock:
        read_mode(mode="neitheradaptnorreconstruct")
    assert str(
        wrongmodeerror_mock.value
    ) == "parameter <mode> must be 'adapt' or 'reconstruct'"

    assert read_mode("adapt") == "adapt"
    assert read_mode("reconstruct") == "reconstruct"
    assert read_mode(None) == "adapt"
    assert read_mode("") == "adapt"


def test_read_connector():
    """test if connector is read correctly"""
    # no setup or teardown needed for these assertions
    assert read_connector(connector=None, mode="adapt") == "<"
    assert read_connector(connector=None, mode=None) == "<"
    assert read_connector(connector=None, mode="reconstruct") == "<*"
    assert read_connector(
        connector=(" from ", " from *"),
        mode="reconstruct") == " from *"


def test_read_scdictbase():
    """test if scdictbase is generated correctly from ipa_all.csv"""

    # no setup needed for this assertion
    assert read_scdictbase(None) == {}

    # setup
    base = {"a": ["e", "o"], "b": ["p", "v"]}
    path = Path(__file__).parent / "test_read_scdictbase.txt"
    with open(path, "w") as f:
        f.write(str(base))

    with patch("loanpy.qfysc.literal_eval") as literal_eval_mock:
        literal_eval_mock.return_value = base

        # assert
        assert read_scdictbase(base) == base
        assert read_scdictbase(path) == base

    # assert call
    literal_eval_mock.assert_called_with(str(base))

    # tear down
    remove(path)
    del base, path


def test_init():
    """test if class Qfy is initiated correctly"""

    # set up
    with patch("loanpy.helpers.Etym.__init__") as super_method_mock:
        with patch("loanpy.qfysc.read_mode") as read_mode_mock:
            read_mode_mock.return_value = "adapt"
            with patch("loanpy.qfysc.read_connector") as read_connector_mock:
                read_connector_mock.return_value = "<"
                # with patch("loanpy.qfysc.read_nsedict") as read_nsedict_mock:
                #    read_nsedict_mock.return_value = {}
                with patch("loanpy.qfysc.read_scdictbase"
                           ) as read_scdictbase_mock:
                    read_scdictbase_mock.return_value = {}
                    mockqfy = Qfy()

                    # assert
                    assert mockqfy.mode == "adapt"
                    assert mockqfy.connector == "<"
                    assert mockqfy.scdictbase == {}
                    assert mockqfy.vfb is None

                    # double check with __dict__
                    assert len(mockqfy.__dict__) == 4
                    assert mockqfy.__dict__ == {
                        'connector': '<',
                        'mode': 'adapt',
                        'scdictbase': {},
                        'vfb': None}

                    # assert calls
                    super_method_mock.assert_called_with(
                        forms_csv=None, source_language=None,
                        target_language=None,
                        most_frequent_phonotactics=9999999,
                        phonotactic_inventory=None)
                    read_mode_mock.assert_called_with("adapt")
                    read_connector_mock.assert_called_with(None, "adapt")
                    # read_nsedict_mock.assert_called_with(None)
                    read_scdictbase_mock.assert_called_with(None)

    # tear down
    del mockqfy


def test_align():
    """test if align assigns the correct alignment-function to 2 strings"""

    # set up mock class
    class QfyMonkeyAlign:
        def __init__(self):
            self.align_lingpy_called_with = []
            self.align_clusterwise_called_with = []

        def align_lingpy(self, *args):
            self.align_lingpy_called_with.append([*args])
            return "lingpyaligned"

        def align_clusterwise(self, *args):
            self.align_clusterwise_called_with.append([*args])
            return "clusterwisealigned"

    # initiate mock class, plug in mode
    mockqfy = QfyMonkeyAlign()
    mockqfy.mode = "adapt"
    # assert if lingpy-alignment is assigned correctly if mode=="adapt"
    assert Qfy.align(
        self=mockqfy,
        left="leftstr",
        right="rightstr") == "lingpyaligned"
    # assert call
    assert mockqfy.align_lingpy_called_with == [['leftstr', 'rightstr']]

    # set up mock class, plug in mode
    mockqfy = QfyMonkeyAlign()
    mockqfy.mode = "reconstruct"
    # assert
    assert Qfy.align(
        self=mockqfy,
        left="leftstr",
        right="rightstr") == "clusterwisealigned"
    # assert call
    assert mockqfy.align_clusterwise_called_with == [["leftstr", "rightstr"]]

    # tear down
    del mockqfy


def test_align_lingpy():
    """test if lingpy's pairwise alignment function is called correctly"""

    # set up instance of basic mock class, plug in phon2cv,
    # mock lingpy.Pairwise, mock pandas.DataFrame
    mockqfy = QfyMonkey()
    mockpairwise = PairwiseMonkey()
    mockqfy.phon2cv = {"h": "C", "a": "V", "l": "C"}
    with patch("loanpy.qfysc.Pairwise") as Pairwise_mock:
        Pairwise_mock.return_value = mockpairwise
        with patch("loanpy.qfysc.DataFrame") as DataFrame_Monkey:
            exp = {"keys": ["h", "a", "l", "V"], "vals": ["k", "a", "l", "a"]}
            DataFrame_Monkey.return_value = DataFrame(exp)

            # assert
            assert_frame_equal(
                Qfy.align_lingpy(
                    self=mockqfy,
                    left="kala",
                    right="hal"),
                DataFrame(exp))

    # assert calls
    Pairwise_mock.assert_has_calls([call(
        seqs='kala', seqB='hal', merge_vowels=False)])
    # Pairwise always initiated by 3 args
    assert DataFrame_Monkey.call_args_list == [call(exp)]
    assert mockpairwise.align_called_with == [[]]

    # tear down
    del mockqfy, exp


def test_align_clusterwise():
    """test if loanpy's own clusterwise-alignment works"""

    # set up basic mock class, plug in phon2cv, create expected output var
    mockqfy = QfyMonkey()
    mockqfy.phon2cv = {
        "ɟ": "C", "ɒ": "V", "l": "C", "o": "V", "ɡ": "C", "j": "C"}
    exp = DataFrame({"keys": ['#-', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                    "vals": ['-', 'j', 'ɑ', 'lk', 'ɑ', '-']})

    # assert
    assert_frame_equal(
        Qfy.align_clusterwise(self=mockqfy, left="ɟɒloɡ", right="jɑlkɑ"), exp)

    # tear down
    del mockqfy, exp


def test_get_sound_corresp():
    """test if sound correspondences are extracted correctly"""

    # set up: the expected outcome of assert while mode=="adapt"
    exp = [{"a": ["a", "y", "ü"], "b": ["p", "v"], "i": ["e", "i"],
            "k": ["h", "c"], "u": ["u", "o"]},
           {'a<a': 1, 'e<i': 2, 'h<k': 2, 'p<b': 2, 'u<u': 1},
           {'a<a': [13], 'e<i': [12], 'h<k': [12], 'p<b': [13], 'u<u': [13]},
           {'d1': 0}, {'d2': 0}, {'d3': 0}]

    # set up: the expected outcome of assert while mode=="reconstruct"
    exp2 = [{'#-': ['-'], '#b': ['p'], '#k': ['h'], 'a#': ['a', 'ə', 'ʌ'],
             'b': ['p'], 'i': ['e', 'ə', 'œ'], 'i#': ['e', 'ə', 'œ'],
             'k': ['h'], 'u': ['u', 'ə', 'ʌ']},
            {'#-<*-': 2, '#b<*p': 1, '#k<*h': 1, 'a#<*a': 1, 'b<*p': 1,
             'i#<*e': 1, 'i<*e': 1, 'k<*h': 1, 'u<*u': 1},
            {'#-<*-': [12, 13], '#b<*p': [13], '#k<*h': [12],
             'a#<*a': [13], 'b<*p': [13], 'i#<*e': [12], 'i<*e': [12],
             'k<*h': [12], 'u<*u': [13]},
            {}, {}, {}]

    # set up: create instance 1 of mock class
    mockqfy = QfyMonkeyGetSoundCorresp(
        mode="adapt", connector="<", alignreturns1=DataFrame(
            {
                "keys": [
                    "k", "i", "k", "i"], "vals": [
                    "h", "e", "h", "e"]}), alignreturns2=DataFrame(
                        {
                            "keys": [
                                "b", "u", "b", "a"], "vals": [
                                    "p", "u", "p", "a"]}))

    # set up: create instance 2 of mock class
    mockqfy2 = QfyMonkeyGetSoundCorresp(  # necessary bc of iter()
        mode="reconstruct", connector="<*", vfb="əœʌ",
        alignreturns1=DataFrame(
            {"keys": ["#-", "#k", "i", "k", "i#"],
             "vals": ["-", "h", "e", "h", "e"]}),
        alignreturns2=DataFrame(
            {"keys": ["#-", "#b", "u", "b", "a#"],
             "vals": ["-", "p", "u", "p", "a"]}))

    # set up the side_effects of pandas.concat
    dfconcat = DataFrame({"keys": list("kikibuba"), "vals": list("hehepupa")})
    dfconcat2 = DataFrame(
        {"keys": ["#-", "#k", "i", "k", "i#", "#-", "#b", "u", "b", "a#"],
         "vals": ["-", "h", "e", "h", "e", "-", "p", "u", "p", "a"]})

    # set up path for param write_to
    path2test_get_sound_corresp = Path(
        __file__).parent / "test_get_sound_corresp.txt"

    # mock pandas.concat
    with patch("loanpy.qfysc.concat", side_effect=[
            dfconcat, dfconcat2]) as concat_mock:
        # groupby too difficult to mock
        # assert while mode=="adapt"
        assert Qfy.get_sound_corresp(
            self=mockqfy, write_to=None) == exp
        try:  # assert that no file was being written
            remove(Path(__file__).parent / "soundchanges.txt")
            assert 1 == 2  # this asserts that the except part was being run
        except FileNotFoundError:  # i.e. the file was correctly not written
            pass
        # assert while mode=="reconstruct"
        assert Qfy.get_sound_corresp(
            self=mockqfy2,
            write_to=path2test_get_sound_corresp) == exp2
        # assert that file was written
        with open(path2test_get_sound_corresp, "r") as f:
            assert literal_eval(f.read()) == exp2

        # assert calls from assert while mode == "adapt"
        assert mockqfy.align_called_with == [
            ("kiki", "hehe"), ("buba", "pupa")]
        assert mockqfy.word2phonotactics_called_with == []  # not called
        assert mockqfy.rank_closest_phonotactics_called_with == []  # no called
        for act, exp in zip(
            concat_mock.call_args_list[0][0][0], [
                mockqfy.df1, mockqfy.df2]):
            # first [0] picks the only call from list of calls
            # second [0] converts call object to tuple
            # third [0] picks the first element of the tuple (2nd is empty),
            # which is a list of two dataframes
            assert_frame_equal(act, exp)

    # assert calls of assert while mode == "reconstruct"
    assert mockqfy2.align_called_with == [
        ("kiki", "hehe"), ("buba", "pupa")]
    assert mockqfy2.word2phonotactics_called_with == []  # not called
    assert mockqfy2.rank_closest_phonotactics_called_with == []  # not called
    for act, exp in zip(
        concat_mock.call_args_list[1][0][0], [
            mockqfy2.df1, mockqfy2.df2]):
        # first [0] picks the only call from list of calls
        # second [0] converts call object to tuple
        # third [0] picks the first element of the tuple (2nd is empty),
        # which is a list of two dataframes
        assert_frame_equal(act, exp)

    # tear down
    remove(path2test_get_sound_corresp)
    del mockqfy, dfconcat, exp, path2test_get_sound_corresp


def test_get_phonotactics_corresp():
    """test if phonotactic correspondences are extracted correctly from data"""

    # set up instance of mock class, plug in attributes
    # def expected outcome var, vars for side-effects of mock-functions,
    # vars for expected calls
    # mock pandasDataFrame
    mockqfy = QfyMonkeyGetSoundCorresp(
        mode="adapt",
        connector="<",
        alignreturns1=None,
        alignreturns2=None)

    mockqfy.dfety = DataFrame({"Target_Form": ["kiki", "buba"],
                               "Source_Form": ["hehe", "pupa"],
                               "Cognacy": [12, 13]})
    mockqfy.left = "Target_Form"
    mockqfy.right = "Source_Form"
    mockqfy.connector = "<"
    exp = [{"CVCV": ["CVCV", "CVC", "CVCCC"]},
           {"CVCV<CVCV": 2},
           {"CVCV<CVCV": [12, 13]}]
    exp_call1 = list(zip(["CVCV", "CVCV"], ["CVCV", "CVCV"], [12, 13]))
    exp_call2 = {"columns": ["keys", "vals", "wordchange"]}

    path2test_get_phonotactics_corresp = Path(
        __file__).parent / "phonotctchange.txt"

    with patch("loanpy.qfysc.DataFrame") as DataFrame_mock:
        DataFrame_mock.return_value = DataFrame(
            {"keys": ["CVCV"] * 2, "vals": ["CVCV"] * 2,
             "wordchange": [12, 13]})

        # assert
        assert Qfy.get_phonotactics_corresp(
            self=mockqfy, write_to=path2test_get_phonotactics_corresp) == exp
        # assert file was written
        with open(path2test_get_phonotactics_corresp, "r") as f:
            assert literal_eval(f.read()) == exp

    # assert calls
    assert list(DataFrame_mock.call_args_list[0][0][0]) == exp_call1
    assert DataFrame_mock.call_args_list[0][1] == exp_call2
    assert mockqfy.word2phonotactics_called_with == [
        'hehe', 'pupa', 'kiki', 'buba']
    assert mockqfy.rank_closest_phonotactics_called_with == ["CVCV"]

    # tear down
    remove(path2test_get_phonotactics_corresp)
    del mockqfy, exp, path2test_get_phonotactics_corresp
