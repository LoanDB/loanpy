import os
import unittest
from unittest.mock import patch

import pandas as pd
from pandas._testing import assert_frame_equal

from loanpy import reconstructor as rc

os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


class Test1(unittest.TestCase):

    def test_launch(self):

        soundchangedict_test = {'#0': ["j", "m"], '#a': ["ɑ"], '0#': ["ɑ"]}
        with open("soundchangedict_test.txt", "w", encoding="utf-8") as data:
            data.write(str(soundchangedict_test))
        dfetymology_test = pd.DataFrame({"New": ["mɛʃɛ", "aːɟ", "ɒl"],
                                         "Old": ["ɑt͡ʃʲɑ", "ɑðʲͽ", "ɑlɑ"],
                                         "Lan": ["FP", "FU", "U"]})
        dfetymology_test.to_csv("dfetymology_test.csv", encoding="utf-8",
                                index=False)
        substi_test = pd.DataFrame({"L2_phons": ["a", "b", "c", "d"],
                                    "L1_substi": ["o", "p", "t", "t"]})
        substi_test.to_csv("substi_test.csv", encoding="utf-8", index=False)
        nsedict_test = {'#0<*0': 433, '#0<*j': 6}
        with open("nsedict_test.txt", "w", encoding="utf-8") as data:
            data.write(str(nsedict_test))

        with patch("loanpy.reconstructor.word2struc",
                   side_effect=["VCV", "VCV", "VCV"]):
            rc.launch(soundchangedict="soundchangedict_test.txt",
                      dfetymology="dfetymology_test.csv",
                      timelayer="",
                      se_or_edict="nsedict_test.txt")

        self.assertDictEqual(rc.scdict, soundchangedict_test)
        self.assertEqual(rc.allowedphonotactics, {"VCV"})
        self.assertEqual(rc.nsedict, nsedict_test)

        os.remove("dfetymology_test.csv")
        os.remove("substi_test.csv")
        os.remove("soundchangedict_test.txt")
        os.remove("nsedict_test.txt")

    def test_getsoundchanges(self):
        with patch("loanpy.reconstructor.ipa2clusters",
                   side_effect=[["ɟ", "ɒ", "l", "o", "ɡ"],
                                ["j", "ɑ", "lk", "ɑ"]]):
            expected_df = pd.DataFrame({"reflex":
                                        ['#0', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                        "root":
                                        ['0', 'j', 'ɑ', 'lk', 'ɑ', '0']})
            assert_frame_equal(rc.getsoundchanges("ɟɒloɡ", "jɑlkɑ"),
                               expected_df, check_dtype=False)
        with patch("loanpy.reconstructor.ipa2clusters",
                   side_effect=[['m', 'ɛ', 'ʃ', 'ɛ'], ['ɑ', 't͡ʃʲ', 'ɑ']]):
            expected_df = pd.DataFrame({"reflex": ['#m', 'ɛ', 'ʃ', 'ɛ#', '0#'],
                                        "root": ['0', 'ɑ', 't͡ʃʲ', 'ɑ', '0']})
            assert_frame_equal(rc.getsoundchanges("mɛʃɛ", "ɑt͡ʃʲɑ"),
                               expected_df, check_dtype=False)
        with patch("loanpy.reconstructor.ipa2clusters",
                   side_effect=[["ɒ", "l"], ["ɑ", "l", "ɑ"]]):
            expected_df = pd.DataFrame({"reflex": ["#0", "#ɒ", "l#", "0#"],
                                        "root": ['0', 'ɑ', 'l', 'ɑ']})
            assert_frame_equal(rc.getsoundchanges("ɒl", "ɑlɑ"),
                               expected_df, check_dtype=False)

    def test_dfetymology2dict(self):
        data = {"New": ["mɛʃɛ", "aːɟ", "ɒl"], "Old": ["ɑt͡ʃʲɑ", "ɑðʲͽ", "ɑlɑ"]}
        pd.DataFrame(data).to_csv("dfural_test.csv", encoding="utf-8",
                                  index=False)
        out = rc.dfetymology2dict(dfetymology="dfural_test.csv",
                                  timelayer="",
                                  name_soundchangedict="scdict_test",
                                  name_sumofexamplesdict="sedict_test",
                                  name_listofexamplesdict="edict_test")
        expectedscdict = {'#0': ['0'], '#aː': ['ɑ'], '#m': ['0'], '#ɒ': ['ɑ'],
                          '0#': ['0', 'ͽ', 'ɑ'], 'l#': ['l'], 'ɛ': ['ɑ'],
                          'ɛ#': ['ɑ'], 'ɟ#': ['ðʲ'], 'ʃ': ['t͡ʃʲ']}
        expectedsedict = {'#0<*0': 2, '#aː<*ɑ': 1, '#m<*0': 1, '#ɒ<*ɑ': 1,
                          '0#<*0': 1, '0#<*ɑ': 1, '0#<*ͽ': 1, 'l#<*l': 1,
                          'ɛ#<*ɑ': 1, 'ɛ<*ɑ': 1, 'ɟ#<*ðʲ': 1, 'ʃ<*t͡ʃʲ': 1}
        expectededict = {'#0<*0': ['aːɟ<*ɑðʲͽ', 'ɒl<*ɑlɑ'],
                         '#aː<*ɑ': ['aːɟ<*ɑðʲͽ'],
                         '#m<*0': ['mɛʃɛ<*ɑt͡ʃʲɑ'], '#ɒ<*ɑ': ['ɒl<*ɑlɑ'],
                         '0#<*0': ['mɛʃɛ<*ɑt͡ʃʲɑ'], '0#<*ɑ': ['ɒl<*ɑlɑ'],
                         '0#<*ͽ': ['aːɟ<*ɑðʲͽ'], 'l#<*l': ['ɒl<*ɑlɑ'],
                         'ɛ#<*ɑ': ['mɛʃɛ<*ɑt͡ʃʲɑ'], 'ɛ<*ɑ': ['mɛʃɛ<*ɑt͡ʃʲɑ'],
                         'ɟ#<*ðʲ': ['aːɟ<*ɑðʲͽ'], 'ʃ<*t͡ʃʲ': ['mɛʃɛ<*ɑt͡ʃʲɑ']}

        df1 = pd.DataFrame({"reflex": ['#m', 'ɛ', 'ʃ', 'ɛ#', '0#'],
                            "root": ['0', 'ɑ', 't͡ʃʲ', 'ɑ', '0']})
        df2 = pd.DataFrame({"reflex": ['#0', '#aː', 'ɟ#', '0#'],
                            "root": ['0', 'ɑ', 'ðʲ', 'ͽ']})
        df3 = pd.DataFrame({"reflex": ["#0", "#ɒ", "l#", "0#"],
                            "root": ['0', 'ɑ', 'l', 'ɑ']})

        with patch("loanpy.reconstructor.getsoundchanges",
                   side_effect=[df1, df2, df3]*2):
            self.assertEqual(out[0], expectedscdict)
            self.assertEqual(out[1], expectedsedict)
            self.assertEqual(out[2], expectededict)

        os.remove("edict_test.txt")
        os.remove("scdict_test.txt")
        os.remove("sedict_test.txt")
        os.remove("dfural_test.csv")

    def test_getnse(self):
        with patch("loanpy.reconstructor.getsoundchanges") as getsndchg_mock:
            getsndchg_mock.return_value = pd.DataFrame({"reflex": ["#l", "e#"],
                                                        "root": ["l", "e"]})
            rc.nsedict = {"#l<*l": ["ló<*ló"], "e#<*e": ["teve<*teve"]}
            out1 = rc.getnse("le", "le", examples=True, normalise=False)
            rc.nsedict = {"#l<*l": 10, "e#<*e": 5}
            out2 = rc.getnse("le", "le", examples=False, normalise=False)
            out3 = rc.getnse("le", "le", examples=False, normalise=True)
            out4 = rc.getnse("le", "le", examples=True, normalise=True)
            self.assertEqual(out1, [["ló<*ló"], ["teve<*teve"]])
            self.assertEqual(out2, 15)
            self.assertEqual(out3, 7.5)
            self.assertEqual(out4, [10, 5])

    def test_reconstruct(self):
        rc.scdict = {"#l": ["l", "r"], "e#": ["e", "a"], "#0": ["0"],
                     "0#": ["0", "o"]}
        rc.nsedict = {"#l<*l": 5, "#l<*r": 3, "e#<*e": 3, "e#<*a": 1,
                      "#0<*0": 10, "0#<*0": 2, "0#<*o": 5}
        rc.suffixes = ["ves#"]

        with patch("loanpy.reconstructor.ipa2clusters") as ipa2clusters_mock:
            ipa2clusters_mock.return_value = ["l", "e"]
            with patch("loanpy.reconstructor.list2regex",
                       side_effect=["", "(l)", "(e)", "", "", "(l|r)",
                                    "(e)", "(o)?"]):
                out1 = rc.reconstruct('le', howmany=1, struc=False,
                                      vowelharmony=False)
                out2 = rc.reconstruct('le', howmany=3, struc=False,
                                      vowelharmony=False,
                                      only_documented_clusters=False)
                rc.allowedphonotactics = ["CCV"]
                out3 = rc.reconstruct('le', howmany=5, struc=True,
                                      vowelharmony=False,
                                      only_documented_clusters=False)

                self.assertEqual(out1, '^(l)(e)$')
                self.assertEqual(out2, '^(l|r)(e)(o)?$')
                self.assertEqual(out3, 'wrong phonotactics')

            with patch("loanpy.reconstructor.getnse",
                       side_effect=[5, 4.5, 4.5, 4]*3):
                with patch("loanpy.reconstructor.word2struc",
                           side_effect=['CV', 'CVV', 'CV', 'CVV',
                                        'CV', 'CVV', 'CV', 'CVV']):
                    rc.allowedphonotactics = ["CV"]
                    out4 = rc.reconstruct('le', howmany=5, struc=True,
                                          vowelharmony=False, sort_by_nse=True,
                                          only_documented_clusters=False)
                    self.assertEqual(out4, '^le$|^re$|^la$|^ra$')

                with patch("loanpy.reconstructor.harmony",
                           side_effect=[True, True, False, False]):
                    out5 = rc.reconstruct('le', howmany=5, struc=True,
                                          vowelharmony=True, sort_by_nse=True,
                                          only_documented_clusters=False)
                    self.assertEqual(out5, '^le$|^la$')

                with patch("loanpy.reconstructor.harmony",
                           side_effect=[False, False, False, False]):
                    out6 = rc.reconstruct('le', howmany=5, struc=True,
                                          vowelharmony=True, sort_by_nse=True,
                                          only_documented_clusters=False)
                    self.assertEqual(out6, 'wrong vowel harmony')

        with patch("loanpy.reconstructor.ipa2clusters") as ipa2clusters_mock:
            ipa2clusters_mock.return_value = ["m", "ɒ", "jd"]
            rc.scdict = {"#m": ["m"], "ɒ": ["ɑ"], "jd#": ["ðʲt"],
                         "#0": ["0"], "0#": ["ͽ"]}
            rc.nsedict = {"#m<*m": 5, "ɒ<*ɑ": 3, "jd#<*ðʲt": 3,
                          "#0<*0": 10, "0#<*ͽ": 5}
            out8 = rc.reconstruct('mɒjd', howmany=1, struc=False,
                                  only_documented_clusters=True,
                                  vowelharmony=False, sort_by_nse=True)
            self.assertEqual(out8, '^(m)(ɑ)(ðʲt)(ͽ)$')

        with patch("loanpy.reconstructor.ipa2clusters") as ipa2clusters_mock:
            ipa2clusters_mock.return_value = ["m", "ɒ", "jd"]
            rc.scdict = {"#m": ["m"], "ɒ": ["ɑ"], "j": ["ðʲ"], "d#": ["t"],
                         "#0": ["0"], "0#": ["ͽ"]}
            out9 = rc.reconstruct('mɒjd', howmany=1, struc=False,
                                  only_documented_clusters=True,
                                  vowelharmony=False, sort_by_nse=True)
            self.assertEqual(out9, 'jd# not old')

if __name__ == "__main__":
    unittest.main()
