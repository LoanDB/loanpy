import os
import unittest
from unittest.mock import patch

import pandas as pd

from loanpy import adapter as ad

os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


class Test1(unittest.TestCase):

    def test_launch(self):

        substidict_test = {"L2_phons": ["a", "b", "c", "d"],
                           "L1_substi": ["o, a", "p, w", "t, s", "t, n"]}
        substi_test = pd.DataFrame(substidict_test)
        substi_test.to_csv("substi_test.csv", encoding="utf-8", index=False)

        dfuralonet_test = pd.DataFrame({"Old": ['lɛvɛdi', 'aːlmoʃ', 'aːrpaːd'],
                                        "Lan": ["U", "'FU", "FU"]})
        dfuralonet_test.to_csv("dfuralonet_test.csv", encoding="utf-8",
                               index=False)

        scdict_test = str({"e": ["e", "i"], "f": ["p"], "g": ["p"]})
        with open("scdict_test.txt", "w") as f:
            f.write(scdict_test)

        with patch("loanpy.adapter.word2struc",
                   side_effect=["CVCVCV", "VCCVC", "VCCVC"]):
            ad.launch(dfetymology="dfuralonet_test.csv", timelayer="",
                      substicsv="substi_test.csv",
                      soundchangedict="scdict_test.txt")

        substidict_out = {'a': ['o', 'a'], 'b': ['p', 'w'], 'c': ['t', 's'],
                          'd': ['t', 'n']}
        self.assertDictEqual(ad.substidict, substidict_out)
        self.assertCountEqual(ad.allowedphonotactics, ["CVCVCV", "VCCVC"])
        self.assertCountEqual(ad.scvalues, {"e", "i", "p"})

        # test timelayer
        with patch("loanpy.adapter.word2struc", side_effect=["VCCVC"]*2):
            ad.launch(dfetymology="dfuralonet_test.csv", timelayer="FU",
                      substicsv="substi_test.csv",
                      soundchangedict="scdict_test.txt")
        self.assertCountEqual(ad.allowedphonotactics, ["VCCVC"])

        os.remove("scdict_test.txt")
        os.remove("substi_test.csv")
        os.remove("dfuralonet_test.csv")

    def test_adapt(self):
        ad.substidict = {'w': ['v', 'b'], 'u': ['u', 'o'],
                         'l': ['l', 'r'], 'ɸ': ['f'], 'ɪ': ['i'],
                         'a': ['a'], 'V': ['o'], 'C': ['v'],
                         'B': ['u'], 'F': ['i']}

        with patch("loanpy.adapter.tokenise") as tokenise_mock:
            tokenise_mock.return_value = ['w', 'u', 'l', 'ɸ', 'ɪ', 'l', 'a']
            out1 = ad.adapt("wulɸɪla", howmany=float("inf"), struc=False,
                            only_documented_clusters=False,
                            vowelharmony=False)
            out2 = ad.adapt("wulɸɪla", howmany=1, struc=False,
                            only_documented_clusters=False,
                            vowelharmony=False)
            out3 = ad.adapt("wulɸɪla", howmany=2, struc=False,
                            only_documented_clusters=False,
                            vowelharmony=False)
            self.assertEqual(out1, 'vulfila, vurfira, volfila, vorfira, \
bulfila, burfira, bolfila, borfira')
            self.assertEqual(out2, 'vulfila')
            self.assertEqual(out3, 'vulfila, bulfila')

        with patch("loanpy.adapter.tokenise") as tokenise_mock:
            tokenise_mock.return_value = ['v', 'ü', 'l', 'ɸ', 'ɪ', 'l', 'a']
            out4 = ad.adapt("vülɸɪla", howmany=float("inf"), struc=False,
                            only_documented_clusters=False,
                            vowelharmony=False)
        self.assertEqual(out4, 'v, ü not in substi.csv')

        with patch("loanpy.adapter.tokenise") as tokenise_mock:
            tokenise_mock.return_value = ['w', 'u', 'l', 'ɸ',
                                          'ɪ', 'l', 'a']
            with patch("loanpy.adapter.word2struc", side_effect=['CVCCVCV']):
                with patch("loanpy.adapter.editdistancewith2ops",
                           side_effect=[3, 0.4]):
                    with patch("loanpy.adapter.editops",
                               side_effect=[[('insert', 3, 3)]]):
                        with patch("loanpy.adapter.apply_edit",
                                   side_effect=['abcVdefg']):
                            ad.allowedphonotactics = ['CVCV', 'CVCVCVCV']
                            out5 = ad.adapt("wulɸɪla", howmany=3, struc=True,
                                            only_documented_clusters=False,
                                            vowelharmony=False)
                            self.assertEqual(out5, 'vulofila, volofila, \
bulofila, bolofila')

        with patch("loanpy.adapter.tokenise") as tokenise_mock:
            tokenise_mock.return_value = ['w', 'u', 'l', 'ɸ', 'ɪ', 'l']
            with patch("loanpy.adapter.word2struc", side_effect=['CVCCVCV']):
                with patch("loanpy.adapter.editdistancewith2ops",
                           side_effect=[3, 1]):
                    with patch("loanpy.adapter.editops",
                               side_effect=[[('delete', 6, 6)]]):
                        with patch("loanpy.adapter.apply_edit",
                                   side_effect=['abcdef']):
                            ad.allowedphonotactics = ['CVCV', 'CVCCVC']
                            ad.scvalues = ["v", "u", "lf", "i", "l", "b", "r"]
                            out6 = ad.adapt("wulɸɪla", howmany=4, struc=True,
                                            only_documented_clusters=True,
                                            vowelharmony=False)
                            self.assertEqual(out6, 'vulfil, bulfil')

        with patch("loanpy.adapter.tokenise") as tokenise_mock:
            tokenise_mock.return_value = ['w', 'u', 'l', 'ɸ', 'ɪ', 'l']
            with patch("loanpy.adapter.word2struc", side_effect=['CVCCVCV']):
                with patch("loanpy.adapter.editdistancewith2ops",
                           side_effect=[3, 1]):
                    with patch("loanpy.adapter.editops",
                               side_effect=[[('delete', 6, 6)]]):
                        with patch("loanpy.adapter.apply_edit",
                                   side_effect=['abcdef']):
                            ad.allowedphonotactics = ['CVCV', 'CVCCVC']
                            ad.scvalues = ["v", "lf", "i", "l", "b", "r"]
                            out7 = ad.adapt("wulɸɪla", howmany=5, struc=True,
                                            only_documented_clusters=True,
                                            vowelharmony=False)
                            self.assertEqual(out7, "every substituted word \
contains at least one cluster undocumented in proto-L1")

        with patch("loanpy.adapter.tokenise") as tokenise_mock:
            tokenise_mock.return_value = ['w', 'u', 'l', 'ɸ', 'ɪ', 'l']
            with patch("loanpy.adapter.word2struc", side_effect=['CVCCVCV']):
                with patch("loanpy.adapter.editdistancewith2ops",
                           side_effect=[3, 1]):
                    with patch("loanpy.adapter.editops",
                               side_effect=[[('delete', 6, 6)]]):
                        with patch("loanpy.adapter.apply_edit",
                                   side_effect=['abcdef']):
                            ad.allowedphonotactics = ['CVCV', 'CVCCVC']
                            ad.scvalues = ["v", "u", "lf", "i", "l", "b", "r"]
                            out8 = ad.adapt("wulɸɪla", howmany=6, struc=True,
                                            only_documented_clusters=True,
                                            vowelharmony=True)
                            self.assertCountEqual(out8, 'vulful, bulful')

if __name__ == "__main__":
    unittest.main()
