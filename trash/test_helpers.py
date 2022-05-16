import os
import unittest
from unittest.mock import patch

import gensim.downloader as api
import pandas as pd
from pandas._testing import assert_frame_equal

from loanpy import helpers as hp

os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


class Test1(unittest.TestCase):

    def test_phon2cv(self):
        self.assertEqual(hp.phon2cv("p"), "C")
        self.assertEqual(hp.phon2cv("a"), "V")

    def test_word2struc(self):
        with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ["h", "o", "r", "t", "o",
                                            "b", "a", "ɟ"]
            with patch("loanpy.helpers.phon2cv",
                       side_effect=["C", "V", "C", "C", "V", "C", "V", "C"]):
                self.assertEqual(hp.word2struc("hortobaːɟ"), "CVCCVCVC")

    def test_ipa2clusters(self):
        with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ["a", "b", "auː", "j",
                                            "k", "eː", "r"]
            self.assertEqual(hp.ipa2clusters("abauːjkeːr"),
                             ['a', 'b', 'auː', 'jk', 'eː', 'r'])

    def test_list2regex(self):
        self.assertEqual(hp.list2regex(["b", "k", "v"]), "(b|k|v)")
        self.assertEqual(hp.list2regex(["b", "k", "0", "v"]), "(b|k|v)?")
        self.assertEqual(hp.list2regex(["b", "k", "0", "v", "mp"]),
                         "(b|k|v|mp)?")
        self.assertEqual(hp.list2regex(["b", "k", "0", "v", "mp", "mk"]),
                         "(b|k|v|mp|mk)?")
        self.assertEqual(hp.list2regex(["o"]), '(o)')
        self.assertEqual(hp.list2regex(["ʃʲk"]), '(ʃʲk)')

    def test_editdistancewith2ops(self):
        self.assertAlmostEqual(hp.editdistancewith2ops("ajka", "Rajka"),  0.4)
        self.assertAlmostEqual(hp.editdistancewith2ops("Debrecen",
                                                       "Mosonmagyaróvár"),
                               12.6)

    def test_gensim(self):
        hp.model = api.load("glove-wiki-gigaword-50")
        hp.L1 = 'chain, concatenation, chemical_chain, Chain, Ernst_Boris_Chain, \
Sir_Ernst_Boris_Chain, range, mountain_range, range_of_mountains, \
mountain_chain, chain_of_mountains, string, strand'
        hp.L2 = 'bridge, span, bridge_circuit, bridgework, nosepiece, \
bridge_deck'

        self.assertAlmostEqual(hp.gensim_similarity("chain", "bridge"),
                               0.34358343)
        model = None  # deletion would lead to errors
        del hp.L1
        del hp.L2

    def test_filterdfin(self):
        dfwinemenu = pd.DataFrame({"wine": ["Egri Bikavér", "Tokaji Aszú"],
                                   "colour": ["red", "white"],
                                   "price_in_forint": [10000, 30000]})
        dfred = pd.DataFrame({"wine": ["Egri Bikavér"],
                              "colour": ["red"],
                              "price_in_forint": [10000]})
        dfwhite = pd.DataFrame({"wine": ["Tokaji Aszú"],
                                "colour": ["white"],
                                "price_in_forint": [30000]})
        dfin1 = hp.filterdf(df=dfwinemenu, col="colour",
                            occurs_or_bigger=True,
                            term="red").reset_index(drop=True)
        dfin2 = hp.filterdf(df=dfwinemenu, col="colour",
                            occurs_or_bigger=False,
                            term="red").reset_index(drop=True)
        dfin3 = hp.filterdf(df=dfwinemenu, col="price_in_forint",
                            occurs_or_bigger=True,
                            term=10000).reset_index(drop=True)
        dfin4 = hp.filterdf(df=dfwinemenu, col="price_in_forint",
                            occurs_or_bigger=False,
                            term=10000).reset_index(drop=True)
        assert_frame_equal(dfin1, dfred.reset_index(drop=True))
        assert_frame_equal(dfin2, dfwhite.reset_index(drop=True))
        assert_frame_equal(dfin3, dfwhite.reset_index(drop=True))
        assert_frame_equal(dfin4, dfred.reset_index(drop=True))

    def test_vow2frontback(self):

        self.assertEqual(hp.vow2frontback("e", replace=False), "F")
        self.assertEqual(hp.vow2frontback("o", replace=False), "B")
        self.assertEqual(hp.vow2frontback("e", replace=True), "B")
        self.assertEqual(hp.vow2frontback("o", replace=True), "F")

    def test_harmony(self):
        with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ['b', 'o', 't͡s', 'i',
                                            'b', 'o', 't͡s', 'i']
            with patch("loanpy.helpers.vow2frontback", side_effect=["B", "F",
                                                                    "B", "F"]):
                self.assertEqual(hp.harmony("bot͡sibot͡si"), False)

        with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ['t', 'ɒ', 'r', 'k', 'ɒ']
            with patch("loanpy.helpers.vow2frontback",
                       side_effect=["B", "B"]) as vow2frontback_mock:
                self.assertEqual(hp.harmony("tɒrkɒ"), True)

        with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ['ʃ', 'ɛ', 'f', 'y', 'l', 'ɛ',
                                            'ʃ', 'ɛ']
            with patch("loanpy.helpers.vow2frontback",
                       side_effect=["F", "F", "F", "F"]) as vow2frontback_mock:
                self.assertEqual(hp.harmony("ʃɛfylɛʃɛ"), True)

    def test_adaptharmony(self):

        with patch("loanpy.helpers.harmony") as harmony_mock:
            harmony_mock.return_value = True
            self.assertEqual(hp.adaptharmony('kɛsthɛj'),
                             ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j'])
        with patch("loanpy.helpers.harmony") as harmony_mock:
            harmony_mock.return_value = False
            with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
                ipa2tokens_mock.return_value = ['ɒ', 'l', 'ʃ', 'oː',
                                                'ø', 'r', 'ʃ']
                with patch("loanpy.helpers.vow2frontback",
                           side_effect=["B", "B", "F", "B"]):
                        self.assertEqual(hp.adaptharmony('ɒlʃoːørʃ'),
                                         ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ'])
        with patch("loanpy.helpers.harmony") as harmony_mock:
            harmony_mock.return_value = False
            with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
                ipa2tokens_mock.return_value = ['ʃ', 'i', 'oː', 'f', 'o', 'k']
                with patch("loanpy.helpers.vow2frontback",
                           side_effect=["F", "B", "B", "B"]):
                    self.assertEqual(hp.adaptharmony('ʃioːfok'),
                                     ['ʃ', 'B', 'oː', 'f', 'o', 'k'])
        with patch("loanpy.helpers.harmony") as harmony_mock:
            harmony_mock.return_value = False
            with patch("loanpy.helpers.tokenise") as ipa2tokens_mock:
                ipa2tokens_mock.return_value = ['b', 'ɒ', 'l', 'ɒ',
                                                't', 'o', 'n',
                                                'k', 'ɛ', 'n', 'ɛ', 'ʃ', 'ɛ']
                with patch("loanpy.helpers.vow2frontback",
                           side_effect=["B", "B", "B", "F", "F", "F",
                                        "B", "B", "B"]):
                    self.assertEqual(hp.adaptharmony('bɒlɒtonkɛnɛʃɛ'),
                                     ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                                      'k', 'B', 'n', 'B', 'ʃ', 'B'])

if __name__ == "__main__":
    unittest.main()
