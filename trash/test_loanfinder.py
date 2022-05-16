import os
import unittest
from unittest.mock import patch

import pandas as pd
from pandas._testing import assert_frame_equal

from loanpy import loanfinder as lf

os.chdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))


class Test1(unittest.TestCase):

    def test_adapt_or_reconstruct_col(self):

        dfplace = pd.DataFrame({"place": ["Tata", "Pápa", "Aka"]})
        dfplace.to_csv("adrc_test.csv", index=False)

        with patch("loanpy.loanfinder.launch_reconstructor") as launch_mock:
            launch_mock.return_value = None
            with patch("loanpy.loanfinder.reconstruct",
                       side_effect=["Dada", "Bába", "Aga"]):
                lf.adapt_or_reconstruct_col(inputcsv="adrc_test.csv",
                                            inputcol="place",
                                            funcname="reconstruct",
                                            howmany=1,
                                            struc=False,
                                            vowelharmony=False,
                                            outputcsv="",
                                            outputcol="",
                                            write=True)

                result = pd.read_csv("adrc_test.csv", encoding="utf-8")
                expected = dfplace.assign(reconstruct=["Dada", "Bába", "Aga"])
                assert_frame_equal(result, expected)

        dfplace = pd.DataFrame({"place": ["Tata", "Pápa", "Aka"]})
        dfplace.to_csv("adrc_test.csv", index=False)

        with patch("loanpy.loanfinder.launch_adapter") as launch_mock:
            launch_mock.return_value = None
            with patch("loanpy.loanfinder.adapt",
                       side_effect=["Tatta", "Páppa", "Akka"]):
                lf.adapt_or_reconstruct_col(inputcsv="adrc_test.csv",
                                            inputcol="place",
                                            funcname="adapt",
                                            howmany=1,
                                            struc=False,
                                            vowelharmony=False,
                                            write=True)

                result = pd.read_csv("adrc_test.csv", encoding="utf-8")
                expected = dfplace.assign(adapt=["Tatta", "Páppa", "Akka"])
                assert_frame_equal(result, expected)

        self.assertRaises(Exception,
                          lf.adapt_or_reconstruct_col,
                          "adrc_test.csv",
                          "place",
                          "wrongfuncname")

        os.remove("adrc_test.csv")

    def test_findphoneticmatches(self):
        lf.dfL2 = pd.Series(["Gulasch", "Paprika", "Tokaj", "Tokay"])
        lf.dfL2.name = "menu_ger"
        shouldbe = pd.DataFrame({"menu_ger_match": ["Tokaj", "Tokay"],
                                 "L1_idx": [0, 0]})
        shouldbe = shouldbe.set_index(pd.Series([2, 3]))
        assert_frame_equal(lf.findphoneticmatches("^(t|T)(o)(k)(a)(j|y)$", 0),
                           shouldbe, check_dtype=False)

    def test_findloans(self):
        dfdinner = pd.DataFrame({"menu_ger": ["Gulasch", "Paprika",
                                              "Tokaj", "Tokay"],
                                 "L2_en": ["soup", "pepper", "wine", "drink"],
                                 "L2_pos": ["n", "n", "n", "n"],
                                 "price": ["3000ft", "200ft",
                                           "30000ft", "20000ft"]})
        dfappetite = pd.DataFrame({"L1_ipa": ["Tokaj", "gulyás"],
                                   "search_for":
                                   ["^(t|T)(o)(k)(a)(j|y)$",
                                    "^(g|G)(u)(l|ly)(á|a)(s|sch)$"],
                                   "L1_en": ["wine", "stew"],
                                   "L1_pos": ["n", "n"]})
        dfdinner.to_csv("dfL2_test.csv", encoding="utf-8", index=False)
        dfappetite.to_csv("dfL1_test.csv", encoding="utf-8", index=False)
        expected_out = pd.DataFrame({'menu_ger_match':
                                     {2: 'Tokaj', 0: 'Gulasch'},
                                     'L1_idx': {2: 0, 0: 1},
                                     'gensim_similarity':
                                     {2: 1.0, 0: 0.8180971145629883},
                                     'menu_ger': {2: 'Tokaj', 0: 'Gulasch'},
                                     'L2_en': {2: 'wine', 0: 'soup'},
                                     'L2_pos': {2: 'n', 0: 'n'},
                                     'price': {2: '30000ft', 0: '3000ft'},
                                     'L1_ipa': {2: 'Tokaj', 0: 'gulyás'},
                                     'search_for':
                                     {2: '^(t|T)(o)(k)(a)(j|y)$',
                                      0: '^(g|G)(u)(l|ly)(á|a)(s|sch)$'},
                                     'L1_en': {2: 'wine', 0: 'stew'},
                                     'L1_pos': {2: 'n', 0: 'n'},
                                     'nse': {2: 1, 0: 4},
                                     'se': {2: 11, 0: 11},
                                     'lne': {2: [11], 0: [11]},
                                     'e': {2: "long list 1",
                                           0: "long list 2"}})

        df1 = pd.DataFrame({"menu_ger_match": ["Tokaj", "Tokay"],
                            "L1_idx": [0, 0]}, [2, 3])
        df2 = pd.DataFrame({"menu_ger_match": ["Gulasch"],
                            "L1_idx": [1]}, [0])
        with patch("loanpy.loanfinder.findphoneticmatches",
                   side_effect=[df1, df2]):
            with patch("loanpy.loanfinder.gensim_similarity",
                       side_effect=[1.0, 0.8180971145629883]):
                with patch("loanpy.loanfinder.launch_reconstructor") as l_mock:
                    l_mock.return_value = None
                    getnse_side_effect = [1, 4, 11, 11, [11], [11],
                                          "long list 1", "long list 2"]
                    with patch("loanpy.loanfinder.getnse",
                               side_effect=getnse_side_effect*2):

                        out = lf.findloans(sheetname="test",
                                           L1="dfL1_test.csv",
                                           L2="dfL2_test.csv",
                                           L1col="search_for",
                                           L2col="menu_ger",
                                           cutoff=2,
                                           write=False,
                                           sedictname="sedict_test.txt",
                                           edictname="edict_test.txt")

                        assert_frame_equal(out, expected_out,
                                           check_dtype=False)

        os.remove("dfL1_test.csv")
        os.remove("dfL2_test.csv")

if __name__ == "__main__":
    unittest.main()
