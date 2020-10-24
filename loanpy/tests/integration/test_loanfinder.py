import os
import unittest

from loanpy import loanfinder as lf

import pandas as pd
from pandas._testing import assert_frame_equal

os.chdir(os.path.join(
         os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))


class Test1(unittest.TestCase):

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
        sedicttest = {"#T<*T": 1, "o<*o": 1, "k<*k": 1, "a<*a": 1,
                      "j#<*j": 1, "#g<*G": 1, "s#<*sch": 1}
        with open("sedict_test.txt", "w", encoding="utf-8") as data:
            data.write(str(sedicttest))
        edicttest = {"#T<*T": ["a"], "o<*o": ["b"], "k<*k": ["c"],
                     "a<*a": ["d", "e"], "j#<*j": ["f"], "#g<*G": ["g"],
                     "s#<*sch": ["h"]}
        with open("edict_test.txt", "w", encoding="utf-8") as data:
            data.write(str(edicttest))

        lf.findloans(sheetname="test", L1="dfL1_test.csv",
                     L2="dfL2_test.csv", L1col="search_for",
                     L2col="menu_ger",
                     cutoff=2, write=True,
                     sedictname="sedict_test.txt", edictname="edict_test.txt")

        expectedframe = pd.DataFrame({'menu_ger_match':
                                      {0: 'Tokaj', 1: 'Gulasch'},
                                      'L1_idx': {0: 0, 1: 1},
                                      'gensim_similarity':
                                      {0: 1.0, 1: 0.8180971145629883},
                                      'menu_ger': {0: 'Tokaj', 1: 'Gulasch'},
                                      'L2_en': {0: 'wine', 1: 'soup'},
                                      'L2_pos': {0: 'n', 1: 'n'},
                                      'price': {0: '30000ft', 1: '3000ft'},
                                      'L1_ipa': {0: 'Tokaj', 1: 'gulyás'},
                                      'search_for':
                                      {0: '^(t|T)(o)(k)(a)(j|y)$',
                                       1: '^(g|G)(u)(l|ly)(á|a)(s|sch)$'},
                                      'L1_en':
                                      {0: 'wine', 1: 'stew'},
                                      'L1_pos': {0: 'n', 1: 'n'},
                                      'nse':
                                      {0: 0.714285714, 1: 0.285714286},
                                      'se': {0: 5, 1: 2},
                                      'lne': {0: "[1, 1, 1, 1, 1]",
                                              1: "[1, 1]"},
                                      'e': {0: "[['a'], ['b'], ['c'], \
['d', 'e'], ['f']]", 1: "[['g'], ['h']]"}})

        assert_frame_equal(pd.read_excel("results.xlsx",
                                         sheet_name="test",
                                         engine="openpyxl"),
                           expectedframe, check_dtype=False)

        os.remove("dfL1_test.csv")
        os.remove("dfL2_test.csv")
        os.remove("sedict_test.txt")
        os.remove("edict_test.txt")

        dict2deltest = pd.read_excel("results.xlsx",
                                     sheet_name=None,
                                     engine="openpyxl")  # delete test-sheet
        del dict2deltest["test"]
        writer = pd.ExcelWriter("results.xlsx", engine='openpyxl')
        for sheet_name in dict2deltest.keys():
            dict2deltest[sheet_name].to_excel(writer,
                                              sheet_name=sheet_name,
                                              index=False)
        writer.save()

if __name__ == '__main__':
    unittest.main()
