import os
import unittest

import pandas as pd

from loanpy import adapter as ad

os.chdir(os.path.join(
         os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))


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

        ad.launch(dfetymology="dfuralonet_test.csv", timelayer="",
                  substicsv="substi_test.csv",
                  soundchangedict="scdict_test.txt")

        substidict_out = {'a': ['o', 'a'], 'b': ['p', 'w'], 'c': ['t', 's'],
                          'd': ['t', 'n']}
        self.assertDictEqual(ad.substidict, substidict_out)
        self.assertCountEqual(ad.allowedphonotactics, ["CVCVCV", "VCCVC"])
        self.assertCountEqual(ad.scvalues, {"e", "i", "p"})

        # test timelayer
        ad.launch(dfetymology="dfuralonet_test.csv", timelayer="FU",
                  substicsv="substi_test.csv",
                  soundchangedict="scdict_test.txt")
        self.assertCountEqual(ad.allowedphonotactics, ["VCCVC"])

        os.remove("scdict_test.txt")
        os.remove("substi_test.csv")
        os.remove("dfuralonet_test.csv")

    def test_adapt(self):

        ad.substidict = {'w': ['v', 'b'], 'u': ['u', 'o'], 'l': ['l', 'r'],
                         'ɸ': ['f'], 'ɪ': ['i'], 'a': ['a'],
                         'V': ['V'], 'C': ['C'], 'B': ['o'], 'F': ['i']}

        out1 = ad.adapt("wulɸɪla", howmany=float("inf"), struc=False,
                        only_documented_clusters=False, vowelharmony=False)
        out2 = ad.adapt("wulɸɪla", howmany=1, struc=False,
                        only_documented_clusters=False, vowelharmony=False)
        out3 = ad.adapt("wulɸɪla", howmany=2, struc=False,
                        only_documented_clusters=False, vowelharmony=False)
        self.assertEqual(out1, 'vulfila, vurfira, volfila, vorfira, \
bulfila, burfira, bolfila, borfira')
        self.assertEqual(out2, 'vulfila')
        self.assertEqual(out3, 'vulfila, bulfila')

        out4 = ad.adapt("vylɸɪla", howmany=float("inf"), struc=False,
                        only_documented_clusters=False, vowelharmony=False)
        self.assertEqual(out4, 'v, y not in substi.csv')

        ad.allowedphonotactics = ['CVCV', 'CVCVCVCV']
        out5 = ad.adapt("wulɸɪla", howmany=3, struc=True,
                        only_documented_clusters=False, vowelharmony=False)
        self.assertEqual(out5, 'vulVfila, volVfila, bulVfila, bolVfila')

        ad.allowedphonotactics = ['CVCV', 'CVCCVC']
        ad.scvalues = ["v", "u", "lf", "i", "l", "b", "r"]
        out6 = ad.adapt("wulɸɪla", howmany=4, struc=True,
                        only_documented_clusters=True, vowelharmony=False)

        ad.allowedphonotactics = ['CVCV', 'CVCCVC']
        ad.scvalues = ["v", "lf", "i", "l", "b", "r"]
        out7 = ad.adapt("wulɸɪla", howmany=5, struc=True,
                        only_documented_clusters=True, vowelharmony=False)
        self.assertEqual(out7, "every substituted word contains \
at least one cluster undocumented in proto-L1")

        ad.allowedphonotactics = ['CVCV', 'CVCCVC']
        ad.scvalues = ["v", "u", "lf", "i", "l", "o", "r"]
        out7 = ad.adapt("wulɸɪla", howmany=6, struc=True,
                        only_documented_clusters=True, vowelharmony=True)
        self.assertEqual(out7, 'vulfol, volfol')

if __name__ == '__main__':
    unittest.main()
