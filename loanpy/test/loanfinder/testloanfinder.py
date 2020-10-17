#To test: flatten, word2struc, ipa2clusters, matchfinder,statiestics, semsim, fndmatches
import unittest
import os
from loanpy import loanfinder as lp
import pandas as pd
import filecmp

testfolder=os.path.dirname(lp.__file__)+r"\test\loanfinder"
resultsfolder=os.path.dirname(os.path.dirname(testfolder))+r"\data\results"
os.chdir(testfolder)
dfstat=pd.read_csv("stattest.csv",encoding="utf-8")

class Test1(unittest.TestCase):
    def test_loanfinder(self):
        
        self.assertEqual(lp.flatten([["Bu","da"],["pest"]]), ["Bu","da","pest"])
        self.assertEqual(lp.word2struc("hortobaɟ"), "CVCCVCVC")
        self.assertEqual(lp.ipa2clusters("ɒjkɒ"), ["ɒ","jk","ɒ"])
        
        lp.matchfinder(r"^(j|m|s|w)?(t)(u|o)(r|jr|rk|ɣ|lm|rw|ɣr)(u|o)(e|æ|ɑ|je|jkɑ|ŋæ|we|ke|me|ŋe|ele)?$",1096,"allmatches")
        self.assertEqual(lp.dfallmatches.shape, (2,3))
        self.assertEqual(lp.dfallmatches.substi.tolist(), ["toro","toru"])
        self.assertEqual(lp.dfallmatches.hun_idx.tolist(), [1096,1096])
        self.assertEqual(lp.dfallmatches.got_idx.tolist(), [1106,1106])
        
        lp.statistics(dfstat, "test", "allmatches")
        self.assertTrue(filecmp.cmp(r"metadata_allmatches_test.txt",resultsfolder+r"\metadata_allmatches_test.txt"))
        os.remove(resultsfolder+r"\metadata_allmatches_test.txt") #delete file that was written to result folder
        
        self.assertAlmostEqual(lp.semsim("Debrecen","Székesfehérvár"), 0.4268976)
        self.assertAlmostEqual(lp.semsimdict["Debrecen, Székesfehérvár"][0], 0.4268976)
        
        os.chdir(testfolder+r"\all")
        lp.findmatches(df1=testfolder+r"\zaicz.csv", method="allmatches", name="_test")
        self.assertTrue(filecmp.cmp(r"cutoff_allmatches_test.csv",resultsfolder+r"\cutoff_allmatches_test.csv"))
        self.assertTrue(filecmp.cmp(r"metadata_allmatches_cutoff_test.txt",resultsfolder+r"\metadata_allmatches_cutoff_test.txt"))
        self.assertTrue(filecmp.cmp(r"metadata_allmatches_precutoff_test.txt",resultsfolder+r"\metadata_allmatches_precutoff_test.txt"))
        os.remove(resultsfolder+r"\cutoff_allmatches_test.csv")
        os.remove(resultsfolder+r"\metadata_allmatches_cutoff_test.txt")
        os.remove(resultsfolder+r"\metadata_allmatches_precutoff_test.txt")
        os.remove(resultsfolder+r"\semsimdistr_allmatches_test_cutoff.pdf")
        os.remove(resultsfolder+r"\semsimdistr_allmatches_test_precutoff.pdf")
        
        os.chdir(testfolder+r"\unique")
        lp.findmatches(df1=testfolder+r"\zaicz.csv", method="uniquematches", name="_test")
        self.assertTrue(filecmp.cmp("cutoff_uniquematches_test.csv",resultsfolder+r"\cutoff_uniquematches_test.csv"))
        self.assertTrue(filecmp.cmp(r"metadata_uniquematches_cutoff_test.txt",resultsfolder+r"\metadata_uniquematches_cutoff_test.txt"))
        self.assertTrue(filecmp.cmp(r"metadata_uniquematches_precutoff_test.txt",resultsfolder+r"\metadata_uniquematches_precutoff_test.txt"))
        os.remove(resultsfolder+r"\cutoff_uniquematches_test.csv")
        os.remove(resultsfolder+r"\metadata_uniquematches_cutoff_test.txt")
        os.remove(resultsfolder+r"\metadata_uniquematches_precutoff_test.txt")
        os.remove(resultsfolder+r"\semsimdistr_uniquematches_test_cutoff.pdf")
        os.remove(resultsfolder+r"\semsimdistr_uniquematches_test_precutoff.pdf")

if __name__ == '__main__':
    unittest.main()