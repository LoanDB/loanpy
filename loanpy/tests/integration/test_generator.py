import unittest
import filecmp
import os

import pandas as pd

from loanpy import generator as gr

path2output=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+r"\data"
path2input=path2output+r"\generator"

class Test1(unittest.TestCase):
    
    def test_generate_zaicz(self):
        zaicz=pd.read_csv(path2input+r"\zaicz.csv")
        gr.addproto(dataframe=zaicz, inputcol="hun_ipa", function=gr.reconstruct, outputcol="regexroot", name="zaicz_test.csv")
        self.assertTrue(filecmp.cmp(path2output+r"\zaicz_test.csv",path2output+r"\zaicz.csv"))
        os.remove(path2output+r"\zaicz_test.csv")
        
    def test_generate_dfgot(self):
        dfgot=pd.read_csv(path2input+r"\dfgot.csv")
        gr.addproto(dataframe=dfgot, inputcol="got_ipa", function=gr.substitute, outputcol="substi", name="dfgot_test.csv")
        self.assertTrue(filecmp.cmp(path2output+r"\dfgot_test.csv",path2output+r"\dfgot.csv"))
        os.remove(path2output+r"\dfgot_test.csv")

if __name__ == '__main__':
    unittest.main()