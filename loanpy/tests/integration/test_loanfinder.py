import unittest
from loanpy import generator as gr
import filecmp
import os
import pandas as pd

path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+r"\data"

class Test1(unittest.TestCase):
    
    def test_generate_zaicz(self):
        zaicz=pd.read_csv("zaicz.csv")
        gr.addproto(dataframe=zaicz, inputcol="hun_ipa", function=gr.reconstruct, outputcol="regexroot", name="zaicz_test.csv")
        self.assertTrue(filecmp.cmp(path+r"\zaicz_test.csv",path+r"\zaicz.csv"))
        os.remove(path+r"\zaicz_test.csv")
        
    def test_generate_dfgot(self):
        dfgot=pd.read_csv("dfgot.csv")
        gr.addproto(dataframe=dfgot, inputcol="got_ipa", function=gr.substitute, outputcol="substi", name="dfgot_test.csv")
        self.assertTrue(filecmp.cmp(path+r"\dfgot_test.csv",path+r"\dfgot.csv"))
        os.remove(path+r"\dfgot_test.csv")
        
    def test_findmatches(self):
        pass #TODO run gr.findmatches() and check if the two outputfiles are identical

if __name__ == '__main__':
    unittest.main()