import unittest

from loanpy import makedicts as md

import filecmp
import os

path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+r"\data\generator"

class Test1(unittest.TestCase):
    
    def test_uralonet2scdict(self):
        md.uralonet2scdict("_test")
        self.assertTrue(filecmp.cmp(path+r"\soundchangedict_test.txt",path+r"\soundchangedict.txt"))
        os.remove(path+"\soundchangedict_test.txt")
        
    def test_substiclust(self):
        md.substiclust("_test")
        self.assertTrue(filecmp.cmp(path+r"\substidict_test.txt",path+r"\substidict.txt"))
        os.remove(path+r"\substidict_test.txt")
        
if __name__ == '__main__':
    unittest.main()