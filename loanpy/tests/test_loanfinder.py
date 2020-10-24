import unittest

import pandas as pd

from loanpy import loanfinder as lf

class Test1(unittest.TestCase):
            
    def test_reconstruct(self):
        lf.dfgot=pd.DataFrame({"substi":[["Baja"],["Tata", "Tatabánya"],["Rajka"]]}).explode("substi")
        lf.matchfinder("(B|R)aj(k)?a",42)
        pd.testing.assert_frame_equal(lf.dfmatches, pd.DataFrame({"substi":["Baja","Rajka"],"hun_idx":[42,42]}).set_index(pd.Series([0,2])).astype("object"))        
        lf.dfmatches=pd.DataFrame()
        
    def test_getsynonyms(self):
        self.assertEqual(lf.getsynonyms("horse","n"),['horse','Equus_caballus','gymnastic_horse','cavalry','horse_cavalry','sawhorse','sawbuck','buck','knight'])
        
    def test_semsim(self):
        lf.hun='chain, concatenation, chemical_chain, Chain, Ernst_Boris_Chain, Sir_Ernst_Boris_Chain,'\
         ' range, mountain_range, range_of_mountains, mountain_chain, chain_of_mountains, string, strand'
        lf.got='bridge, span, bridge_circuit, bridgework, nosepiece, bridge_deck'
        self.assertAlmostEqual(lf.semsim("chain","bridge"),0.31614783)
         
if __name__ == "__main__":
    unittest.main()