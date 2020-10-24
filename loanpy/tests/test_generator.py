import unittest
from unittest.mock import patch
from mock import call
import os

import pandas as pd

from loanpy import generator as gr

path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+r"\data"

class Test1(unittest.TestCase):
            
    def test_reconstruct(self):
        gr.scdict = {"#b": "(p)","ɒ": "(e|o|u|æ|ɑ)","l": "(l|lk|lw|lɣ|m|r|w|wl|wð|ð)","t": "(j|kt|pt|tk)","o": "(e|i|o|u|ɑ)","n#": ""}
        with patch("loanpy.generator.ipa2clusters") as ipa2clusters_mock:
            ipa2clusters_mock.return_value = ["b", "ɒ", "l", "ɒ", "t", "o", "n"]
            self.assertEqual(gr.reconstruct("bɒlɒton"), "^(j|m|s)?(p)(e|o|u|æ|ɑ)(l|lk|lw|lɣ|m|r|w|wl|wð|ð)(e|o|u|æ|ɑ)(j|kt|pt|tk)(e|i|o|u|ɑ)(e|æ|ɑ|je|ke|ŋe|ŋæ)?$")
        del gr.scdict
        
    def test_substitute(self):
        gr.maxnrofclusters=4
        gr.wordstruc={"CV", "CVCCV", "CVCV", "V", "VCCV", "VCV"}
        gr.substidict={"dr": "t, n, r","ɔ": "ɑ, o","hsn": "j, k, ŋ, s, n, ks, ŋs","a":"a, æ","m":"m","θl": "t, s, l","i": "i, e","s":"s"}
        word2struc_side_effect= ['CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCCV','CVCCV','CVCCV','CVCCV','CVCV','CVCV','CVCV',\
                                 'CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCCV','CVCCV','CVCCV','CVCCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV',\
                                 'CVCV','CVCV','CVCV','CVCV','CVCCV','CVCCV','CVCCV','CVCCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV',\
                                 'CVCV','CVCCV','CVCCV','CVCCV','CVCCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCCV','CVCCV',\
                                 'CVCCV','CVCCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCV','CVCCV','CVCCV','CVCCV','CVCCV']
        word2struc_called_with= [call('tɑja'),call('tɑjæ'),call('tɑka'),call('tɑkæ'),call('tɑŋa'),call('tɑŋæ'),call('tɑsa'),call('tɑsæ'),call('tɑna'),\
                                 call('tɑnæ'),call('tɑksa'),call('tɑksæ'),call('tɑŋsa'),call('tɑŋsæ'),call('toja'),call('tojæ'),call('toka'),call('tokæ'),\
                                 call('toŋa'),call('toŋæ'),call('tosa'),call('tosæ'),call('tona'),call('tonæ'),call('toksa'),call('toksæ'),call('toŋsa'),\
                                 call('toŋsæ'),call('nɑja'),call('nɑjæ'),call('nɑka'),call('nɑkæ'),call('nɑŋa'),call('nɑŋæ'),call('nɑsa'),call('nɑsæ'),\
                                 call('nɑna'),call('nɑnæ'),call('nɑksa'),call('nɑksæ'),call('nɑŋsa'),call('nɑŋsæ'),call('noja'),call('nojæ'),call('noka'),\
                                 call('nokæ'),call('noŋa'),call('noŋæ'),call('nosa'),call('nosæ'),call('nona'),call('nonæ'),call('noksa'),call('noksæ'),\
                                 call('noŋsa'),call('noŋsæ'),call('rɑja'),call('rɑjæ'),call('rɑka'),call('rɑkæ'),call('rɑŋa'),call('rɑŋæ'),call('rɑsa'),\
                                 call('rɑsæ'),call('rɑna'),call('rɑnæ'),call('rɑksa'),call('rɑksæ'),call('rɑŋsa'),call('rɑŋsæ'),call('roja'),call('rojæ'),\
                                 call('roka'),call('rokæ'),call('roŋa'),call('roŋæ'),call('rosa'),call('rosæ'),call('rona'),call('ronæ'),call('roksa'),\
                                 call('roksæ'),call('roŋsa'),call('roŋsæ')]
        with patch("loanpy.generator.ipa2clusters",side_effect=[["dr","ɔ","hsn","a"],["m","a","θl","i","s"]]) as ipa2clusters_mock:
            with patch("loanpy.generator.word2struc", side_effect=word2struc_side_effect) as word2struc_mock:
                

                self.assertEqual(gr.substitute("drɔhsna"), "tɑja, tɑjæ, tɑka, tɑkæ, tɑŋa, tɑŋæ, tɑsa, tɑsæ, tɑna, tɑnæ, tɑksa, tɑksæ, tɑŋsa, tɑŋsæ, toja, tojæ, toka, tokæ, toŋa, toŋæ, tosa, tosæ, tona, tonæ, toksa, "\
"toksæ, toŋsa, toŋsæ, nɑja, nɑjæ, nɑka, nɑkæ, nɑŋa, nɑŋæ, nɑsa, nɑsæ, nɑna, nɑnæ, nɑksa, nɑksæ, nɑŋsa, nɑŋsæ, noja, nojæ, noka, nokæ, noŋa, noŋæ, nosa, nosæ, nona, nonæ, "\
"noksa, noksæ, noŋsa, noŋsæ, rɑja, rɑjæ, rɑka, rɑkæ, rɑŋa, rɑŋæ, rɑsa, rɑsæ, rɑna, rɑnæ, rɑksa, rɑksæ, rɑŋsa, rɑŋsæ, roja, rojæ, roka, rokæ, roŋa, roŋæ, rosa, rosæ, rona, "\
"ronæ, roksa, roksæ, roŋsa, roŋsæ")
                
                self.assertEqual(gr.substitute("maθlis"), "too long")
                self.assertEqual(ipa2clusters_mock.call_args_list, [call("drɔhsna"), call("maθlis")])
                self.assertEqual(word2struc_mock.call_args_list, (word2struc_called_with))
        del gr.maxnrofclusters
        del gr.wordstruc
        del gr.substidict
        
    def test_addproto(self):
        def adjectify(word):
            return word+"i"
        cities=pd.DataFrame({"noun":["Debrecen","Nagykanizsa","Pécs"]})
        gr.addproto(cities, "noun", adjectify, "adjective", "dfaddproto_test.csv")
        result = pd.read_csv(path+r"\dfaddproto_test.csv", encoding="utf-8")
        pd.testing.assert_frame_equal(result, pd.DataFrame({"noun":["Debrecen","Nagykanizsa","Pécs"], "adjective":["Debreceni","Nagykanizsai","Pécsi"]}))
        os.remove(path+"\dfaddproto_test.csv")
        
if __name__ == "__main__":
    unittest.main()