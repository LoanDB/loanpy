import unittest
from unittest.mock import patch
import os

import pandas as pd

from loanpy import makedicts as md

class Test1(unittest.TestCase):
    
    def test_extractsoundchange(self):
        with patch("loanpy.makedicts.ipa2clusters", side_effect=[["ɟ","ɒ","l","o","ɡ"],["j", "ɑ", "lk", "ɑ"]]) as ipa2clusters_mock:
            md.substiphons={"","a","e","i","j","k","l","m","n","o","p","r","s","t","u","w","y","æ","ŋ","ɑ","ɣ","ʃʲ","γ","ð"}
            md.extractsoundchange("ɟɒloɡ","jɑlkɑ")
            self.assertEqual(md.dfsoundchange.reflex.tolist(), ["#ɟ","ɒ","l","o","ɡ#"])
            self.assertEqual(md.dfsoundchange.root.tolist(), ["j","ɑ","lk","ɑ","0"])
            
    def test_uralonet2scdict(self):
        md.dfural=pd.DataFrame({"New": ["mɛʃɛ","aːɟ","ɒl"], "Old": ["ɑt͡ʃʲɑ","ɑðʲɜ","ɑlɑ"]})
        md.dfsoundchange=pd.DataFrame({"reflex":["#m","ɛ","ɛ#","#a","#ɒ","l","0#"],"root":["0","ɑ","ɑ","ɑ","ɑ","l","ɑ"]})
        regex1="(ɑ)" #e.g. ʃ-t͡ʃʲ pair got thrown out b/c t͡ʃʲ won't substitute any Gothic phoneme
        r2=""
        r3="(ɑ)"
        r4="(ɑ)"
        r5="(l)"
        r6="(ɑ)"
        r7="(ɑ)"
        with patch("loanpy.makedicts.extractsoundchange", side_effect=["whatever","whatever","whatever"]):
            with patch("loanpy.makedicts.list2regex", side_effect=[regex1, r2,r3,r4,r5,r6,r7]):
                md.uralonet2scdict(name="_test")
                
                with open("soundchangedict_test.txt", encoding="utf-8") as f:
                    scdict_test=f.read()
                    self.assertEqual(scdict_test, "{'#a': '(ɑ)', '#m': '', '#ɒ': '(ɑ)', '0#': '(ɑ)', 'l': '(l)', 'ɛ': '(ɑ)', 'ɛ#': '(ɑ)'}")
                os.remove("soundchangedict_test.txt")
                
    def test_deletion(self):
        md.maxclustlen=2
        md.allowedclust={"e","i","j","jm","jr","jt͡ʃʲ","k","kl","ks","kt","l","lk","lm","m","mp","mt","n","nt","nʲ","nʲt͡ʃʲ","o","p","pp","r","s","t","tk","tt",\
                         "t͡ʃʲ","u","w","¨","æ","ð","ðʲ","ðʲk","ŋ","ŋs","ȣ","ɑ","ɜ","ɤ","ʃ","ʃʲ","ʃʲk"}
        md.sudict={"a": "a, æ", "b": "p, m, w", "c": "t, s", "d": "t, n", "e": "e, i", "h": ", j, k, ŋ", "i": "i, e", "j": "j", "k": "k", "l": "l", "m": "m", \
                   "n": "n", "o": "o, u", "p": "p", "r": "r", "s": "s", "t": "t", "u": "u, o", "v": "w, p, u", "w": "w, u","x": "k, s", "y": "y", "z": "s, ʃʲ",\
                   "ð": "ð", "ŋ": "ŋ, k", "ɔ": "ɑ, o", "ɛ": "e, æ", "ɡ": "k, γ, ŋ", "ɣ": "ɣ", "ɪ": "i, e", "ɸ": "p, w, s", "β": "w, p, u", "θ": "t, s"}
        with patch("loanpy.makedicts.ipa2tokens",side_effect=[["h", "t", "s"],["a","u","a","ɪ"]]):
            self.assertEqual(md.deletion("hts"),"j, k, ŋ, t, s, kt, ks, ŋs")
            self.assertEqual(md.deletion("auaɪ"),"æ, u, o, i, e")
            
    def test_substiclust(self):
        md.dfgot=pd.DataFrame({"got_ipa":["aɸarsabbate","aɸaruh","aβrs"]})
        md.dfsubsti=pd.DataFrame({"to_substitute":["a","b","c","d","e","h","i","j","k","l","m","n","o","p","r","s","t","u","v","w","x","y","z","ð","ŋ","ɔ","ɛ","ɡ",\
                                                   "ɣ","ɪ","ɸ","β","θ"],\
                                  "substitution":["a, æ","p, m, w","t, s","t, n","e, i",", j, k, ŋ","i, e","j","k","l","m","n","o, u","p","r","s","t","u, o","w, p, u",\
                                                  "w, u","k, s","y","s, ʃʲ","ð","ŋ, k","ɑ, o","e, æ","k, γ, ŋ","ɣ","i, e","p, w, s","w, p, u","t, s"]})
        with patch("loanpy.makedicts.flatten") as flatten_mock:
            flatten_mock.return_value=["a", "ɸ", "a", "rs", "a", "bb", "a", "t", "e","a", "ɸ", "a", "r", "u", "h","a", "βrs"]
            with patch("loanpy.makedicts.deletion", side_effect=["p, m, w, pp, mp","r, s","w, p, u, r, s"]):
                md.substiclust(name="_test")
                with open("substidict_test.txt", encoding="utf-8") as f:
                    sudict_test=f.read()
        self.assertEqual(sudict_test, str({"a": "a, æ", "b": "p, m, w", "c": "t, s", "d": "t, n", "e": "e, i", "h": ", j, k, ŋ", "i": "i, e", "j": "j", "k": "k","l": "l",\
                                           "m": "m", "n": "n", "o": "o, u", "p": "p", "r": "r", "s": "s", "t": "t", "u": "u, o", "v": "w, p, u", "w": "w, u", "x": "k, s",\
                                           "y": "y", "z": "s, ʃʲ", "ð": "ð","ŋ": "ŋ, k", "ɔ": "ɑ, o", "ɛ": "e, æ", "ɡ": "k, γ, ŋ", "ɣ": "ɣ", "ɪ": "i, e", "ɸ": "p, w, s", \
                                           "β": "w, p, u", "θ": "t, s", "bb": "p, m, w, pp, mp","rs": "r, s", "βrs": "w, p, u, r, s"}))
        os.remove("substidict_test.txt")
        
if __name__ == "__main__":
    unittest.main()