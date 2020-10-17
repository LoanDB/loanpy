import unittest
import os
import filecmp
from loanpy import generator as gr

rootdir=os.path.dirname(gr.__file__)
os.chdir(rootdir+r"\data\generator")

allowedclust={"e","i","j","jm","jr","jt͡ʃʲ","k","kl","ks","kt","l","lk","lm","m","mp","mt","n","nt","nʲ","nʲt͡ʃʲ","o","p","pp","r","s","t","tk","tt","t͡ʃʲ","u","w","¨","æ","ð","ðʲ","ðʲk","ŋ","ŋs","ȣ","ɑ","ɜ","ɤ","ʃ","ʃʲ","ʃʲk"}
wordstruc={"CV", "CVCCV", "CVCV", "V", "VCCV", "VCV"}
substiphons={"","a","e","i","j","k","l","m","n","o","p","r","s","t","u","w","y","æ","ŋ","ɑ","ɣ","ʃʲ","γ","ð"}
substi="tɑja, tɑjæ, tɑka, tɑkæ, tɑŋa, tɑŋæ, tɑsa, tɑsæ, tɑna, tɑnæ, tɑksa, tɑksæ, tɑŋsa, tɑŋsæ, toja, tojæ, toka, tokæ, toŋa, toŋæ, tosa, tosæ, tona, tonæ, toksa, toksæ, toŋsa, toŋsæ, nɑja, nɑjæ, nɑka, nɑkæ, nɑŋa, nɑŋæ, nɑsa, nɑsæ, nɑna, nɑnæ, nɑksa, nɑksæ, nɑŋsa, nɑŋsæ, noja, nojæ, noka, nokæ, noŋa, noŋæ, nosa, nosæ, nona, nonæ, noksa, noksæ, noŋsa, noŋsæ, rɑja, rɑjæ, rɑka, rɑkæ, rɑŋa, rɑŋæ, rɑsa, rɑsæ, rɑna, rɑnæ, rɑksa, rɑksæ, rɑŋsa, rɑŋsæ, roja, rojæ, roka, rokæ, roŋa, roŋæ, rosa, rosæ, rona, ronæ, roksa, roksæ, roŋsa, roŋsæ"

class Test1(unittest.TestCase):
    def test_loanfinder(self):
        
        self.assertEqual(gr.flatten([["Bu","da"],["pest"]]), ["Bu","da","pest"])
        self.assertEqual(gr.word2struc("hortobaɟ"), "CVCCVCVC")
        self.assertEqual(gr.ipa2clusters("ɒjkɒ"), ["ɒ","jk","ɒ"])
        self.assertEqual(gr.list2regex(["a","b","c"]), "(a|b|c)")
        self.assertEqual(gr.list2regex(["a","b","c", "0"]), "(a|b|c)?")
        self.assertEqual(gr.list2regex(["0"]), "")
        
        gr.extractsoundchange("ɟɒloɡ","jɑlkɑ")
        self.assertEqual(gr.dfsoundchange.reflex.tolist(), ["#ɟ","ɒ","l","o","ɡ#"])
        self.assertEqual(gr.dfsoundchange.root.tolist(), ["j","ɑ","lk","ɑ","0"])
        self.assertEqual(gr.affix, ["ɡ#"])        
        
        gr.uralonet2scdict(name="_test")
        self.assertTrue(filecmp.cmp(r"soundchangedict.txt",r"soundchangedict_test.txt"))
        os.remove(r"soundchangedict_test.txt")
        
        gr.setconstraints("U")
        self.assertEqual(gr.allowedclust, allowedclust)
        self.assertEqual(gr.maxclustlen, 2)
        self.assertEqual(gr.wordstruc, wordstruc)
        self.assertEqual(gr.maxnrofclusters, 4)
        self.assertEqual(gr.substiphons, substiphons)
        
        self.assertEqual(gr.reconstruct("bɒlɒton"), "^(j|m|s|w)?(p)(e|o|u|æ|ɑ)(l|lk|lw|lɣ|m|r|w|wl|wð|ð)(e|o|u|æ|ɑ)(j|kt|pt|tk)(e|i|o|u|ɑ)(e|æ|ɑ|je|jkɑ|ŋæ|we|ke|me|ŋe|ele)?$")
        
        gr.addreconstr("_test")
        self.assertTrue(filecmp.cmp(rootdir+r"\data\zaicz.csv",rootdir+r"\data\zaicz_test.csv"))
        os.remove(rootdir+r"\data\zaicz_test.csv")
        
        self.assertEqual(gr.deletion("hts"),"j, k, ŋ, t, s, kt, ks, ŋs")
        self.assertEqual(gr.deletion("auaɪ"),"æ, u, o, i, e")
        
        gr.getsubstis("_test")
        self.assertTrue(filecmp.cmp(r"substidict.txt",r"substidict_test.txt"))
        os.remove(r"substidict_test.txt")
        
        self.assertEqual(gr.substitute("drɔhsna"),substi)
        self.assertEqual(gr.substitute("maθlis"), "too long")
        
        gr.addsubsti("_test")
        self.assertTrue(filecmp.cmp(rootdir+r"\data\dfgot.csv",rootdir+r"\data\dfgot_test.csv"))
        os.remove(rootdir+r"\data\dfgot_test.csv")