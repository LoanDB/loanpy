import os
import unittest

from loanpy import helpers as hp

os.chdir(os.path.join(
         os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))


class Test1(unittest.TestCase):

    def test_gensim(self):
        hp.loadvectors("glove-wiki-gigaword-50")

        self.assertAlmostEqual(hp.gensim_similarity("chain", "bridge"),
                               0.34358343)

    def test_word2struc(self):
        self.assertEqual(hp.word2struc("hortobaːɟ"), "CVCCVCVC")

    def test_ipa2clusters(self):
        self.assertEqual(hp.ipa2clusters("abauːjkeːr"),
                         ['a', 'b', 'auː', 'jk', 'eː', 'r'])

    def test_harmony(self):
        self.assertEqual(hp.harmony("bot͡sibot͡si"), False)
        self.assertEqual(hp.harmony("tɒrkɒ"), True)
        self.assertEqual(hp.harmony("ʃɛfylɛʃɛ"), True)

    def test_adaptharmony(self):
        self.assertEqual(hp.adaptharmony('kɛsthɛj'),
                         ['k', 'ɛ', 's', 't', 'h', 'ɛ', 'j'])
        self.assertEqual(hp.adaptharmony('ɒlʃoːørʃ'),
                         ['ɒ', 'l', 'ʃ', 'oː', 'B', 'r', 'ʃ'])
        self.assertEqual(hp.adaptharmony('ʃioːfok'),
                         ['ʃ', 'B', 'oː', 'f', 'o', 'k'])
        self.assertEqual(hp.adaptharmony('bɒlɒtonkɛnɛʃɛ'),
                         ['b', 'ɒ', 'l', 'ɒ', 't', 'o', 'n',
                          'k', 'B', 'n', 'B', 'ʃ', 'B'])

if __name__ == "__main__":
    unittest.main()
