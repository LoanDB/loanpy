import unittest
from loanpy import helpers as hp
from unittest.mock import patch

class Test1(unittest.TestCase):
    def test_flatten(self):
        self.assertEqual(hp.flatten([["Bu","da"],["pest"]]), ["Bu","da","pest"])
        
    def test_list2regex(self):
        self.assertEqual(hp.list2regex(["a","b","c"]), "(a|b|c)")
        self.assertEqual(hp.list2regex(["a","b","c", "0"]), "(a|b|c)?")
        self.assertEqual(hp.list2regex(["0"]), "")
        
    def test_word2struc(self):
        hp.cns="jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC"
        hp.vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅"
        with patch("loanpy.helpers.ipa2tokens") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ["h", "o", "r", "t", "o", "b", "a", "ɟ"]
            self.assertEqual(hp.word2struc("hortobaɟ"), "CVCCVCVC")
        
    def test_ipa2clusters(self):
        hp.vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅"
        with patch("loanpy.helpers.ipa2tokens") as ipa2tokens_mock:
            ipa2tokens_mock.return_value = ["ɒ","j","k","ɒ"]
            assert hp.ipa2clusters("ɒjkɒ") == ["ɒ","jk","ɒ"]

if __name__ == "__main__":
    unittest.main()