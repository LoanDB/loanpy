import os
import unittest

import pandas as pd
from pandas._testing import assert_frame_equal

from loanpy import reconstructor as rc

os.chdir(os.path.join(
         os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))


class Test1(unittest.TestCase):

    def test_launch(self):
        soundchangedict_test = {'#0': ["j", "m"], '#a': ["ɑ"], '0#': ["ɑ"],
                                'ʃ#': [""]}
        with open("soundchangedict_test.txt", "w", encoding="utf-8") as data:
            data.write(str(soundchangedict_test))
        dfetymology_test = pd.DataFrame({"New": ["mɛʃɛ", "aːɟ", "ɒl"],
                                         "Old": ["ɑt͡ʃʲɑ", "ɑðʲͽ", "ɑlɑ"],
                                         "Lan": ["FP", "FU", "U"]})
        dfetymology_test.to_csv("dfetymology_test.csv", encoding="utf-8",
                                index=False)
        substi_test = pd.DataFrame({"L2_phons": ["a", "b", "c", "d"],
                                    "L1_substi": ["o", "p", "t", "t"]})
        substi_test.to_csv("substi_test.csv", encoding="utf-8", index=False)
        nsedict_test = {'#0<*0': 433, '#0<*j': 6}
        with open("nsedict_test.txt", "w", encoding="utf-8") as data:
            data.write(str(nsedict_test))

        rc.launch(soundchangedict="soundchangedict_test.txt",
                  dfetymology="dfetymology_test.csv",
                  timelayer="",
                  se_or_edict="nsedict_test.txt")

        self.assertDictEqual(rc.scdict, soundchangedict_test)
        self.assertEqual(rc.allowedphonotactics, {"VCV"})
        self.assertEqual(rc.nsedict, nsedict_test)

        os.remove("dfetymology_test.csv")
        os.remove("substi_test.csv")
        os.remove("soundchangedict_test.txt")
        os.remove("nsedict_test.txt")

    def test_getsoundchanges(self):
        expected_df = pd.DataFrame({"reflex":
                                   ['#0', '#ɟ', 'ɒ', 'l', 'o', 'ɡ#'],
                                    "root": ['0', 'j', 'ɑ', 'lk', 'ɑ', '0']})
        assert_frame_equal(rc.getsoundchanges("ɟɒloɡ", "jɑlkɑ"),
                           expected_df, check_dtype=False)

        expected_df = pd.DataFrame({"reflex": ['#m', 'ɛ', 'ʃ', 'ɛ#', '0#'],
                                    "root": ['0', 'ɑ', 't͡ʃʲ', 'ɑ', '0']})
        assert_frame_equal(rc.getsoundchanges("mɛʃɛ", "ɑt͡ʃʲɑ"),
                           expected_df, check_dtype=False)

        expected_df = pd.DataFrame({"reflex": ["#0", "#ɒ", "l#", "0#"],
                                    "root": ['0', 'ɑ', 'l', 'ɑ']})
        assert_frame_equal(rc.getsoundchanges("ɒl", "ɑlɑ"),
                           expected_df, check_dtype=False)

    def test_dfetymology2dict(self):
        data = {"New": ["mɛʃɛ", "aːɟ", "ɒl"], "Old": ["ɑt͡ʃʲɑ", "ɑðʲͽ", "ɑlɑ"]}
        pd.DataFrame(data).to_csv("dfural_test.csv", encoding="utf-8",
                                  index=False)
        out = rc.dfetymology2dict(dfetymology="dfural_test.csv",
                                  timelayer="",
                                  name_soundchangedict="scdict_test",
                                  name_sumofexamplesdict="sedict_test",
                                  name_listofexamplesdict="edict_test")
        expectedout = (({'#0': ['0'], '#aː': ['ɑ'], '#m': ['0'], '#ɒ': ['ɑ'],
                         '0#': ['0', 'ͽ', 'ɑ'], 'l#': ['l'], 'ɛ': ['ɑ'],
                         'ɛ#': ['ɑ'], 'ɟ#': ['ðʲ'], 'ʃ': ['t͡ʃʲ']},
                        {'#0<*0': 2, '#aː<*ɑ': 1, '#m<*0': 1, '#ɒ<*ɑ': 1,
                         '0#<*0': 1, '0#<*ɑ': 1, '0#<*ͽ': 1, 'l#<*l': 1,
                         'ɛ#<*ɑ': 1, 'ɛ<*ɑ': 1, 'ɟ#<*ðʲ': 1, 'ʃ<*t͡ʃʲ': 1},
                        {'#0<*0': ['aːɟ<*ɑðʲͽ', 'ɒl<*ɑlɑ'],
                         '#aː<*ɑ': ['aːɟ<*ɑðʲͽ'], '#m<*0': ['mɛʃɛ<*ɑt͡ʃʲɑ'],
                         '#ɒ<*ɑ': ['ɒl<*ɑlɑ'], '0#<*0': ['mɛʃɛ<*ɑt͡ʃʲɑ'],
                         '0#<*ɑ': ['ɒl<*ɑlɑ'], '0#<*ͽ': ['aːɟ<*ɑðʲͽ'],
                         'l#<*l': ['ɒl<*ɑlɑ'], 'ɛ#<*ɑ': ['mɛʃɛ<*ɑt͡ʃʲɑ'],
                         'ɛ<*ɑ': ['mɛʃɛ<*ɑt͡ʃʲɑ'], 'ɟ#<*ðʲ': ['aːɟ<*ɑðʲͽ'],
                         'ʃ<*t͡ʃʲ': ['mɛʃɛ<*ɑt͡ʃʲɑ']}))
        self.assertEqual(out, expectedout)

    def test_getnse(self):
        rc.launch(se_or_edict="edict.txt", soundchangedict="scdict.txt",
                  dfetymology="dfuralonet.csv")
        out1 = rc.getnse('iv', 'juɣe', examples=True, normalise=False)
        rc.launch(se_or_edict="sedict.txt", soundchangedict="scdict.txt")
        out2 = rc.getnse('iv', 'juɣe', examples=False, normalise=False)
        out3 = rc.getnse('iv', 'juɣe', examples=False, normalise=True)
        out4 = rc.getnse('iv', 'juɣe', examples=True, normalise=True)
        expectedout1 = [['iːz<*jȣ̈tͽ', 'iːj<*joŋsͽ', 'iv<*juɣe', 'iktɒt<*jɑkkɑ',
                         'idɛɡ<*jænte', 'eːv<*jikæ'],
                        ['irt<*ʃurͽ', 'iv<*juɣe'],
                        ['viv<*wiɣe', 'iv<*juɣe'],
                        ['eːj<*eje', 'ɒj<*ɑŋe', 'uːs<*uje', 'iːɲ<*ike',
                         'feːl<*pele', 'feːl<*pele', 'fɛj<*pæŋe',
                         'hɒt<*kutte', 'hɒlː<*kule', 'huːɟ<*kunʲt͡ʃʲe',
                         'haːj<*kuje', 'foɡ<*piŋe', 'vɒʃ<*wɑʃʲke', 'mɛn<*mene',
                         'kɛlː<*kelke', 'meːz<*mete', 'veːr<*wire', 'ɲɛl<*nʲele',
                         'keːz<*kæte', 'tɛv<*teke', 'vol<*wole', 'keːj<*keje',
                         'moʃ<*muʃʲke', 'meːh<*mekʃe', 'ɲɒl<*nʲole', 'tɒt<*totke',
                         'vɒj<*woje', 'tud<*tumte', 'køt<*kitke', 'køɲː<*kinʲe',
                         'viːz<*wete', 'jɛl<*jælke', 'ɲiːl<*nʲɤle', 'jeːɡ<*jæŋe',
                         'viv<*wiɣe', 'viːv<*woje', 'loːɡ<*loŋe', 'saːj<*ʃʲuwe',
                         'ɲust<*nʲukʃʲe', 'zɒj<*ʃʲoje', 'neːz<*næke', 'ʃeːrt<*t͡ʃʲærke',
                         'duɡ<*tuŋke', 'ɲeːl<*niðe', 'neːv<*nime', 'øl<*sile',
                         'eːɡ<*sæŋe', 'ɛlː<*sæle', 'eːv<*sæje', 'iːn<*sɤne',
                         'ɛv<*sewe', 'ɛɡeːr<*ʃiŋere', 'øːs<*sikʃʲe', 'øt<*witte',
                         'iv<*juɣe', 'uːj<*wuðʲe']]
        self.assertEqual(out1, expectedout1)
        self.assertEqual(out2, 66)
        self.assertEqual(out3, 16.5)
        self.assertEqual(out4, [6, 2, 2, 56])

    def test_reconstruct(self):
        rc.launch(soundchangedict="scdict.txt", se_or_edict="sedict.txt",
                  dfetymology="dfuralonet.csv")
        out1 = rc.reconstruct('iv', howmany=1, struc=False,
                              vowelharmony=False, sort_by_nse=True)
        out2 = rc.reconstruct('iv', howmany=3, struc=False,
                              vowelharmony=False, sort_by_nse=True)
        out3 = rc.reconstruct('iv', howmany=5, struc=True,
                              vowelharmony=False, sort_by_nse=True)
        out5 = rc.reconstruct('iv', howmany=5, struc=True,
                              vowelharmony=True, sort_by_nse=True)

        self.assertEqual(out1, '^(ɑ)(ŋ)(ͽ)$')
        self.assertEqual(out2, '^(ɑ|u)(ŋ|k)(ͽ)$')
        self.assertEqual(out3, '^ɑɣͽ$|^ɑŋͽ$|^ɑkͽ$|^uɣͽ$|^uŋͽ$|^ukͽ$')
        self.assertEqual(out5, '^ɑɣͽ$|^ɑŋͽ$|^ɑkͽ$|^uɣͽ$|^uŋͽ$|^ukͽ$')


if __name__ == '__main__':
    unittest.main()
