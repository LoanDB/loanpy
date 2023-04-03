# -*- coding: utf-8 -*-
from loanpy.scminer import get_correspondences

def test_get_correspondences1():
    input_table = [    ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
    ['0', '1', 'H', '#aː t͡ʃ# -#', 'VC'],
    ['1', '1', 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV']
]

    expected_output = [
    {'#aː': ['a.ɣ.a'], 't͡ʃ#': ['t͡ʃ'], '-#': ['i']},
    {'#aː a.ɣ.a': 1, 't͡ʃ# t͡ʃ': 1, '-# i': 1},
    {'#aː a.ɣ.a': [1], 't͡ʃ# t͡ʃ': [1], '-# i': [1]},
    {'VC': ['VCVCV']},
    {'VC VCVCV': 1},
    {'VC VCVCV': [1]}]
    assert get_correspondences(input_table) == expected_output

def test_get_correspondences2():
    input_table = [    ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
    ['0', '1', 'H', '#aː t͡ʃ# -#', 'VC'],
    ['1', '1', 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV'],
    ['2', '2', 'H', '#aː ɟ uː#', 'VCV'],
    ['3', '2', 'EAH', 'a l.d a.ɣ', 'VCCVC'],
    ['4', '3', 'H',	'#ɒ j n', 'VCC'],
    ['5', '3', 'EAH',	'a j.a n', 'VCVC']

]

    expected_output = [
    {'#aː': ['a.ɣ.a', 'a'], 't͡ʃ#': ['t͡ʃ'], '-#': ['i'],
        'ɟ': ['l.d'], 'uː#': ['a.ɣ'], '#ɒ': ['a'], 'j': ['j.a'], 'n': ['n']
        },
    {'#aː a.ɣ.a': 1, 't͡ʃ# t͡ʃ': 1, '-# i': 1, '#aː a': 1, 'ɟ l.d': 1,
        'uː# a.ɣ': 1, '#ɒ a': 1, 'j j.a': 1, 'n n': 1
        },
    {'#aː a.ɣ.a': [1], 't͡ʃ# t͡ʃ': [1], '-# i': [1], '#aː a': [2],
        'ɟ l.d': [2], 'uː# a.ɣ': [2], '#ɒ a': [3], 'j j.a': [3], 'n n': [3]
        },
    {'VC': ['VCVCV'], 'VCV': ['VCCVC'], 'VCC': ['VCVC']
    },
    {'VC VCVCV': 1, 'VCV VCCVC': 1, 'VCC VCVC': 1
    },
    {'VC VCVCV': [1], 'VCV VCCVC': [2], 'VCC VCVC': [3]
    }
    ]
    assert get_correspondences(input_table) == expected_output
