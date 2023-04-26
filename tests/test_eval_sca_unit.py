# -*- coding: utf-8 -*-
import pytest
from loanpy.eval_sca import eval_all, eval_one
from unittest.mock import call, patch, MagicMock

@patch('loanpy.eval_sca.eval_one', side_effect=[0.4, 0.5, 0.6])
def test_evaluate_all_returns_expected_output(mock_eval_one):

    fp_vs_tp = eval_all(
        intable="x", heur="y", adapt=True, guess_list=[1, 2, 3]
        )
    assert fp_vs_tp == [(0.33, 0.4), (0.67, 0.5), (1.0, 0.6)]
    assert mock_eval_one.call_args_list == [call('x', 'y', True, 1, False),
        call('x', 'y', True, 2, False), call('x', 'y', True, 3, False)]

# Mock the external dependencies
class AdrcMonkey:
    def __init__(self, *args):
        self.init_called_with = [*args]
        self.sc = []
        self.prosodic_inventory = []
        self.adapt_returns = iter(["tip", "sip"])
        self.adapt_called_with = []
        self.reconstruct_returns = iter(["tip", "sip"])
        self.reconstruct_called_with = []
    def adapt(self, *args):
        self.adapt_called_with.append([*args])
        return next(self.adapt_returns)
    def reconstruct(self, *args):
        self.reconstruct_called_with.append([*args])
        return next(self.reconstruct_returns)
    def set_sc(self, sc):
        self.sc = sc
    def set_prosodic_inventory(self, inv):
        self.prosodic_inventory = inv

adrc_monkey = AdrcMonkey()
@patch("loanpy.eval_sca.Adrc", side_effect=[adrc_monkey, adrc_monkey])
@patch("loanpy.eval_sca.get_correspondences")
@patch("loanpy.eval_sca.get_prosodic_inventory")
def test_eval_one_adapt(get_prosodic_inventory_mock, get_correspondences_mock, adrc_mock):
    # define patched functions' return value
    get_prosodic_inventory_mock.return_value = 321
    get_correspondences_mock.return_value = 123
    # define input variables
    intable = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
  ['0', '1', 'H', '#aː t͡ʃ# -#', 'VC'],
  ['1', '1', 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV'],
  ['2', '2', 'H', '#aː ɟ uː#', 'VCV'],
  ['3', '2', 'EAH', 'a l.d a.ɣ', 'VCCVC']
]
    heuristic = "some_heuristic"
    adapt = True
    howmany = 2
    pros = True

    # assert results
    result = eval_one(intable, heuristic, adapt, howmany, pros)
    assert result == 0.0

    # assert calls
    assert adrc_monkey.adapt_called_with == [['#aː t͡ʃ# -#', 2, "VC"],
                                            ['#aː ɟ uː#', 2, "VCV"]
                                            ]
    get_correspondences_mock.assert_called_with(intable, heuristic)
    get_prosodic_inventory_mock.assert_called_with(intable)
    assert not adrc_monkey.init_called_with

adrc_monkey2 = AdrcMonkey()
@patch("loanpy.eval_sca.Adrc", side_effect=[adrc_monkey2, adrc_monkey2])
@patch("loanpy.eval_sca.get_correspondences")
@patch("loanpy.eval_sca.get_prosodic_inventory")
@patch("loanpy.eval_sca.re.match")
def test_eval_one_reconstruct(match_mock, get_prosodic_inventory_mock, get_correspondences_mock, adrc_mock):
    match_mock.return_value = 111
    get_prosodic_inventory_mock.return_value = 321
    get_correspondences_mock.return_value = 123
    adrc_mock.return_value = AdrcMonkey()
    intable = [  ['ID', 'COGID', 'DOCULECT', 'ALIGNMENT', 'PROSODY'],
  ['0', '1', 'H', '#aː t͡ʃ# -#', 'VC'],
  ['1', '1', 'EAH', 'a.ɣ.a t͡ʃ i', 'VCVCV'],
  ['2', '2', 'H', '#aː ɟ uː#', 'VCV'],
  ['3', '2', 'EAH', 'a l.d a.ɣ', 'VCCVC']
]
    heuristic = "some_heuristic"
    adapt = False
    num_reconstructions = 2
    additional_args = ()

    result = eval_one(intable, heuristic, adapt, num_reconstructions, *additional_args)
    assert result == 1.0
