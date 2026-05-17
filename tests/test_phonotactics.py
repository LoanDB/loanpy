"""Tests for loanpy.phonotactics."""

import pytest

from loanpy import expand_phonotactics, get_closest_phonotactics
from loanpy.phonotactics import _expand_syllable_template


class TestExpandSyllableTemplate:
    def test_optional_c(self):
        variants = _expand_syllable_template("(C)V")
        assert ["V"] in variants
        assert ["C", "V"] in variants

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _expand_syllable_template("X")


class TestExpandPhonotactics:
    def test_two_syllables(self):
        words = expand_phonotactics("(C)V+CV")
        assert "V C V" in words
        assert "C V C V" in words

    def test_three_slot_formula(self):
        words = expand_phonotactics("CV+CV")
        assert words == ["C V C V"]


class TestGetClosestPhonotactics:
    def test_prefers_extra_consonant_over_vowel(self):
        inventory = ["V C V", "C V C V", "C V C C V"]
        closest = get_closest_phonotactics(list("CVCV"), inventory)
        assert closest in {"CVCV", "CVCCV"}

    def test_exact_match(self):
        inventory = ["C V"]
        assert get_closest_phonotactics(["C", "V"], inventory) == "CV"
