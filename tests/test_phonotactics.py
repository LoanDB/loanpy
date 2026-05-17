"""Tests for loanpy.phonotactics."""

import pytest

from loanpy import expand_phonotactics, get_closest_phonotactics
from loanpy.phonotactics import _expand_syllable_template


class TestExpandSyllableTemplate:
    def test_cv_only(self):
        assert _expand_syllable_template("CV") == [["C", "V"]]

    def test_optional_initial_consonant(self):
        variants = _expand_syllable_template("(C)V")
        assert ["V"] in variants
        assert ["C", "V"] in variants
        assert len(variants) == 2

    def test_cvc_all_four_combinations(self):
        variants = _expand_syllable_template("(C)V(C)")
        assert ["V"] in variants
        assert ["C", "V", "C"] in variants
        assert len(variants) == 4

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="invalid syllable template"):
            _expand_syllable_template("XV")


class TestExpandPhonotactics:
    def test_single_syllable_cv(self):
        assert expand_phonotactics("CV") == ["C V"]

    def test_two_syllable_formula(self):
        words = expand_phonotactics("(C)V+CV")
        assert "V C V" in words
        assert "C V C V" in words
        assert len(words) == 2  # (C)V → 2 variants × CV → 1

    def test_three_syllable_fixed(self):
        assert expand_phonotactics("CV+CV+CV") == ["C V C V C V"]

    def test_whitespace_stripped_around_plus(self):
        assert expand_phonotactics(" CV + CV ") == ["C V C V"]


class TestGetClosestPhonotactics:
    def test_exact_template_match(self):
        assert get_closest_phonotactics(["C", "V"], ["C V", "C V C V"]) == "CV"

    def test_prefers_consonant_slot_over_vowel_insertion(self):
        inventory = ["V C V", "C V C V", "C V C C V"]
        assert get_closest_phonotactics(list("CVCV"), inventory) in {"CVCV", "CVCCV"}

    def test_longer_profile_needs_extra_slots(self):
        inventory = ["C V", "C V C V", "C V C C V"]
        closest = get_closest_phonotactics(list("CVCVC"), inventory)
        assert closest == "CVCV"

    def test_inventory_with_spaces_stripped(self):
        assert get_closest_phonotactics(["V"], ["C V", "V"]) == "V"
