"""Tests for loanpy.adapt."""

from loanpy import Adapt


def _hamming(a: str, b: str) -> float:
    if len(a) != len(b):
        return float(len(a) + len(b))
    return sum(x != y for x, y in zip(a, b))


class TestAdaptSubstitutions:
    def test_learns_closest_recipient_phoneme(self):
        ad = Adapt()
        ad.get_substitutions({"x"}, {"a", "b"}, lambda d, r: _hamming(d, r), {})
        assert ad.substitutions["x"] in {"a", "b"}

    def test_tie_breaker_first_minimum_in_iteration_order(self):
        ad = Adapt()
        ad.get_substitutions({"x"}, {"a", "b"}, lambda d, r: 0.0, {})
        assert ad.substitutions["x"] in {"a", "b"}

    def test_extra_overrides_learned_mapping(self):
        ad = Adapt()
        ad.get_substitutions({"x"}, {"a", "b"}, lambda d, r: _hamming(d, r), {"x": "b"})
        assert ad.substitutions["x"] == "b"

    def test_donor_in_recipient_inventory_not_mapped(self):
        ad = Adapt()
        ad.get_substitutions({"a", "b"}, {"a", "b"}, lambda d, r: 0, {})
        assert "a" not in ad.substitutions
        assert "b" not in ad.substitutions


class TestAdaptSubstitute:
    def test_identity_for_unmapped_segments(self):
        ad = Adapt()
        ad.substitutions = {"p": "b"}
        assert ad.substitute(["p", "a", "t"]) == ["b", "a", "t"]

    def test_empty_substitution_skips_segment(self):
        ad = Adapt()
        ad.substitutions = {"ʔ": ""}
        assert ad.substitute(["k", "ʔ", "a"]) == ["k", "a"]

    def test_empty_input(self):
        ad = Adapt()
        ad.substitutions = {}
        assert ad.substitute([]) == []


class TestAdaptRepair:
    def test_repair_inserts_placeholder_segments(self):
        ad = Adapt()
        inventory = ["C V C V", "C V C C V"]
        out = ad.repair(["k", "a", "t"], ["C", "V", "C"], inventory)
        assert isinstance(out, list)
        assert len(out) >= 2

    def test_extra_repair_bypasses_inventory_search(self):
        ad = Adapt()
        out = ad.repair(
            ["a", "b"],
            ["C", "V"],
            ["V C V C V"],
            extra_repair={"CV": "CVCV"},
        )
        assert len(out) >= 2

    def test_repair_cvvcv_override(self):
        ad = Adapt()
        out = ad.repair(
            ["x"],
            ["V"],
            ["C V C V"],
            extra_repair={"V": "CVCVCV"},
        )
        assert isinstance(out, list)

    def test_full_pipeline_substitute_then_repair(self):
        ad = Adapt()
        ad.get_substitutions(
            {"θ", "a"},
            {"t", "a", "o"},
            lambda d, r: 0 if d == r else 1,
            {"θ": "t"},
        )
        adapted = ad.substitute(["θ", "a"])
        assert adapted == ["t", "a"]
        repaired = ad.repair(adapted, ["C", "V"], ["C V", "C V C V"])
        assert len(repaired) >= 1
