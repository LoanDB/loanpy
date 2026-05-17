"""Tests for loanpy.adapt."""

from loanpy import Adapt


def _hamming(a: str, b: str) -> float:
    if len(a) != len(b):
        return float(len(a) + len(b))
    return sum(x != y for x, y in zip(a, b))


class TestAdaptSubstitutions:
    def test_learns_closest_recipient(self):
        ad = Adapt()
        ad.get_substitutions({"x"}, {"a", "b"}, lambda d, r: _hamming(d, r), {})
        assert ad.substitutions["x"] in {"a", "b"}

    def test_extra_overrides(self):
        ad = Adapt()
        ad.get_substitutions(set(), {"a"}, lambda d, r: 0, {"z": "a"})
        assert ad.substitutions["z"] == "a"


class TestAdaptSubstitute:
    def test_identity_and_mapping(self):
        ad = Adapt()
        ad.substitutions = {"p": "b"}
        assert ad.substitute(["p", "a"]) == ["b", "a"]


class TestAdaptRepair:
    def test_repair_returns_list(self):
        ad = Adapt()
        inventory = ["C V", "C V C V"]
        out = ad.repair(["k", "a"], ["C", "V"], inventory)
        assert isinstance(out, list)
        assert len(out) >= 1

    def test_extra_repair_bypass(self):
        ad = Adapt()
        out = ad.repair(
            ["a"],
            ["V"],
            ["C V C V"],
            extra_repair={"V": "CV"},
        )
        assert isinstance(out, list)
