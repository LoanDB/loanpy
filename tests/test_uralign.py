"""Tests for loanpy.uralign."""

from loanpy import Uralign


class TestUralignHu:
    def test_initial_vowel_gap(self):
        hu = ["a", "l", "m"]
        pu = ["a", "l"]
        cv_h, cv_p = "V", "V"
        hu_copy, pu_copy = hu.copy(), pu.copy()
        out_h, out_p = Uralign.hu(hu_copy, pu_copy, cv_h, cv_p)
        assert out_h[0] == "#-"
        assert out_p[0] == "-"

    def test_final_gap_shorter_descendant(self):
        hu = ["k", "a"]
        pu = ["k", "a", "t", "a"]
        out_h, out_p = Uralign.hu(hu.copy(), pu.copy(), "C", "C")
        assert out_h[-1] == "-#"
        assert "." in out_p[-1] or len(out_p) <= len(pu)

    def test_no_final_gap_truncates(self):
        hu = ["a", "b", "c"]
        pu = ["x", "y"]
        out_h, out_p = Uralign.hu(
            hu.copy(), pu.copy(), "V", "C", initial_gap=False, final_gap=False
        )
        assert len(out_h) == len(out_p) == 2


class TestUralignGetScore:
    def test_positive_pairs(self):
        scorer = {"a < x": 5, "b < y": 3}
        assert Uralign.get_score(["a", "b"], ["x", "y"], scorer, freq_filter=2) == 8

    def test_penalty_below_threshold(self):
        scorer = {"a < x": 1}
        score = Uralign.get_score(["a"], ["x"], scorer, freq_filter=2)
        assert score < 0
