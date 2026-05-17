"""Tests for loanpy.cluster."""

import pytest

from loanpy import Cluster


class TestClusterCv:
    def test_consonant_run(self):
        assert Cluster.cv(["f", "l", "a"], ["C", "C", "V"]) == ["f.l", "a"]

    def test_vowel_run(self):
        assert Cluster.cv(["a", "ʊ", "ə"], ["V", "V", "V"]) == ["a.ʊ.ə"]

    def test_single_segments(self):
        assert Cluster.cv(["k", "a"], ["C", "V"]) == ["k", "a"]


class TestClusterGlides:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Cluster.glides(["a"], ["C", "V"])

    def test_glide_between_vowels(self):
        segments = ["a", "w", "a"]
        cv = ["V", "C", "V"]
        out = Cluster.glides(segments, cv, cluster_between_vowels=("w",))
        assert len(out) <= len(segments)
        assert "w" in out[0] or any("w" in t for t in out)


class TestClusterGaps:
    def test_collapses_adjacent_gaps(self):
        seq_a = ["x", "y", "z"]
        seq_b = ["a", "-", "-"]
        a2, b2 = Cluster.gaps(seq_a, seq_b)
        assert "y.z" in ".".join(a2) or a2[-1] == "y.z"
        assert b2[-1] != "-"

    def test_trailing_gap_marker(self):
        seq_a = ["a", "b", "c"]
        seq_b = ["x", "y", "-"]
        a2, b2 = Cluster.gaps(seq_a, seq_b)
        assert "+" in a2 or b2[-1] != "-"
