"""Tests for package metadata and public API."""

import loanpy


def test_version():
    assert loanpy.__version__ == "4.0.0"


def test_public_exports():
    expected = {
        "Adapt",
        "Cluster",
        "Uralign",
        "apply_edit",
        "edit_distance_matrix",
        "edit_distance_with2ops",
        "expand_phonotactics",
        "get_closest_phonotactics",
        "get_sound_correspondences",
        "path_to_edit_operations",
        "shortest_edit_path",
        "substitute_operations",
    }
    assert expected <= set(loanpy.__all__)
