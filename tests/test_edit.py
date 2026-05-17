"""Tests for loanpy.edit."""

import pytest

from loanpy.edit import (
    apply_edit,
    edit_distance_matrix,
    edit_distance_with2ops,
    path_to_edit_operations,
    shortest_edit_path,
    substitute_operations,
)


class TestEditDistanceWith2ops:
    def test_identical(self):
        assert edit_distance_with2ops("abc", "abc") == 0

    def test_insertion_weight(self):
        assert edit_distance_with2ops("ab", "abc", w_ins=10) == 10

    def test_deletion_weight(self):
        assert edit_distance_with2ops("abc", "ab", w_del=5) == 5

    def test_asymmetric_weights(self):
        d1 = edit_distance_with2ops("rajka", "ajka", w_del=100, w_ins=49)
        d2 = edit_distance_with2ops("ajka", "rajka", w_del=100, w_ins=49)
        assert d1 == 100
        assert d2 == 49


class TestSubstituteOperations:
    def test_merge_delete_insert(self):
        ops = ["delete a", "insert b", "keep c"]
        assert substitute_operations(ops) == ["substitute a by b", "keep c"]

    def test_merge_insert_delete(self):
        ops = ["insert x", "delete y"]
        assert substitute_operations(ops) == ["substitute y by x"]


class TestEditDistanceMatrix:
    def test_shape_and_corners(self):
        mtx = edit_distance_matrix("ab", "acb")
        assert len(mtx) == 4  # # + len(source)
        assert len(mtx[0]) == 3  # # + len(target)
        assert mtx[0][0] == 0


class TestShortestEditPath:
    def test_returns_path_to_corner(self):
        mtx = edit_distance_matrix("CV", "CVC")
        path = shortest_edit_path(mtx)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (len(mtx) - 1, len(mtx[0]) - 1)


class TestPathToEditOperations:
    def test_produces_apply_edit_compatible_ops(self):
        mtx = edit_distance_matrix("CV", "CVC")
        path = shortest_edit_path(mtx)
        ops = path_to_edit_operations(path, "CV", "CVC")
        assert any("insert" in o or "keep" in o or "delete" in o for o in ops)


class TestApplyEdit:
    def test_keep_and_insert(self):
        word = list("ab")
        ops = ["keep a", "keep b", "insert c"]
        result = apply_edit(word, ops)
        assert "a" in result and "b" in result

    def test_substitute(self):
        word = list("x")
        ops = ["substitute x by y"]
        assert apply_edit(word, ops) == ["y"]
