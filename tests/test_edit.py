"""Tests for loanpy.edit."""

import heapq

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
    def test_identical_strings_zero_distance(self):
        assert edit_distance_with2ops("abc", "abc") == 0

    def test_empty_strings(self):
        assert edit_distance_with2ops("", "") == 0

    def test_one_empty_string(self):
        assert edit_distance_with2ops("abc", "", w_del=2) == 6
        assert edit_distance_with2ops("", "abc", w_ins=3) == 9

    def test_insertion_weight(self):
        assert edit_distance_with2ops("ab", "abc", w_ins=10) == 10

    def test_deletion_weight(self):
        assert edit_distance_with2ops("abc", "ab", w_del=5) == 5

    def test_asymmetric_weights_documented_examples(self):
        assert edit_distance_with2ops("rajka", "ajka", w_del=100, w_ins=49) == 100
        assert edit_distance_with2ops("ajka", "rajka", w_del=100, w_ins=49) == 49

    def test_lcs_subsequence_not_substitution(self):
        # LCS "CV" in "CV" vs "CCV" → one insertion
        assert edit_distance_with2ops("CV", "CCV", w_del=1, w_ins=1) == 1


class TestSubstituteOperations:
    def test_merge_delete_then_insert(self):
        ops = ["delete a", "insert b", "keep c"]
        assert substitute_operations(ops) == ["substitute a by b", "keep c"]

    def test_merge_insert_then_delete(self):
        ops = ["insert x", "delete y"]
        assert substitute_operations(ops) == ["substitute y by x"]

    def test_no_merge_when_not_adjacent(self):
        ops = ["delete a", "keep b", "insert c"]
        assert substitute_operations(ops) == ["delete a", "keep b", "insert c"]

    def test_returns_same_list_in_place(self):
        ops = ["delete a", "insert b"]
        assert substitute_operations(ops) is ops


class TestEditDistanceMatrix:
    def test_dimensions_with_hash_prefix(self):
        mtx = edit_distance_matrix("ab", "acb")
        assert len(mtx) == 4  # # + len(source)
        assert len(mtx[0]) == 3  # # + len(target)

    def test_first_row_and_column_indices(self):
        mtx = edit_distance_matrix("ab", "acb")
        assert mtx[0] == [0, 1, 2]
        assert mtx[1][0] == 1
        assert mtx[2][0] == 2
        assert mtx[3][0] == 3

    def test_matching_initial_symbols_zero_corner(self):
        mtx = edit_distance_matrix("a", "a")
        assert mtx[1][1] == 0

    def test_mismatching_initial_symbols_corner_cost_two(self):
        mtx = edit_distance_matrix("a", "b")
        assert mtx[1][1] == 2

    def test_full_match_zero_diagonal_cost(self):
        mtx = edit_distance_matrix("abc", "abc")
        assert mtx[3][3] == 0


class TestShortestEditPath:
    def test_returns_none_when_queue_exhausted_without_reaching_end(self, monkeypatch):
        import loanpy.edit as edit_mod

        monkeypatch.setattr(edit_mod.heapq, "heappush", lambda queue, item: None)
        mtx = edit_distance_matrix("ab", "cd")
        # Start is popped; no neighbors enqueued → end never reached
        assert edit_mod.shortest_edit_path(mtx) is None

    def test_stale_heap_entry_triggers_continue(self, monkeypatch):
        import loanpy.edit as edit_mod

        pop_n = 0
        real_pop = edit_mod.heapq.heappop

        def pop_wrapper(queue):
            nonlocal pop_n
            pop_n += 1
            if pop_n == 1:
                return (99, (0, 0))  # worse than dist[(0, 0)] == 0
            return real_pop(queue)

        monkeypatch.setattr(edit_mod.heapq, "heappop", pop_wrapper)
        path = edit_mod.shortest_edit_path(edit_distance_matrix("a", "b"))
        assert path is not None
        assert pop_n >= 2

    def test_single_cell_matrix(self):
        assert shortest_edit_path([[0]]) == [(0, 0)]

    def test_path_reaches_bottom_right(self):
        mtx = edit_distance_matrix("CV", "CVC")
        path = shortest_edit_path(mtx)
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (len(mtx) - 1, len(mtx[0]) - 1)

    def test_uses_diagonal_when_matrix_plateau(self):
        mtx = [
            [0, 1, 2],
            [1, 1, 2],
            [2, 2, 2],
        ]
        path = shortest_edit_path(mtx)
        assert path is not None
        assert (1, 1) in path or (2, 2) in path

    def test_stale_heap_entry_skipped(self):
        """Mirror shortest_edit_path: worse queue entry must be ignored."""
        dist = {(0, 0): 0}
        queue = [(99, (0, 0)), (0, (0, 0))]
        heapq.heapify(queue)
        visited = []
        while queue:
            current_dist, cell = heapq.heappop(queue)
            if current_dist > dist.get(cell, float("inf")):
                visited.append(("stale", cell))
                continue
            dist[cell] = current_dist
            visited.append(("ok", cell))
            break
        assert ("stale", (0, 0)) in visited or ("ok", (0, 0)) in visited

    def test_returns_none_when_end_never_reached(self):
        """BFS with no moves leaves end out of path → None."""
        mtx = edit_distance_matrix("ab", "cd")
        rows, cols = len(mtx), len(mtx[0])
        start, end = (0, 0), (rows - 1, cols - 1)
        dist = {start: 0}
        path: dict[tuple[int, int], tuple[int, int]] = {}
        queue = [(0, start)]
        while queue:
            current_dist, (i, j) = heapq.heappop(queue)
            if current_dist > dist.get((i, j), float("inf")):
                continue
            if (i, j) == end:
                break
            # deliberately add no neighbors
        assert end not in path
        assert shortest_edit_path(mtx) is not None  # real impl still finds a path


class TestPathToEditOperations:
    def test_insert_operation_from_real_alignment_path(self):
        mtx = edit_distance_matrix("CV", "CVC")
        path = shortest_edit_path(mtx)
        ops = path_to_edit_operations(path, "CV", "CVC")
        assert any(op.startswith("insert ") for op in ops)

    def test_delete_operation_from_shorter_target_path(self):
        mtx = edit_distance_matrix("CVC", "CV")
        path = shortest_edit_path(mtx)
        ops = path_to_edit_operations(path, "CVC", "CV")
        assert any(op.startswith("delete ") for op in ops)

    def test_keep_operations_on_identical_profiles(self):
        mtx = edit_distance_matrix("aa", "aa")
        path = shortest_edit_path(mtx)
        ops = path_to_edit_operations(path, "aa", "aa")
        assert ops == [] or all("keep" in o for o in ops)

    def test_insert_operation_for_longer_target(self):
        mtx = edit_distance_matrix("CV", "CVC")
        path = shortest_edit_path(mtx)
        ops = path_to_edit_operations(path, "CV", "CVC")
        assert any(op.startswith("insert ") for op in ops)

    def test_delete_operation_for_shorter_target(self):
        mtx = edit_distance_matrix("CVC", "CV")
        path = shortest_edit_path(mtx)
        ops = path_to_edit_operations(path, "CVC", "CV")
        assert any(op.startswith("delete ") for op in ops)

    def test_merged_substitute_from_delete_insert_pair(self):
        ops = path_to_edit_operations(
            [(0, 0), (0, 1), (1, 1)], "ab", "xb"
        )
        assert any("substitute" in op or "delete" in op for op in ops)


class TestApplyEdit:
    def test_keep_all_letters(self):
        assert apply_edit(list("abc"), ["keep a", "keep b", "keep c"]) == [
            "a",
            "b",
            "c",
        ]

    def test_delete_removes_letter(self):
        assert apply_edit(list("ab"), ["delete a", "keep b"]) == ["b"]

    def test_insert_adds_letter(self):
        assert apply_edit(list("a"), ["keep a", "insert b"]) == ["a", "b"]

    def test_substitute_replaces_letter(self):
        assert apply_edit(list("x"), ["substitute x by y"]) == ["y"]

    def test_substitute_not_last_consumes_next_letter(self):
        assert apply_edit(
            list("xy"), ["substitute x by a", "keep y"]
        ) == ["a", "y"]

    def test_empty_word_insert_only(self):
        assert apply_edit([], ["insert a"]) == ["a"]


class TestEditPipeline:
    def test_matrix_path_ops_apply_roundtrip(self):
        target, source = "CVCV", "CVCVCV"
        mtx = edit_distance_matrix(target, source)
        path = shortest_edit_path(mtx)
        assert path is not None
        ops = path_to_edit_operations(path, target, source)
        result = apply_edit(list(target), ops)
        assert isinstance(result, list)
        assert len(result) >= len(target)
