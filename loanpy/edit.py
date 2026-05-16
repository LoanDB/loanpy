"""Sequence edit distance and edit-operation utilities."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import Union


def edit_distance_with2ops(
    string1: str,
    string2: str,
    w_del: Union[int, float] = 1,
    w_ins: Union[int, float] = 1,
) -> Union[int, float]:
    """Edit distance allowing only insertions and deletions."""
    m = len(string1)
    n = len(string2)
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcs_table[i][j] = 0
            elif string1[i - 1] == string2[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    lcs = lcs_table[m][n]
    return (m - lcs) * w_del + (n - lcs) * w_ins


def apply_edit(word: Iterable[str], editops: list[str]) -> list[str]:
    """Apply human-readable edit operations to a word."""
    out, letter = [], iter(word)
    for i, op in enumerate(editops):
        if i != len(editops):
            if "keep" in op:
                out.append(next(letter))
            elif "delete" in op:
                next(letter)
        if "substitute" in op:
            out.append(op[op.index(" by ") + 4 :])
            if i != len(editops) - 1:
                next(letter)
        elif "insert" in op:
            out.append(op[len("insert ") :])
    return out


def substitute_operations(operations: list[str]) -> list[str]:
    """Merge adjacent delete/insert pairs into substitute operations."""
    i = 0
    while i < len(operations) - 1:
        if operations[i].startswith("delete ") and operations[i + 1].startswith(
            "insert "
        ):
            deleted = operations[i][7:]
            inserted = operations[i + 1][7:]
            operations[i : i + 2] = [f"substitute {deleted} by {inserted}"]
        elif operations[i].startswith("insert ") and operations[i + 1].startswith(
            "delete "
        ):
            inserted = operations[i][7:]
            deleted = operations[i + 1][7:]
            operations[i : i + 2] = [f"substitute {deleted} by {inserted}"]
        else:
            i += 1
    return operations


def path_to_edit_operations(op_list: list[tuple[int, int]], s1: str, s2: str) -> list[str]:
    """Convert matrix path coordinates to human-readable edit operations."""
    s1, s2 = "#" + s1, "#" + s2
    out = []
    for i in range(1, len(op_list)):
        direction = [
            op_list[i][0] - op_list[i - 1][0],
            op_list[i][1] - op_list[i - 1][1],
        ]
        if direction == [1, 1]:
            out.append(f"keep {s1[op_list[i][1]]}")
        elif direction == [0, 1]:
            out.append(f"delete {s1[op_list[i][1]]}")
        elif direction == [1, 0]:
            out.append(f"insert {s2[op_list[i][0]]}")
    return substitute_operations(out)


def edit_distance_matrix(target: Iterable, source: Iterable) -> list[list[int]]:
    """Minimum edit-distance matrix (insert/delete cost 1 each)."""
    target = ["#"] + list(target)
    source = ["#"] + list(source)
    sol = [[0] * len(target) for _ in range(len(source))]
    sol[0] = list(range(len(target)))
    for j in range(len(source)):
        sol[j][0] = j
    if target[1] != source[1]:
        sol[1][1] = 2
    for c in range(1, len(target)):
        for r in range(1, len(source)):
            if target[c] != source[r]:
                sol[r][c] = min(sol[r - 1][c], sol[r][c - 1]) + 1
            else:
                sol[r][c] = sol[r - 1][c - 1]
    return sol


def shortest_edit_path(mtx):
    """Shortest edit path through a distance matrix (corner to corner)."""
    rows, cols = len(mtx), len(mtx[0])
    start = (0, 0)
    end = (rows - 1, cols - 1)
    if start == end:
        return [start]
    dist = {start: 0}
    path = {}
    queue = [(0, start)]
    while queue:
        current_dist, (i, j) = heapq.heappop(queue)
        if current_dist > dist.get((i, j), float("inf")):
            continue
        if (i, j) == end:
            break
        if j < cols - 1:
            neighbor = (i, j + 1)
            weight = 1 if mtx[i][j + 1] != mtx[i][j] else 0
            new_dist = current_dist + weight
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                path[neighbor] = (i, j)
                heapq.heappush(queue, (new_dist, neighbor))
        if i < rows - 1:
            neighbor = (i + 1, j)
            weight = 1 if mtx[i + 1][j] != mtx[i][j] else 0
            new_dist = current_dist + weight
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                path[neighbor] = (i, j)
                heapq.heappush(queue, (new_dist, neighbor))
        if i < rows - 1 and j < cols - 1 and mtx[i + 1][j + 1] == mtx[i][j]:
            neighbor = (i + 1, j + 1)
            new_dist = current_dist
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                path[neighbor] = (i, j)
                heapq.heappush(queue, (new_dist, neighbor))
    if end not in path:
        return None
    shortest_path = [end]
    while shortest_path[-1] != start:
        shortest_path.append(path[shortest_path[-1]])
    shortest_path.reverse()
    return shortest_path
