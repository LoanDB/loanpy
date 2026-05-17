"""Sound correspondences from aligned cognate tables."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


def _is_alternating_language_sequence(
    table: Sequence[Mapping[str, str]],
    descendant_language_ids: set[str],
    ancestor_language_ids: set[str],
) -> bool:
    """Return True if rows strictly alternate descendant / ancestor languages."""
    if len(table) % 2:
        logging.info("Odd number of rows.")
        return False
    for index, row in enumerate(table):
        allowed = (
            descendant_language_ids if index % 2 == 0 else ancestor_language_ids
        )
        if row["Language_ID"] not in allowed:
            logging.info("Problem in row %s: %s not in %s", index, row, allowed)
            return False
    return True


def get_sound_correspondences(
    table: Sequence[Mapping[str, str]],
    aligned_col: str,
    prefix_descendant: str = "",
    prefix_ancestor: str = "",
    sep: str = " ",
) -> dict[str, dict]:
    """Extract segment correspondences from paired cognate alignment rows.

    Expects ``table`` to list cognate rows in **descendant, ancestor, descendant,
    ancestor, …** order (same convention as many CLDF ``cognates.csv`` exports).
    Each consecutive pair of rows is zipped segment-wise along ``aligned_col``.

    Parameters
    ----------
    table:
        Sequence of row dicts (e.g. from ``csv.DictReader``).
    aligned_col:
        Column with space-separated aligned segments (e.g. ``"Uralign"``).
    prefix_descendant, prefix_ancestor:
        Optional prefixes for correspondence keys and examples.
    sep:
        Separator between descendant and ancestor parts in pair keys.

    Returns
    -------
    dict
        Keys:

        * ``SoundCorrespondences`` — descendant segment → ranked ancestor segments
        * ``AbsoluteFrequency`` — ``"desc < anc"`` → count
        * ``Cognateset_IDs`` — pair → cognate set ids
        * ``Examples`` — pair → example alignment strings

    Examples
    --------
    Build a frequency table for alignment scoring::

        rows = list(csv.DictReader(open("cognates.csv", encoding="utf-8")))
        stats = get_sound_correspondences(rows, "Uralign", sep=" < ")
        scorer = stats["AbsoluteFrequency"]

    Notes
    -----
    * **Quantitative analysis** — ``make_results.py`` in the Indo-Iranian–Hungarian
      study calls this on CLDF cognate tables to build TOML scorers and in-memory
      weights for :class:`~loanpy.uralign.Uralign`.
    * **CLDF workflows** — training data from any wordlist with alternating
      descendant/ancestor rows and an alignment column can be passed in; no
      hard-coded language names are required.
    """
    correspondences: dict[str, dict] = {
        key: defaultdict(list)
        for key in (
            "SoundCorrespondences",
            "AbsoluteFrequency",
            "Cognateset_IDs",
            "Examples",
        )
    }

    for index in range(0, len(table) - 1, 2):
        descendant_row, ancestor_row = table[index], table[index + 1]
        for descendant_seg, ancestor_seg in zip(
            descendant_row[aligned_col].split(),
            ancestor_row[aligned_col].split(),
        ):
            correspondences["SoundCorrespondences"][descendant_seg].append(
                ancestor_seg
            )
            pair_key = (
                f"{prefix_descendant}{descendant_seg}{sep}"
                f"{prefix_ancestor}{ancestor_seg}"
            )
            correspondences["AbsoluteFrequency"][pair_key].append(1)
            correspondences["Cognateset_IDs"][pair_key].append(
                ancestor_row["Cognateset_ID"]
            )
            example = (
                f"{prefix_descendant}{descendant_row[aligned_col]}"
                f"{sep}{prefix_ancestor}{ancestor_row[aligned_col]}"
            )
            correspondences["Examples"][pair_key].append(example)

    correspondences["SoundCorrespondences"] = {
        descendant: [
            ancestor for ancestor, _ in Counter(ancestors).most_common()
        ]
        for descendant, ancestors in correspondences["SoundCorrespondences"].items()
    }
    correspondences["AbsoluteFrequency"] = {
        pair: sum(counts)
        for pair, counts in correspondences["AbsoluteFrequency"].items()
    }
    correspondences["AbsoluteFrequency"] = dict(
        sorted(correspondences["AbsoluteFrequency"].items(), key=lambda item: item[1])
    )
    correspondences["Cognateset_IDs"] = {
        pair: list(dict.fromkeys(ids))
        for pair, ids in correspondences["Cognateset_IDs"].items()
    }

    return correspondences
