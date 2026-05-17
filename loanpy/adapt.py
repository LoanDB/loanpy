"""Loanword adaptation: substitution and phonotactic repair."""

from __future__ import annotations

from collections.abc import Callable

from loanpy.edit import (
    apply_edit,
    edit_distance_matrix,
    path_to_edit_operations,
    shortest_edit_path,
)
from loanpy.phonotactics import get_closest_phonotactics


class Adapt:
    """Map donor segments onto a recipient inventory and repair prosody.

    Typical pipeline: learn substitutions from segment inventories, apply them to
    donor segments, then repair the CV profile against a phonotactic template list.

    Examples
    --------
    In a loanword-detection loop over two wordlists::

        ad = Adapt()
        ad.get_substitutions(donor_phonemes, recipient_phonemes, distance_fn, extra={})
        adapted = ad.substitute(donor_segments)
        repaired = ad.repair(adapted, cv_profile, phonotactic_templates)

    Notes
    -----
    Used in loanword-detection pipelines (e.g. Indo-Iranian–Hungarian
    ``make_results.py`` inside ``find_loanwords``): donor segments are
    substituted toward a recipient inventory, optionally repaired to legal CV
    templates, then aligned and scored.
    """

    substitutions: dict[str, str]

    def get_substitutions(
        self,
        donor_inventory: set[str],
        recipient_inventory: set[str],
        distance_func: Callable[[str, str], float],
        extra: dict[str, str],
    ) -> None:
        """Learn one-to-one donor→recipient substitutions by minimum distance.

        For each donor phoneme not in the recipient inventory, pick the recipient
        phoneme with smallest ``distance_func(donor, recipient)``. Merges with
        ``extra`` (manual overrides) into :attr:`substitutions`.

        Parameters
        ----------
        donor_inventory, recipient_inventory:
            Segment inventories (sets of phoneme symbols).
        distance_func:
            Callable returning a numeric distance (e.g. feature-based).
        extra:
            Fixed substitutions applied on top of learned ones.
        """
        substitutions = {}
        for donor_phoneme in donor_inventory - recipient_inventory:
            best_substitution = ""
            lowest_distance = float("inf")
            for recipient_phoneme in recipient_inventory:
                distance = distance_func(donor_phoneme, recipient_phoneme)
                if distance < lowest_distance:
                    lowest_distance = distance
                    best_substitution = recipient_phoneme
            substitutions[donor_phoneme] = best_substitution
        self.substitutions = substitutions | extra

    def substitute(self, segments: list[str]) -> list[str]:
        """Replace segments using :attr:`substitutions` (identity if unmapped).

        Parameters
        ----------
        segments:
            Donor segment list.

        Returns
        -------
        list[str]
            Substituted segments.
        """
        substitute = []
        for seg in segments:
            if sub := self.substitutions.get(seg, seg):
                substitute.append(sub)
        return substitute

    def repair(
        self,
        segments: list[str],
        cv_profile: list[str],
        phonotactic_inventory: list[str],
        extra_repair: dict[str, str] | None = None,
    ) -> list[str]:
        """Align segments to the closest legal CV template via edit operations.

        Parameters
        ----------
        segments:
            Segment list (often after :meth:`substitute`).
        cv_profile:
            Parallel C/V profile for ``segments``.
        phonotactic_inventory:
            Allowed templates (see :func:`~loanpy.phonotactics.expand_phonotactics`).
        extra_repair:
            Optional map from joined CV strings to fixed templates, bypassing
            nearest-neighbour search.

        Returns
        -------
        list[str]
            Segments after applying insert/delete/substitute operations implied by
            the CV-profile edit path (may include ``"C"`` / ``"V"`` placeholders).

        Notes
        -----
        **make_results.py** may post-process placeholder vowels/consonants for
        vowel harmony; loanpy only returns the structurally repaired sequence.
        """
        cv_profile_str = "".join(cv_profile)
        if extra_repair is not None and cv_profile_str in extra_repair:
            predicted_phonotactics = extra_repair[cv_profile_str]
        else:
            predicted_phonotactics = get_closest_phonotactics(
                cv_profile, phonotactic_inventory
            )

        matrix = edit_distance_matrix(cv_profile_str, predicted_phonotactics)
        path = shortest_edit_path(matrix)
        editops = path_to_edit_operations(path, cv_profile_str, predicted_phonotactics)
        return apply_edit(segments, editops)
