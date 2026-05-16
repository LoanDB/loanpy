"""Utilities for loanword adaptation."""

from loanpy.edit import (
    apply_edit,
    edit_distance_matrix,
    path_to_edit_operations,
    shortest_edit_path,
)
from loanpy.phonotactics import get_closest_phonotactics


class Adapt:
    """Utilities for loanword adaptation."""

    def get_substitutions(
        self,
        donor_inventory,
        recipient_inventory,
        distance_func,
        extra,
    ):
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
        substitute = []
        for seg in segments:
            if (sub := self.substitutions.get(seg, seg)):
                substitute.append(sub)
        return substitute

    def repair(self, segments, cv_profile, phonotactic_inventory, extra_repair=None):
        if extra_repair is not None and " ".join(cv_profile) in extra_repair:
            predicted_phonotactics = extra_repair[" ".join(cv_profile)]
        else:
            predicted_phonotactics = get_closest_phonotactics(
                cv_profile, phonotactic_inventory
            )
        cv_str = "".join(cv_profile)
        matrix = edit_distance_matrix(cv_str, predicted_phonotactics)
        path = shortest_edit_path(matrix)
        editops = path_to_edit_operations(path, cv_str, predicted_phonotactics)
        return apply_edit(segments, editops)
