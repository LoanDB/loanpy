"""Prosodic template expansion and closest-template matching."""

from __future__ import annotations

from itertools import product

from loanpy.edit import edit_distance_with2ops


def _expand_syllable_template(syllable: str) -> list[list[str]]:
    """Expand one syllable template, e.g. ``(C)V(C)`` -> V, CV, VC, CVC."""
    slots: list[tuple[str, str]] = []
    i = 0
    while i < len(syllable):
        if syllable.startswith("(C)", i):
            slots.append(("optional", "C"))
            i += 3
        elif syllable[i] == "C":
            slots.append(("required", "C"))
            i += 1
        elif syllable[i] == "V":
            slots.append(("required", "V"))
            i += 1
        else:
            raise ValueError(
                f"invalid syllable template {syllable!r} at {syllable[i:]!r}"
            )
    n_optional = sum(1 for kind, _ in slots if kind == "optional")
    variants = []
    for include in product((False, True), repeat=n_optional):
        out: list[str] = []
        opt = iter(include)
        for kind, symbol in slots:
            if kind == "optional":
                if next(opt):
                    out.append(symbol)
            else:
                out.append(symbol)
        variants.append(out)
    return variants


def expand_phonotactics(formula: str) -> list[str]:
    """
    Expand a prosodic formula into space-separated CV templates.

    ``(C)`` marks an optional consonant slot; bare ``C`` and ``V`` are required.
    Syllables are joined with ``+``, e.g. ``"(C)V(C)+CV(C)+CV"``.
    """
    syllables = [s.strip() for s in formula.split("+")]
    words = []
    for combo in product(*(_expand_syllable_template(s) for s in syllables)):
        segments: list[str] = []
        for part in combo:
            segments.extend(part)
        words.append(" ".join(segments))
    return words


def get_closest_phonotactics(cv_profile, phonotactic_inventory):
    """Pick the inventory template closest to a CV profile (insert/delete only)."""
    cv_profile_str = "".join(cv_profile)
    dist_and_strucs = [
        (
            edit_distance_with2ops(cv_profile_str, tpl.replace(" ", ""), w_ins=100),
            tpl.replace(" ", ""),
        )
        for tpl in phonotactic_inventory
    ]
    return min(dist_and_strucs)[1]
