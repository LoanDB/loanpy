"""Sound correspondences from aligned cognate tables."""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_PROTO_LANGUAGES = ("PU", "PFU", "PUg")
DEFAULT_HUN_IGNORE_LEXEMES = (
    "ár", "iv", "hány", "ház", "magyar", "méh", "méz",
    "ostor", "ágyék", "fej", "fasz", "szomjú", "szarv",
    "arany", "agg", "úr", "hét", "száz", "szeg",
    "titok", "tavasz", "tehén", "szar",
)


def _is_alternating_language_sequence(
    entries: Sequence[Mapping[str, str]],
    source_langs: Sequence[str],
    target_langs: Sequence[str],
    forms: Mapping[str, Mapping[str, str]],
) -> bool:
    """True if rows alternate source language(s) then target language(s)."""
    if len(entries) % 2:
        logging.info("Odd number of rows; source/target language is missing.")
        return False
    for idx, row in enumerate(entries):
        lang = forms[row["Form_ID"]]["Language_ID"]
        expected = source_langs if idx % 2 == 0 else target_langs
        if lang not in expected:
            logging.info("Problem in row %s: %s", idx, row)
            return False
    return True


def get_correspondences(
    table: Sequence[Mapping[str, str]],
    heur: Mapping[str, Sequence[str]] | None = None,
    *,
    aligned_col: str = "ALIGNMENT",
    prefix_src: str = "",
    prefix_tgt: str = "",
    sep: str = " ",
) -> dict[str, dict]:
    """Extract segment and prosody correspondences from paired alignment rows."""
    out: dict[str, dict] = {
        key: defaultdict(list)
        for key in (
            "corr", "freq", "COGIDS", "examples",
            "corr_ptct", "freq_ptct", "COGIDS_ptct", "examples_ptct",
        )
    }

    for i in range(0, len(table) - 1, 2):
        row1, row2 = table[i], table[i + 1]
        for seg_a, seg_b in zip(
            row1[aligned_col].split(), row2[aligned_col].split()
        ):
            out["corr"][seg_a].append(seg_b)
            key = f"{prefix_src}{seg_a}{sep}{prefix_tgt}{seg_b}"
            out["freq"][key].append(1)
            out["COGIDS"][key].append(int(row2["Cognateset_ID"]))
            ex = (
                f"{prefix_src}{row1[aligned_col]}"
                f"{sep}{prefix_tgt}{row2[aligned_col]}"
            )
            out["examples"][key].append(ex)

        cv1, cv2 = row1["PROSODY"], row2["PROSODY"]
        out["corr_ptct"][cv1].append(cv2)
        ptct_key = f"{prefix_src}{cv1}{sep}{prefix_tgt}{cv2}"
        out["freq_ptct"][ptct_key].append(1)
        out["COGIDS_ptct"][ptct_key].append(int(row2["Cognateset_ID"]))
        out["examples_ptct"][ptct_key].append(ex)

    for key in ("corr", "corr_ptct"):
        out[key] = {
            k: [j[0] for j in Counter(out[key][k]).most_common()]
            for k in out[key]
        }
    for key in ("freq", "freq_ptct"):
        out[key] = {k: sum(out[key][k]) for k in out[key]}
        out[key] = dict(sorted(out[key].items(), key=lambda kv: kv[1]))
    for key in ("COGIDS", "COGIDS_ptct"):
        out[key] = {k: list(dict.fromkeys(out[key][k])) for k in out[key]}

    if heur:
        for k, targets in heur.items():
            if k in out["corr"]:
                out["corr"][k].extend(targets)
                out["corr"][k] = list(dict.fromkeys(out["corr"][k]))
            else:
                out["corr"][k] = list(targets)

    return out


def _forms_path_for_cognates(cognates_path: Path) -> Path:
    return Path(str(cognates_path).replace("cognates.csv", "forms.csv"))


def _scorer_source_tag(cognates_path: Path) -> str:
    path = str(cognates_path)
    if "redeiuralic" in path:
        return "uew"
    if "holopainenprotouralic" in path:
        return "holo"
    return ""


def _write_scorer_toml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from toml import dump
    except ImportError as exc:
        raise ImportError(
            "Install the 'toml' package to write scorer files, or pass scorer_out_dir=None."
        ) from exc
    with open(path, "w", encoding="utf-8") as f:
        dump(data, f)
    logging.info("Wrote %s.", path)


def get_sound_correspondences(
    cognates_path: Path | str,
    alignment_col: str,
    *,
    freq_filter: int = 0,
    recipient_lang: str = "hun",
    proto_languages: Sequence[str] = DEFAULT_PROTO_LANGUAGES,
    ignore_lexemes: Sequence[str] = DEFAULT_HUN_IGNORE_LEXEMES,
    scorer_out_dir: Path | str | None = None,
) -> dict[str, dict]:
    """
    Build sound-correspondence statistics from a CLDF cognates table.

    Reads ``forms.csv`` beside ``cognates.csv``, filters training pairs,
    and returns correspondence dicts (including ``freq`` for Uralign scoring).
    """
    cognates_path = Path(cognates_path)
    forms_path = _forms_path_for_cognates(cognates_path)

    with open(forms_path, encoding="utf-8") as f:
        forms = {row["ID"]: row for row in csv.DictReader(f)}
    with open(cognates_path, encoding="utf-8") as f:
        entries = list(csv.DictReader(f))

    ignore_lexemes_set = set(ignore_lexemes)
    ignored_cogids = {
        row["Cognateset_ID"]
        for row in entries
        if forms[row["Form_ID"]]["Language_ID"] == recipient_lang
        and forms[row["Form_ID"]]["Form"] in ignore_lexemes_set
    }
    entries = [row for row in entries if row["Cognateset_ID"] not in ignored_cogids]
    entries = [
        row
        for i, row in enumerate(entries)
        if (
            forms[row["Form_ID"]]["Language_ID"] == recipient_lang
            and forms[entries[i + 1]["Form_ID"]]["Language_ID"] in proto_languages
        )
        or forms[row["Form_ID"]]["Language_ID"] in proto_languages
    ]

    if "redeiuralic" in str(cognates_path):
        entries = [
            row for row in entries if forms[row["Form_ID"]]["Certain"] == "true"
        ]

    for row in entries:
        alignment = row[alignment_col]
        if " +" in alignment:
            row[alignment_col] = alignment[: alignment.index(" +")]

    for row in entries:
        if (
            forms[row["Form_ID"]]["Language_ID"] != recipient_lang
            and row[alignment_col].endswith("e")
        ):
            row[alignment_col] = row[alignment_col][:-1] + "i"

    assert _is_alternating_language_sequence(
        entries, (recipient_lang,), proto_languages, forms
    ), f"Language sequence in {cognates_path} not valid."
    logging.info("language sequence valid")

    for row in entries:
        row["COGID"] = row["Cognateset_ID"]
        row["ALIGNMENT"] = row[alignment_col]
        cv = forms[row["Form_ID"]]["CV_profile"]
        row["PROSODY"] = cv.replace(" ", "")

    sep = " < "
    correspondences = get_correspondences(entries, sep=sep)
    fishy_cogids = {
        cogid
        for corr, freq in correspondences["freq"].items()
        if freq <= freq_filter
        for cogid in correspondences["COGIDS"][corr]
    }
    entries = [
        row for row in entries if int(row["Cognateset_ID"]) not in fishy_cogids
    ]
    correspondences = get_correspondences(entries, sep=sep)

    if scorer_out_dir is not None:
        out_dir = Path(scorer_out_dir)
        tag = _scorer_source_tag(cognates_path)
        _write_scorer_toml(
            correspondences, out_dir / f"uralign_{tag}_{alignment_col}.toml"
        )

    return correspondences
