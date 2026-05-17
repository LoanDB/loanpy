"""Tests for loanpy.correspondences."""

import logging

from loanpy.correspondences import (
    _is_alternating_language_sequence,
    get_sound_correspondences,
)


def _row(lang, aligned, cog_id="cs1"):
    return {
        "Language_ID": lang,
        "Uralign": aligned,
        "Cognateset_ID": cog_id,
    }


class TestAlternatingLanguageSequence:
    def test_valid_two_row_table(self):
        table = [_row("hun", "a b"), _row("pu", "a c")]
        assert _is_alternating_language_sequence(table, {"hun"}, {"pu"})

    def test_valid_four_rows(self):
        table = [
            _row("d", "x"),
            _row("a", "y"),
            _row("d", "x"),
            _row("a", "z"),
        ]
        assert _is_alternating_language_sequence(table, {"d"}, {"a"})

    def test_odd_length_returns_false(self, caplog):
        with caplog.at_level(logging.INFO):
            ok = _is_alternating_language_sequence(
                [_row("hun", "a")], {"hun"}, {"pu"}
            )
        assert ok is False
        assert "Odd number" in caplog.text

    def test_wrong_language_on_even_row(self, caplog):
        table = [_row("hun", "a"), _row("hun", "b")]
        with caplog.at_level(logging.INFO):
            ok = _is_alternating_language_sequence(table, {"hun"}, {"pu"})
        assert ok is False
        assert "Problem in row" in caplog.text

    def test_empty_table(self):
        assert _is_alternating_language_sequence([], {"hun"}, {"pu"})


class TestGetSoundCorrespondences:
    def test_segment_pair_frequencies_and_examples(self):
        table = [
            _row("desc", "ɟ ŋ", "1"),
            _row("anc", "j ŋ", "1"),
        ]
        result = get_sound_correspondences(
            table, "Uralign", sep=" < ", prefix_descendant="H:", prefix_ancestor="P:"
        )
        assert result["AbsoluteFrequency"]["H:ɟ < P:j"] == 1
        assert result["SoundCorrespondences"]["ɟ"] == ["j"]
        assert result["Cognateset_IDs"]["H:ɟ < P:j"] == ["1"]
        assert "H:ɟ ŋ < P:j ŋ" in result["Examples"]["H:ɟ < P:j"]

    def test_duplicate_ancestor_ranking_by_frequency(self):
        table = [
            _row("d", "a", "1"),
            _row("a", "x", "1"),
            _row("d", "a", "2"),
            _row("a", "y", "2"),
            _row("d", "a", "3"),
            _row("a", "x", "3"),
        ]
        sc = get_sound_correspondences(table, "Uralign", sep=" < ")["SoundCorrespondences"]
        assert sc["a"][0] == "x"
        assert "y" in sc["a"]

    def test_absolute_frequency_sorted_descending(self):
        table = [
            _row("d", "x", "1"),
            _row("a", "p", "1"),
            _row("d", "x", "2"),
            _row("a", "p", "2"),
            _row("d", "x", "3"),
            _row("a", "p", "3"),
            _row("d", "y", "4"),
            _row("a", "q", "4"),
        ]
        freq = get_sound_correspondences(table, "Uralign", sep=" < ")["AbsoluteFrequency"]
        counts = list(freq.values())
        assert counts == sorted(counts)  # ascending by count in implementation
        assert freq["x < p"] == 3
        assert freq["y < q"] == 1

    def test_cognateset_ids_deduplicated(self):
        table = [
            _row("d", "a", "1"),
            _row("a", "b", "1"),
            _row("d", "a", "1"),
            _row("a", "b", "1"),
        ]
        ids = get_sound_correspondences(table, "Uralign", sep=" < ")["Cognateset_IDs"]
        assert ids["a < b"] == ["1"]

    def test_custom_aligned_column_name(self):
        table = [
            {"Language_ID": "d", "Alignment": "k a", "Cognateset_ID": "1"},
            {"Language_ID": "a", "Alignment": "k o", "Cognateset_ID": "1"},
        ]
        result = get_sound_correspondences(table, "Alignment", sep=" < ")
        assert "k < k" in result["AbsoluteFrequency"]
