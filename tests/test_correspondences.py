"""Tests for loanpy.correspondences."""

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
    def test_valid_pair(self):
        table = [_row("hun", "a b"), _row("pu", "a c")]
        assert _is_alternating_language_sequence(table, {"hun"}, {"pu"})

    def test_odd_length(self):
        assert not _is_alternating_language_sequence(
            [_row("hun", "a")], {"hun"}, {"pu"}
        )

    def test_wrong_language(self):
        table = [_row("hun", "a"), _row("hun", "b")]
        assert not _is_alternating_language_sequence(table, {"hun"}, {"pu"})


class TestGetSoundCorrespondences:
    def test_basic_counts(self):
        table = [
            _row("desc", "ɟ ŋ", "1"),
            _row("anc", "j ŋ", "1"),
            _row("desc", "a", "2"),
            _row("anc", "a", "2"),
        ]
        result = get_sound_correspondences(
            table, "Uralign", sep=" < ", prefix_descendant="", prefix_ancestor=""
        )
        assert result["AbsoluteFrequency"]["ɟ < j"] == 1
        assert "j" in result["SoundCorrespondences"]["ɟ"]
        assert "1" in result["Cognateset_IDs"]["ɟ < j"]

    def test_sorted_by_frequency(self):
        table = [
            _row("d", "x", "1"),
            _row("a", "x", "1"),
            _row("d", "x", "2"),
            _row("a", "y", "2"),
            _row("d", "x", "3"),
            _row("a", "x", "3"),
        ]
        freq = get_sound_correspondences(table, "Uralign", sep=" < ")["AbsoluteFrequency"]
        items = list(freq.items())
        assert items == sorted(items, key=lambda kv: kv[1])
