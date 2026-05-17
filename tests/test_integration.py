"""End-to-end workflows mirroring CLDF conversion and loanword scoring."""

import csv
import io

from loanpy import Adapt, Cluster, Uralign, get_sound_correspondences


class TestCldfStyleClustering:
    def test_uew_style_cv_clusters(self):
        segments = "f l a".split()
        cv = ["C", "C", "V"]
        assert Cluster.cv(segments, cv) == ["f.l", "a"]

    def test_uesz_style_glide_clustering(self):
        segments = ["a", "ɣ", "a"]
        cv = ["V", "C", "V"]
        glides = Cluster.glides(segments, cv)
        assert glides == ["a.ɣ.a"]


class TestCorrespondenceToScoring:
    def test_mined_scorer_used_by_uralign(self):
        rows = [
            {
                "Language_ID": "hun",
                "Uralign": "ɟ ŋ",
                "Cognateset_ID": "1",
            },
            {
                "Language_ID": "pu",
                "Uralign": "j ŋ",
                "Cognateset_ID": "1",
            },
        ]
        stats = get_sound_correspondences(rows, "Uralign", sep=" < ")
        scorer = stats["AbsoluteFrequency"]
        seg_h = ["ɟ", "ŋ"]
        seg_p = ["j", "ŋ"]
        alm_h, alm_p = Uralign.hu(seg_h.copy(), seg_p.copy(), "C", "C")
        score = Uralign.get_score(alm_h, alm_p, scorer, freq_filter=1)
        assert score > 0


class TestMakeResultsStyleAdaptation:
    def test_substitute_and_repair_align_input_shape(self):
        ad = Adapt()
        ad.substitutions = {"ʔ": "", "n.j": "nʲ", "θ": "t"}
        segments = ["θ", "a", "k"]
        profile = ["C", "V", "C"]
        phonotactics = ["C V", "C V C V", "C V C C V"]
        substituted = ad.substitute(segments)
        assert "θ" not in substituted
        repaired = ad.repair(
            substituted,
            profile,
            phonotactics,
            extra_repair={"CVVCV": "CVCVCV"},
        )
        assert repaired == ["t", "a"]  # repair may drop segments on delete ops


class TestCsvRoundtrip:
    def test_read_cognates_csv_shape(self):
        csv_text = """Language_ID,Uralign,Cognateset_ID
hun,k a,1
pu,k o,1
"""
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        stats = get_sound_correspondences(rows, "Uralign", sep=" < ")
        assert "k < k" in stats["AbsoluteFrequency"]
