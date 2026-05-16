"""Phoneme clustering: CV grouping and glide/liquid clustering."""


class Cluster:
    """Phoneme clustering: CV grouping and glide/liquid clustering."""

    @staticmethod
    def cv(segments: list[str], cv_profile: list[str]) -> list[str]:
        """Cluster consonants and vowels together.

        Example: 'f l a ʊ ə' + 'C C V V V' -> 'f.l a.ʊ.ə'
        """
        result = []
        for i, (segment, cv) in enumerate(zip(segments, cv_profile)):
            if i == 0 or cv != cv_profile[i - 1]:
                result.append(segment)
            else:
                result[-1] += "." + segment
        return result

    @staticmethod
    def glides(
        segments: list[str],
        cv_profile: list[str],
        cluster_between_vowels: tuple[str, ...] = ("ɣ", "w", "v"),
        cluster_after_l: tuple[str, ...] = ("t͡ʃ", "d"),
    ) -> list[str]:
        """
        Cluster phonemes between vowels and after l
        (e.g. ɣ, w, v between V; t͡ʃ, d after l).
        """
        if len(segments) != len(cv_profile):
            raise ValueError("segments and cv_profile must have the same length")
        cluster2 = []
        profile2 = []
        for idx, phoneme in enumerate(segments):
            if (
                idx != 0
                and phoneme in cluster_between_vowels
                and cv_profile[idx - 1] == "V"
            ):
                cluster2[-1] += f".{phoneme}"
                profile2[-1] += f".{cv_profile[idx]}"
            elif (
                idx != 0
                and phoneme in cluster_after_l
                and len(cluster2) > 0
                and cluster2[-1] == "l"
            ):
                cluster2[-1] += f".{phoneme}"
                profile2[-1] += f".{cv_profile[idx]}"
            else:
                cluster2.append(phoneme)
                profile2.append(cv_profile[idx])

        cluster3 = []
        for idx, phoneme in enumerate(cluster2):
            if (
                idx != 0
                and profile2[idx] == "V"
                and any(f".{ph}" in cluster3[-1] for ph in cluster_between_vowels)
            ):
                cluster3[-1] += f".{phoneme}"
            else:
                cluster3.append(phoneme)
        return cluster3

    @staticmethod
    def gaps(seqA: list[str], seqB: list[str]) -> tuple[list[str], list[str]]:
        """Collapse gaps to at most one per position."""
        seqA_new, seqB_new = [], []
        for idx, (tokA, tokB) in enumerate(zip(seqA, seqB)):
            if idx != 0 and tokB == "-" and seqB_new[-1] == "-":
                seqA_new[-1] += f".{tokA}"
            else:
                seqA_new.append(tokA)
                seqB_new.append(tokB)
        if seqB_new[-1] == "-":
            seqA_new.insert(-1, "+")  # Yes, -1 means penultimate, weirdly.
            seqB_new.pop(-1)  # And yes, THIS -1 now means the last one.
        return seqA_new, seqB_new
