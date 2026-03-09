"""Core loanpy functionality: phoneme clustering, sound change, alignment."""

from __future__ import annotations


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


class Uralign:
    """Alignment utilities for Uralic data."""

    @staticmethod
    def hu(
        seqHU: list[str],
        seqPU: list[str],
        seqHU_cv0: str,
        seqPU_cv0: str,
        initial_gap: bool = True,
        final_gap: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Align Hungarian and PU/PFU/PUg data sequentially; optional initial and final gap handling."""
        if initial_gap:
            if seqHU_cv0 == "V":
                seqHU.insert(0, "#-")
                if seqPU_cv0 == "V":
                    seqPU.insert(0, "-")

        if final_gap:
            diff = abs(len(seqPU) - len(seqHU))
            if len(seqHU) < len(seqPU):
                seqHU.append("-#")
                seqPU = seqPU[:-diff] + [".".join(seqPU[-diff:])]
            elif len(seqHU) > len(seqPU):
                seqHU = seqHU[:-diff] + ["+"] + [".".join(seqHU[-diff:])]
        else:
            n = min(len(seqHU), len(seqPU))
            seqHU, seqPU = seqHU[:n], seqPU[:n]
        return seqHU, seqPU
