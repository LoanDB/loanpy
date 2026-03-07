"""Core loanpy functionality: phoneme clustering, sound change, alignment."""


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

def cluster_cv(segments: list[str], cv_profile: list[str]) -> list[str]:
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
