"""Alignment utilities for Uralic data."""


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

    @staticmethod
    def get_score(seqA, seqB, scorer, freq_filter=1):
        score = 0
        for a, b in zip(seqA, seqB):
            local_score = max(scorer.get(f"{a} < {i}", -1000) for i in b.split("|"))
            if local_score < freq_filter and "." in b:
                local_score = scorer.get(f"{a} < {b}", -1000)
            if local_score > freq_filter:
                score += local_score
            else:
                score -= 1000
        return score
