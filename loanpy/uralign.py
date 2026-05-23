"""Descendant–ancestor alignment and correspondence-based scoring."""


class Uralign:
    """Sequential alignment and scoring for etymological comparison.

    The API is language-pair agnostic: method names such as ``hu`` reflect
    historical use (Hungarian vs. proto-Uralic) but accept any two segment lists
    with CV profiles.

    Examples
    --------
    In a loanword-detection pipeline, align donor and recipient segments then
    score against mined correspondences::

        alm_d, alm_a = Uralign.hu(seg_d, seg_a, cv_d[0], cv_a[0])
        score = Uralign.get_score(alm_d, alm_a, scorer, freq_filter=2)

    Notes
    -----
    * **CLDF conversion** — ``Uralign.hu`` writes ``Uralign`` / ``Uralign_cluster``
      columns in cognate tables (UEW-hu, SeimaTurbino-hu).
    * **Quantitative analysis** — loanword-detection pipelines (e.g.
      Indo-Iranian–Hungarian ``make_results.py``) use ``Uralign.hu`` and
      ``Uralign.get_score`` with correspondence scorers from
      :func:`~loanpy.correspondences.get_sound_correspondences`.
    """

    @staticmethod
    def hu(
        seqHU: list[str],
        seqPU: list[str],
        seqHU_cv0: str,
        seqPU_cv0: str,
        initial_gap: bool = True,
        final_gap: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Align two segment sequences with optional initial and final gap rules.

        Parameters
        ----------
        seqHU, seqPU:
            Segment lists (modified in place when gaps are inserted).
        seqHU_cv0, seqPU_cv0:
            Word-initial C/V labels for gap decisions.
        initial_gap:
            If True and the descendant begins with a vowel, prepend ``#-`` /
            ``-`` markers.
        final_gap:
            If True, pad or cluster the longer sequence at the word edge.

        Returns
        -------
        tuple[list[str], list[str]]
            Aligned segment pair.

        Notes
        -----
        Used in **CLDF conversion** and in **make_results.py** (loanword scoring).
        """
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
    def get_score(
        seqA: list[str],
        seqB: list[str],
        scorer: dict[tuple[str, str], float],
        freq_filter: int = 2,
    ) -> int:
        """Sum correspondence scores along an alignment.

        For each aligned pair ``(a, b)`` the key ``(a, b)`` is looked up in
        ``scorer``. Pairs below ``freq_filter`` incur a large penalty.

        Parameters
        ----------
        seqA, seqB:
            Parallel aligned token lists.
        scorer:
            Mapping from correspondence keys to weights (often absolute
            frequencies from :func:`~loanpy.correspondences.get_sound_correspondences`).
        freq_filter:
            Minimum score for a pair to count positively.

        Returns
        -------
        int
            Aggregate alignment score.

        Notes
        -----
        Used in **make_results.py** together with scores from
        :func:`~loanpy.correspondences.get_sound_correspondences`.
        """
        score = 0
        for a, b in zip(seqA, seqB):
            local_score = scorer.get((a, b), -1000)
            if local_score >= freq_filter:
                score += local_score
            else:
                score -= 1000
        return score
