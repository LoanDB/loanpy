"""Core loanpy functionality: phoneme clustering, sound change, alignment."""


def uralign(seqHU, seqPU, seqHU_cv, seqPU_cv, *, initial_gap=True, final_gap=True):
    """Align Hungarian and PU/PFU/PUg data sequentially; optional initial and final gap handling."""
    if initial_gap and seqHU_cv.startswith("V"):
        seqHU = "#- " + seqHU
        if seqPU_cv.startswith("V"):
            seqPU = "- " + seqPU

    hu, pu = seqHU.split(), seqPU.split()
    if final_gap:
        diff = abs(len(pu) - len(hu))
        if len(hu) < len(pu):
            hu.append("-#")
            pu = pu[:-diff] + [".".join(pu[-diff:])]
        elif len(hu) > len(pu):
            hu = hu[:-diff] + ["+"] + [".".join(hu[-diff:])]
    else:
        n = min(len(hu), len(pu))
        hu, pu = hu[:n], pu[:n]
    return " ".join(hu), " ".join(pu)


def cluster_cv(segments, cv_profile):
    """Cluster consonants and vowels together.

    Example: 'f l a ʊ ə' + 'C C V V V' -> 'f.l a.ʊ.ə'
    """
    chars, props, result = segments.split(" "), cv_profile.split(" "), []
    for i, (char, prop) in enumerate(zip(chars, props)):
        if i == 0 or prop != props[i - 1]:
            result.append(char)
        else:
            result[-1] += "." + char
    return " ".join(result)
