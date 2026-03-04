"""Core loanpy functionality: phoneme clustering, sound change, alignment."""


def cluster_phonemes(ipa_str, props):
    """Cluster consonants and vowels together.

    Example: 'f l a ʊ ə' + 'C C V V V' -> 'f.l a.ʊ.ə'
    """
    chars, props, result = ipa_str.split(), props.split(), []
    for i, (char, prop) in enumerate(zip(chars, props)):
        if i == 0 or prop != props[i - 1]:
            result.append(char)
        else:
            result[-1] += "." + char
    return " ".join(result)
