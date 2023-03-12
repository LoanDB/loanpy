"""
recover sound correspondences
"""
# TODO: write tests and documentation with ChatGPT

from collections import defaultdict, Counter
from heapq import nsmallest
import json

from panphon import FeatureTable
from panphon.distance import Distance

def qfy(table, heur=""):
    """
    Get sound correspondences
    """

    table = table.split("\n")[:-1]
    cols = {col: i for i, col in enumerate(table.pop(0).split("\t"))}
    iterrows = iter(table)
    out = [defaultdict(list), defaultdict(int), defaultdict(list),
           defaultdict(list), defaultdict(int), defaultdict(list)]  # not *2!

    while True:
        try:
            row1, row2 = next(iterrows).split("\t"), next(iterrows).split("\t")
            for i, j in zip(row1[cols["ALIGNMENT"]].split(" "),
                            row2[cols["ALIGNMENT"]].split(" ")):
                out[0][i].append(j)
                out[1][f"{i} {j}"] += 1
                out[2][f"{i} {j}"].append(row2[cols["COGID"]])
            cv1, cv2 = row1[cols["CV_SEGMENTS"]], row2[cols["CV_SEGMENTS"]]
            out[3][cv1].append(cv2)
            out[4][f"{cv1} {cv2}"] += 1
            out[5][f"{cv1} {cv2}"].append(row2[cols["COGID"]])
        except StopIteration:
            break

    for i in [0, 3]: # sort by freq
        out[i] = {k: [j[0] for j in Counter(out[i][k]).most_common()]
                  for k in out[i]}

    if heur:
        with open(f"loanpy/{heur}", "r") as f:
            he = json.load(f)
        return {k: (list(dict.fromkeys(out[k] + he[k])) if k in out else he[k])
                for k in he}

    return out

def get_heur(n=5):

    # read and define data
    ft = FeatureTable()
    ipa_all = list(ft.seg_dict)
    with open ("cldf/.transcription-report.json") as f:
        inv = json.load(f)["by_language"]["EAH"]["segments"]
    msr = Distance().weighted_feature_edit_distance

    # run the analysis
    heur = {ph: [j[1] for j in nsmallest(n,
        [(msr(ph, i), i) for i in inv])] for ph in ipa_all}

    # add heuristics for C V F B
    for cvfb, c, b in zip("CVFB", [1, -1, -1, -1], [0, 0, -1, 1]):
        cond = {"cons": c} if cvfb in "CV" else {"cons": c, "back": b}
        heur[cvfb] = [ph for ph in heur["ə"] if ft.word_fts(ph)[0].match(cond)]

    return heur

def uralign(left, right):
    """
    custom alignment for Hungarian-preHungarian
    """

    left, right = left.split(), right.split()
    # tag word initial & final cluster, only in left
    left[0], left[-1] = "#" + left[0], left[-1] + "#"

    # go sequentially and squeeze the leftover together to one suffix
    # e.g. "a,b","c,d,e,f,g->"a,b,-#","c,d,efg
    diff = abs(len(right) - len(left))
    if len(left) < len(right):
        left += ["-#"]
        right = right[:-diff] + ["".join(right[-diff:])]
    elif len(left) > len(right):
        left = left[:-diff] + ["+"] + ["".join(left[-diff:])]
    else:
        left, right = left + ["-#"], right + ["-"]

    return f'{" ".join(left)}\n{" ".join(right)}'
