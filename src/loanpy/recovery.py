"""
recover sound correspondences
"""
# TODO: write tests and documentation with ChatGPT

from collections import defaultdict, Counter

def qfy(table, heuristics=False):
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

    for k in out[0]:   # sort keys by freq
        out[0][k] = [i[0] for i in Counter(out[0][k]).most_common()]
    for k in out[3]:   # sort keys by freq
        out[3][k] = [i[0] for i in Counter(out[3][k]).most_common()]

    if heuristics:
        # 1. create heuristics file with helpers (see qfysc.py on github)
        # 2. json_read the file and merge the info
        pass

    return out
