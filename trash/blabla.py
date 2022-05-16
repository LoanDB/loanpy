import numpy as np
from functools import wraps
import networkx as nx
from ipatok import tokenise, ipa
from unittest.mock import patch, call
from numpy import array

from loanpy import helpers as hp

equilist = []

def getequilist(func):

  return func

def getequilist(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        result = method(*args, **kwargs)
        global equilist
        equilist.append(result)
        return result
    return wrapped

np.array_equiv = getequilist(np.array_equiv)

def tuples2editops(op_list, s1, s2):
    equilist = []
    s1, s2 = "#"+s1, "#"+s2
    out = []
    for i, todo in enumerate(op_list):
        if i == 0: #so that i-i won't be out of range
            continue
        #where does the arrow point?
        direction = np.subtract(todo, op_list[i-1])
        print(direction)

        if np.array_equiv(direction, [1, 1]): #if diagonal
            out.append(f"keep {s1[todo[1]]}")
        elif np.array_equiv(direction, [0, 1]): #if horizontal
            if i > 1: #if previous was verical -> substitute
                print("here1")
                if np.array_equiv(np.subtract(op_list[i-1], op_list[i-2]), [1, 0]):
                    out = out[:-1]
                    out.append(f"substitute {s1[todo[1]]} by {s2[todo[0]]}")
                    continue
            out.append(f"delete {s1[todo[1]]}")
        elif np.array_equiv(direction, [1, 0]): #if vertical
            if i > 1: #if previous was horizontal -> substitute
                print("here2")
                if np.array_equiv(np.subtract(op_list[i-1], op_list[i-2]), [0, 1]):
                    print(np.subtract(op_list[i-1], op_list[i-2]), [0, 1])
                    out = out[:-1]
                    out.append(f"substitute {s1[todo[1]]} by {s2[todo[0]]}")
                    continue
            out.append(f"insert {s2[todo[0]]}")

    return out

def getallpaths():

    G = nx.DiGraph()
    G.add_weighted_edges_from(  [((2, 2), (2, 1), 100),
                                ((2, 2), (1, 2), 49),
                                ((2, 2), (1, 1), 0),
                                ((2, 1), (2, 0), 100),
                                ((2, 1), (1, 1), 49),
                                ((2, 0), (1, 0), 49),
                                ((1, 2), (1, 1), 100),
                                ((1, 2), (0, 2), 49),
                                ((1, 1), (1, 0), 100),
                                ((1, 1), (0, 1), 49),
                                ((1, 0), (0, 0), 49),
                                ((0, 2), (0, 1), 100),
                                ((0, 1), (0, 0), 100)])
    return list(nx.all_shortest_paths(G, (2,2),(0,0), weight="weight"))

#print(getallpaths())
#print(tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)], "ló", "hó"))
#print(equilist)

def clusterise(string, strict=False, replace=False,
               diphthongs=False, tones=False, unknown=False, merge=None):
    def grouped(ipalist):
        """Merge subsequent consonants and vowels to clusters."""
        tmp = []
        it = iter(ipalist)
        nextit = next(it)
        for char in ipalist:
            charcv = any(ipa.is_vowel(i) for i in char)
            while any(ipa.is_vowel(i) for i in nextit) != charcv:
                yield ''.join(tmp)
                tmp = []
                nextit = next(it)
            tmp.append(char)
        yield ''.join(tmp)

    args = list(locals().values())[:-1]
    print(args)
    return [i for i in grouped(tokenise(*args)) if i]

#print(clusterise("gag", strict=True))
class Adrcbla:
    def __init__(self):
        pass

def getdiff(self, sclistlist, connector, ipa):
    difflist = []
    for idx, sclist in enumerate(sclistlist):

        # exception 1:
        if len(sclist) == 1:  # if list has reached the end
            difflist.append(float("inf"))  # indicate that it cant be moved
            continue

        firstsc = self.nsedict.get(ipa[idx] + connector + sclist[0], 0)
        nextsc = self.nsedict.get(ipa[idx] + connector + sclist[1], 0)

        #exception 2:
        if firstsc == 0 and nextsc == 0: #if both soundchanges have zero examples
            for i, sl in enumerate(sclistlist):  # look at the other sounds in the word
                if len(sl) > 1:  #if there is at least one that hasnt reached the end
                    if self.nsedict.get(ipa[i]+connector+sl[0],0) > 0:  # AND has more than 0 examples
                        difflist.append(float("inf"))  # "dont move (the sound with 0 examples) in readsc()"
                    # instead give (the one that hasnt reached the end and has more than zero examples) a chance for later
                        continue

        difflist.append(firstsc - nextsc)

    return difflist

def test_getdiff():
    A = Adrcbla()
    A.nsedict = {"k<k": 2, "k<c": 1, "i<e": 2, "i<o":1}
    sclistlist = [["k", "c"], ["e", "o"], ["k", "c"], ["e", "o"]]
    assert getdiff(self=A, sclistlist=sclistlist, connector="<",
                   ipa=["k","i","k","i"]) == [1, 1, 1, 1]

"""
Contains the Adrc class to adapt or reconstruct words
"""

from ast import literal_eval
from itertools import count, product, cycle
from math import prod
from string import ascii_letters
from operator import itemgetter

from tqdm import tqdm

from loanpy.helpers import (Etym, apply_edit, editops, editdistancewith2ops,
                            list2regex)
from loanpy.qfysc import Qfy

def read_scdictlist(scdictlist, dictnr=0):
    if scdictlist is None: return None
    with open(scdictlist, "r", encoding="utf-8") as f:
        return literal_eval(f.read())[dictnr]

def brk(listlist, howmany):
    # as soon as the product is bigger or equal howmany, activate break by returning True
    return howmany <= prod([len(scl) for scl in listlist]) #-1 for "$"

def movesc(sclistlist, whichsound, out):
    print("in:", sclistlist, out)
    out[whichsound].append(sclistlist[whichsound][0]) # transfer the sound to out
    sclistlist[whichsound] = sclistlist[whichsound][1:]  # move input by 1

    print("out:", sclistlist, out)
    return sclistlist, out

class Adrc(Qfy):
    """
    Adapts or reconstructs words based on etymological and statistical data

    :param substi: name of the dictionary containing the sound substitutions. \
    Generated by loanpy.qfysc.Qfy().dfetymology2dict()
    :type substi: str, default="substi.txt"

    :param photct: name of the file containing the list of allowed phonotactic structures
    :type photct: str, default="phonotactics.txt"

    :param sndcng: name of the file containing sound changes
    :type sndcng: str, default="scdict.txt"

    :param se_or_edict: name of the file containing the sum of examples, or examples
    :type se_or_edict: str, default="sedict.txt"

    """
    def __init__(self, scdictlist=None, scdict2=None,
                 ptct_thresh=0, struc_inv=None, mode=None, formscsv=None):
        """inits Adrc with allowed phonotactics, dictionaries of sound substitution and sound changes """

        super().__init__(formscsv=formscsv, mode=mode,
                         ptct_thresh=ptct_thresh, struc_inv=struc_inv)

        self.scdict = read_scdictlist(scdictlist, dictnr=0)
        self.scdict_struc = read_scdictlist(scdictlist, dictnr=3)
        self.scdict_struc2 = read_scdictlist(scdict2)
        self.workflow = []

    def getdiff(self, sclistlist, ipa):
        difflist = []
        firstsclist = []
        for idx, sclist in enumerate(sclistlist):

            # exception 1:
            if len(sclist) == 1:  # if list has reached the end
                difflist.append(float("inf"))  # indicate that it cant be moved
                continue

            firstsc = self.nsedict.get(ipa[idx] + self.connector + sclist[0], 0)
            firstsclist.append(firstsc)
            nextsc = self.nsedict.get(ipa[idx] + self.connector + sclist[1], 0)

            difflist.append(firstsc - nextsc)

        # exception 2: avoid adding more sound changes with 0 examples
        if 0 in firstsclist and not all(0==i for i in firstsclist):
            difflist = [float("inf") if fsc==0 else diff for fsc, diff in zip(firstsclist, difflist)]

        return difflist

    def readsc(self, ipa, howmany=1):
        if isinstance(ipa, str):  # if str not tokenised yet
            ipa = self.tokenise(ipa)  # tokenise it

        sclistlist = [self.scdict[i] for i in ipa]  # grab the soundchanges
        if howmany >= prod([len(scl) for scl in sclistlist]): return sclistlist #<if howmany is bigger than the product

        sclistlist = [sclist+["$"] for sclist in sclistlist]
        out = [[i[0]] for i in sclistlist] #grab first sound changes for output
        sclistlist = [sclist[1:] for sclist in sclistlist]

        while not brk(out, howmany): #or brk2(sclistlist, 1)): #getdiff needs the next sound
            difflist = self.getdiff(sclistlist, ipa)  # which one should we move?
            minimum = min(difflist)  # the one that makes the least difference!
            indices = [i for i, v in enumerate(difflist) if v == minimum]  # woops, someimtes multiple values make the least difference
            if len(indices) == 1: #if only 1 element makes the least difference
                sclistlist, out = movesc(sclistlist, indices[0], out) # grab it
                continue # thats it

            difflist2 = difflist #but if multiple elements are the minimum...
            idxpool = cycle(indices) #...then cycle through them...
            while difflist2 == difflist and not brk(out, howmany):# or brk2(sclistlist)):
            #...until something changes about the differences they make
            # latest if a sound reaches the end of the list it turns to infinite
                sclistlist, out = movesc(sclistlist, next(idxpool), out)
                difflist2 = self.getdiff(sclistlist, ipa)

        return out

class AdrcMock:
    def __init__(self, tokenised=None, getdiff="abc"):
        self.tokenise_returns = tokenised
        self.getdiff_returns = iter(getdiff)
    def tokenise(self, word): return self.tokenise_returns
    def getdiff(self, sclistlist, ipa): return next(self.getdiff_returns)

#A = AdrcMock(getdiff=[[4, 5], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3], [4, 3]])
#A = Adrc()
#A.scdict = {"k": ["k", "h"], "i": ["e", "o"]}
#A.nsedict = {"k<k": 2, "k<h": 1, "i<e": 3, "i<o":1}
#A.connector="<"
#print(Adrc.readsc(self=A, ipa=["k", "i"], howmany=2000))

def test_tuples2editops():
    class ArrayEquivMock:
        def __init__(self):
            self.called_with = []
            self.returns = iter([False, True, False, False, True, True, True])
        def array_equiv_mock(self, arr1, arr2):
            self.called_with.append((arr1, arr2))
            return next(self.returns)

    #ae_mock = ArrayEquivMock()
    #hp.array_equiv = ae_mock.array_equiv_mock
    arraylist = [array([0, 1]), array([1, 0]), array([0, 1]), array([1, 1])]
    exp_npmock_call = [
    (array([0, 1]), array([1, 1])), #  first step diagonal? No.
    (array([0, 1]), array([0, 1])), #  first step horizontal? Yes.
    (array([1, 0]), array([1, 1])), #  second step diagonal? No.
    (array([1, 0]), array([0, 1])), #  second step horizontal? No.
    (array([1, 0]), array([1, 0])), #  second step vertical? Yes.
    (array([1, 0]), array([0, 1])), #  step before (first step) horizontal? Yes.
    (array([1, 1]), array([1, 1]))] #  third step diagonal? Yes.

    with patch("loanpy.helpers.subtract", side_effect=arraylist) as subtract_mock:
        with patch("loanpy.helpers.array_equiv", side_effect=[False, True, False, False, True, True, True]) as array_equiv_mock:
            assert tuples2editops([(0, 0), (0, 1), (1, 1), (2, 2)], "ló", "hó") == ['substitute l by h', 'keep ó']

            print(array_equiv_mock.mock_calls)
            for arraypair0, arraypair1 in zip(array_equiv_mock.call_args_list, exp_npmock_call):
                assert_array_equal(arraypair0[0], arraypair1[0])
                assert_array_equal(arraypair0[1], arraypair1[1])

    del arraylist

test_tuples2editops()
