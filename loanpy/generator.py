import os
import ast
from collections import defaultdict
from itertools import count
from itertools import product

from lingpy import ipa2tokens
import pandas as pd

from loanpy.helpers import ipa2clusters
from loanpy.helpers import word2struc

os.chdir(os.path.dirname(os.path.abspath(__file__))+r"\data\generator")

with open("soundchangedict.txt", "r", encoding="utf-8") as f:         
    scdict = ast.literal_eval(f.read())
with open("substidict.txt", "r", encoding="utf-8") as f:
    substidict = ast.literal_eval(f.read())
dfural=pd.read_csv("uralonet.csv",encoding="utf-8")
dfural=dfural[dfural.Lan=="U"]
wordstruc=set(dfural["old_struc"])
maxnrofclusters=sorted(map(len,map(ipa2tokens,wordstruc)))[-1]
    
oldprefix="()?"
oldsuffix="()?"
if "#0" in scdict:
    oldprefix=scdict["#0"]+"?"
if "0#" in scdict:
    oldsuffix="|".join([scdict["0#"][:-1],scdict["00#"][1:]])+"?"
    
def reconstruct(ipaword):
    ipaword=ipa2clusters(ipaword.replace("ː",""))
    ipaword[0]="#"+ipaword[0]
    ipaword[-1]=ipaword[-1]+"#"
    return ", ".join([i for i in ipaword if i not in list(scdict.keys())])+" not old/Gothic" if not all(elem in list(scdict.keys()) for elem in ipaword) else "^"+oldprefix+"".join([scdict[i] for i in ipaword])+oldsuffix+"$"

def substitute(gotipa):                            
    args=[]
    substilist=[]
    gotipa=ipa2clusters(gotipa)
    
    if len(gotipa)>maxnrofclusters:
        return "too long"

    idxdict = defaultdict(count(1).__next__) #same letter same substitution. e.g. a->v/w: acab -> vxvy/wxwy and not vxwy/wxvy
    idxlist = [idxdict[cluster]-1 for cluster in gotipa] #'acab' -> idxlist=[0,1,0,2]
    gotipa = list(dict.fromkeys(gotipa)) #'acab'->'acb'

    for i in gotipa:
        args.append(substidict[i].split(", ")) #'acb'-> [['v','w'],['x'],['y','z']]
    for subst in product(*args): #[['v','w'],['x'],['y','z']] -> vxy,vxz,wxy,wxz
        mirror=[] #vxy,vxz,wxy,wxz -> vxvy,vxvz,wxwy,wxwz 
        for i in idxlist:
            mirror+=subst[i]
            mirror="".join(mirror)
        substilist.append(mirror)
    return ", ".join([i for i in substilist if word2struc(i) in wordstruc])

def addproto(dataframe, inputcol, function, outputcol, name):
    dataframe[outputcol]=dataframe[inputcol].apply(function)
    dataframe.to_csv(os.path.dirname(os.path.abspath(__file__))+r"\data\\"+name,encoding="utf-8",index=False)