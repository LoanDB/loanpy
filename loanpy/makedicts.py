import os
from itertools import combinations
from itertools import product

import pandas as pd
from lingpy import ipa2tokens

from loanpy.helpers import ipa2clusters
from loanpy.helpers import list2regex
from loanpy.helpers import flatten

os.chdir(os.path.dirname(os.path.abspath(__file__))+r"\data\generator")

cns="jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC"
vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅" 
dfural=pd.read_csv("uralonet.csv",encoding="utf-8")
dfsubsti=pd.read_csv("filledout.csv",encoding="utf-8")
dfgot=pd.read_csv("dfgot.csv",encoding="utf-8")
dfsoundchange=pd.DataFrame()
dfural=dfural[dfural.Lan=="U"]
substiphons=set(dfsubsti["substitution"].str.split(", ").explode())
sudict=dict(zip(dfsubsti.to_substitute, dfsubsti.substitution))
allowedclust=set(flatten(dfural["Old"].apply(ipa2clusters).tolist()))
tokenizedclust=list(map(ipa2tokens,allowedclust))
maxclustlen=sorted(map(len,tokenizedclust),reverse=True)[0]

def extractsoundchange(reflex,root): #requires an ipastring as input    
    reflex=ipa2clusters(reflex.replace("ː",""))
    root=ipa2clusters(root.replace("ː",""))

    if reflex[0] in vow and root[0] in cns:
        reflex=["0"]+reflex
    elif reflex[0] in cns and root[0] in vow:
        root=["0"]+root

    diff=abs(len(root)-len(reflex)) # "a,b", "c,d,e,f,g" -> "a,b,000","c,d,efg"
    if len(reflex)<len(root):
        reflex+=["0"*diff]
        root=root[:-diff]+["".join(root[-diff:])]
    elif len(reflex)>len(root):
        root+=["0"*diff]
        reflex= reflex[:-diff]+["".join(reflex[-diff:])]

    reflex[0]="#"+reflex[0]
    reflex[-1]=reflex[-1]+"#"

    dfrootrefl=pd.DataFrame({"reflex":reflex,"root":root})
    dfrootrefl=dfrootrefl[dfrootrefl.root.apply(lambda x: all(elem in substiphons or elem=="0" for elem in ipa2tokens(x)))] #root has no hashtags to remove
    global dfsoundchange
    dfsoundchange=dfsoundchange.append(dfrootrefl)
    
def uralonet2scdict(name=""):    
    dfural.apply(lambda x: extractsoundchange(x["New"], x["Old"]), axis=1)
    dfsc=dfsoundchange.groupby("reflex")["root"].apply(lambda x: list2regex(sorted(set(x)))).reset_index()
    soundchangedict=str(dict(zip(dfsc.reflex,dfsc.root)))
    
    with open("soundchangedict"+name+".txt","w",encoding="utf-8") as data:
        data.write(soundchangedict)
        
def deletion(cluster):       
    cluster = ipa2tokens(cluster,merge_vowels=False,merge_geminates=False) #"abcd"-> ["a","b","c","d"]
    allsubsti = list(map(sudict.get, cluster)) #["a","b"]->["x, y", "z"]
    allsubsti = flatten([i.split(", ") for i in allsubsti]) #["x, y", "z"] -> ["x","y","z"]
    
    for index in range(2,maxclustlen+1): #combinations ONLY for 2 or more phonemes 
        for clu in combinations(cluster, index): #abcd->ab,ac,ad,bc,bd,cd (tuples)
            clu=list(map(sudict.get, clu)) #(a,b) -> ('x, y','z') get substitutions from dict
            clu=[i.split(", ") for i in clu] #turn substitutions to lists: ('x, y','z') -> ([x,y],[z])
            for j in product(*clu): #([x,y],[z]) -> [(x,z),(y,z)] list of tuples
                allsubsti.append("".join(list(j))) #tuple->list->string (x, z)->[x,z]->"xz"
    
    return ", ".join(list(dict.fromkeys([x for x in allsubsti if x in allowedclust])))

def substiclust(name=""):
    global dfsubsti
    clusters = sorted(set(flatten(dfgot.got_ipa.apply(ipa2clusters).tolist()))) #get all possible gothic clusters #not the same as substiphons
    clusters = [clu for clu in clusters if clu not in dfsubsti.to_substitute.tolist()] #subtract single phonemes (non-clusters), b/c combinations() can't handle them
    dfclust=pd.DataFrame({"to_substitute": clusters, "substitution": list(map(deletion,clusters))}) #col1: actual clusters, col2: their possible substitutions
    dfsubsti=dfsubsti.append(dfclust) #merge df of single phonemes and clusters
    substidict=str(dict(zip(dfsubsti.to_substitute,dfsubsti.substitution))) #transform csv to machine-readable dictionary
    with open("substidict"+name+".txt","w",encoding="utf-8") as data:
        data.write(substidict)