#I. add regex col to zaicz.csv
    #1. define helper functions
        #a. flatten()
        #b. ipa2clusters()
    #1. extract and define all constraints
    #2. extract soundchanges from uralonet.csv and write them to file (new layout, regex)
        #consider constraint: which sounds appear in filledout.csv
    #3. add regex-col to zaicz.csv (=generator)
    #4. get list of how to substitute got. clusters, consider constraints
    #5. generate substitutions (constraints!) and add to substitutions-col
    
#II. add substitution list col to dfgot.csv

import os
import ast
import pandas as pd
from lingpy import ipa2tokens #ipa2clusters(), getconstraints()
from itertools import combinations #for deletion()
from itertools import product #for deletion()
from collections import defaultdict #for substitute()
from itertools import count #for substitute()

cns="jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC"
vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅"

dfsoundchange=pd.DataFrame()
affix=[]
substiphons=""
dfsubsti=pd.DataFrame()
sudict={}
substidict={}

rootdir=os.path.dirname(os.path.abspath(__file__))
os.chdir(rootdir+r"\data\generator") #change to folder "generator"

#I.

#1. define helper functions

#a. flatten nested list
def flatten(mylist):
    return [item for sublist in mylist for item in sublist]

def word2struc(ipaword): #in: "baba" out: "CVCV"
    return "".join([(lambda x: "C" if (x in cns) else ("V" if (x in vow) else ""))(i) for i in ipa2tokens(ipaword, merge_vowels=False, merge_geminates=False)])

def ipa2clusters(ipaword): 
    return [j for j in "".join([(lambda x: "€"+x+"€" if x[0] in vow else x)(i) for i in ipa2tokens(ipaword, merge_vowels=True)]).split("€") if j]

def list2regex(sclist):
    return "" if sclist == ["0"] else ("("+"".join([i+"|" for i in sclist])[:-1].replace("|0","").replace("0|","")+")"+"?" if "0" in sclist and sclist != ["0"]\
                                       else "("+"".join([i+"|" for i in sclist])[:-1]+")")
def extractsoundchange(reflex,root): #requires an ipastring as input, use epitran and uewscrape2ipa to transcribe your data
    global substiphons
    if substiphons=="":
        global dfsubsti
        if dfsubsti.empty:
            dfsubsti=pd.read_csv("filledout.csv",encoding="utf-8")
        substiphons=set(dfsubsti["substitution"].str.split(", ").explode())
    
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

    global affix
    for ind,i in enumerate(root):
        if i=="0" and reflex[ind] not in affix:
            affix.append(reflex[ind])
    
    dfrootrefl=pd.DataFrame({"reflex":reflex,"root":root})
    dfrootrefl=dfrootrefl[dfrootrefl.root.apply(lambda x: all(elem in substiphons or elem=="0" for elem in ipa2tokens(x)))] #root has no hashtags to remove
    
    global dfsoundchange
    dfsoundchange=dfsoundchange.append(dfrootrefl)
    
def uralonet2scdict(name=""):
    dfural=pd.read_csv("uralonet.csv",encoding="utf-8")
    global dfsoundchange
    dfsoundchange=pd.DataFrame() #make sure this df is really empty before you start the process...
    dfural.apply(lambda x: extractsoundchange(x["New"], x["Old"]), axis=1)
    dfsc=dfsoundchange.groupby("reflex")["root"].apply(lambda x: list2regex(sorted(set(x)))).reset_index()
    soundchangedict=str(dict(zip(dfsc.reflex,dfsc.root)))
    with open("soundchangedict"+name+".txt","w",encoding="utf-8") as data:
        data.write(soundchangedict)

#1. extract and define constraints
def setconstraints(timelayer): #later: add timelayers and turn global variables to global dictionaries
    os.chdir(os.path.dirname(os.path.abspath(__file__))+r"\data\generator")
    
    dfural=pd.read_csv("uralonet.csv",encoding="utf-8")
    scdictfile=open("soundchangedict.txt", "r", encoding="utf-8")
    global scdict #reconstruct() will need this later 
    scdict = ast.literal_eval(scdictfile.read())
    scdictfile.close()

    substidictfile=open("substidict.txt", "r", encoding="utf-8")
    global substidict
    substidict = ast.literal_eval(substidictfile.read())
    substidictfile.close()

    dfural=dfural[dfural.Lan==timelayer]
    
    global allowedclust
    allowedclust=set(flatten(dfural["Old"].apply(ipa2clusters).tolist()))
    
    global maxclustlen
    tokenizedclust=list(map(ipa2tokens,allowedclust))
    maxclustlen=sorted(map(len,tokenizedclust),reverse=True)[0] #do ipa2tokens first   
    #later: set a threshold: cluster has to appear more than 4 times
    
    global wordstruc
    wordstruc=set(dfural["old_struc"])
    
    global maxnrofclusters
    maxnrofclusters=sorted(map(len,map(ipa2tokens,wordstruc)),reverse=True)[0]
        
def reconstruct(ipaword):
    ipaword=ipa2clusters(ipaword)
    ipaword[0]="#"+ipaword[0]
    ipaword[-1]=ipaword[-1]+"#"
    return ", ".join([i for i in ipaword if i not in list(scdict.keys())+affix])+" not old/Gothic" if not all(elem in list(scdict.keys())+affix for elem in ipaword) else "^(j|m|s|w)?"+"".join([scdict[i] for i in ipaword])+"(e|æ|ɑ|je|jkɑ|ŋæ|we|ke|me|ŋe|ele)?$"
    
def addreconstr(name=""):
    zaicz=pd.read_csv("zaicz.csv",encoding="utf-8")
    zaicz["regexroot"]=zaicz.wordipa.apply(reconstruct)
    zaicz.to_csv(rootdir+"\data\zaicz"+name+".csv",encoding="utf-8",index=False)
    
##########################################################################################################
#II.

def deletion(cluster):
    global sudict
    if sudict=={}:
        dfsubsti=pd.read_csv("filledout.csv",encoding="utf-8")
        sudict=dict(zip(dfsubsti.to_substitute, dfsubsti.substitution))
    
    cluster = ipa2tokens(cluster,merge_vowels=False,merge_geminates=False) #"abcd"-> ["a","b","c","d"]
    allsubsti = list(map(sudict.get, cluster)) #["a","b"]->["x, y", "z"]
    allsubsti = flatten([i.split(", ") for i in allsubsti]) #["x, y", "z"] -> ["x","y","z"]
    
    for index in range(2,maxclustlen+1):
        for clu in combinations(cluster, index): #abcd->ab,ac,ad,bc,bd,cd (tuples)
            clu=list(map(sudict.get, clu)) #(a,b) -> ('x, y','z') get substitutions from dict
            clu=[i.split(", ") for i in clu] #turn substitutions to lists: ('x, y','z') -> ([x,y],[z])
            for j in product(*clu): #([x,y],[z]) -> [(x,z),(y,z)] list of tuples
                allsubsti.append("".join(list(j))) #tuple->list->string (x, z)->[x,z]->"xz"
    
    return ", ".join(list(dict.fromkeys([x for x in allsubsti if x in allowedclust])))

def getsubstis(name=""): 
    dfgot=pd.read_csv("dfgot.csv",encoding="utf-8")
    dfsubsti=pd.read_csv("filledout.csv",encoding="utf-8")
    clusters = sorted(set(flatten(dfgot.got_ipa.apply(ipa2clusters).tolist()))) #not the same as substiphons
    clustlist=[j for j in clusters if j not in dfsubsti.to_substitute.tolist()]
    dfclust=pd.DataFrame({"to_substitute": clustlist, "substitution": list(map(deletion,clustlist))}) #list of clusters
    dfsubsti=dfsubsti.append(dfclust)
    substidict=str(dict(zip(dfsubsti.to_substitute,dfsubsti.substitution)))
    with open("substidict"+name+".txt","w",encoding="utf-8") as data:
        data.write(substidict)
    
def substitute(gotipa):      
    args=[]
    substilist=[]
    gotipa=ipa2clusters(gotipa)
    
    if len(gotipa)>maxnrofclusters:
        return "too long"

    idxdict = defaultdict(count(1).__next__) #same letter same substitution. a->v/w: acab -> vxvy/wxwy and not vxwy/wxvy
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

def addsubsti(name=""):
    dfgot=pd.read_csv("dfgot.csv",encoding="utf-8")
    dfgot["substi"]=dfgot.got_ipa.apply(substitute)
    dfgot.to_csv(rootdir+r"\data\dfgot"+name+".csv",encoding="utf-8",index=False)