import os

import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from loanpy.helpers import flatten

path=os.path.dirname(os.path.abspath(__file__))+r"\data"
os.chdir(path)

dfmatches=pd.DataFrame(columns=["substi","hun_idx"])

dfgot=pd.read_csv("dfgot.csv",encoding="utf-8",usecols=["substi"])
dfgot.substi=dfgot.substi.str.split(", ")
dfgot=dfgot.explode("substi")

zaicz=pd.read_csv("zaicz.csv",encoding="utf-8",usecols=["regexroot","orig_tag","hun_year"])
zaicz["ind"]=zaicz.index
zaicz=zaicz[~zaicz.regexroot.str.contains("not old/Gothic")]
ztime=zaicz[zaicz.orig_tag=="U"]
ztime=sorted(list(set(ztime["hun_year"])))
yearcap=ztime[-1]
yearfloor=ztime[0]
cond1=zaicz.hun_year>=yearfloor
cond2=zaicz.hun_year<=yearcap
zaicz=zaicz[cond1&cond2]
zaicz=zaicz.drop(["orig_tag","hun_year"],axis=1)

model = KeyedVectors.load_word2vec_format(path+r"\GoogleNews-vectors-negative300.bin", binary=True)

def matchfinder(regexstr,index):
    dftotal=dfgot
    dftotal["hun_idx"]=dftotal["substi"].replace(regexstr,index,regex=True)
    dftotal=dftotal[dftotal.hun_idx.astype(str).str.isdigit()]
    global dfmatches
    dfmatches=dfmatches.append(dftotal[~dftotal.index.duplicated()])
    
def getsynonyms(enword, pos="n"):
    enword=enword.split(', ')
    return list(dict.fromkeys(flatten([enword]+[y.lemma_names() for y in [wn.synsets(x) for x in enword][0] if y.pos() in pos])))[:20]
        
def semsim(hun_en, got_en, nvarhun='nvar', nvargot='nvar'):
    """
    "Download Vectors from:\nhttps://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit\n"\
    "More infos: https://code.google.com/archive/p/word2vec/")
    """
    if isinstance(hun_en, float) or isinstance(got_en, float): #missing translations = empty cells = nans = floats
        return -1
    
    else: #get names of synsets, if they match the wordtype, flatten list, remove duplicates
        hun=getsynonyms(hun_en)
        got=getsynonyms(got_en)
        
        topsim=-1 #score of the most similar word pair
        for hun_syn in hun:
            for got_syn in got: #calculate semantic similarity of all pairs0
                try:
                    modsim=model.similarity(hun_syn, got_syn)
                except KeyError: #if word is not in KeyedVecors of gensim continue
                    continue                                   
                if modsim>topsim: #replace topsim if word pair is more similar than the current topsim, 
                    topsim=modsim                 
        return topsim
      
def findmatches(name=""):
    tqdm.pandas(desc="Searching for phonetic matches")
    zaicz.progress_apply(lambda x: matchfinder(x.regexroot, x.ind), axis=1)

    global dfmatches #add cols for semsim()
    dfmatches=dfmatches.merge(pd.read_csv("zaicz.csv",encoding="utf-8",usecols=["hun_pos","hun_en"]), left_on="hun_idx", right_index=True)
    dfmatches=dfmatches.merge(pd.read_csv("dfgot.csv",encoding="utf-8",usecols=["got_pos","got_en"]), left_index=True, right_index=True)
    tqdm.pandas(desc="Calculating semantic similarity")
    dfmatches["semsim"]=dfmatches.progress_apply(lambda x: semsim(x.hun_en, x.got_en, x.hun_pos, x.got_pos), axis=1)
    dfmatches=dfmatches.sort_values(by="semsim",ascending=False).head(200000) #sort, cut off
    dfmatches = dfmatches.drop(["hun_pos","hun_en","got_pos","got_en"], axis=1)
    
    #merge with original dataframes
    dfmatches=dfmatches.merge(pd.read_csv("dfgot.csv",encoding="utf-8"), left_index=True, right_index=True)
    dfmatches=dfmatches.merge(pd.read_csv("zaicz.csv",encoding="utf-8"), left_on="hun_idx", right_index=True)
    dfmatches.to_csv(os.path.dirname(os.path.abspath(__file__))+r"\data\results\matches"+name+".csv",encoding="utf-8",index=False)