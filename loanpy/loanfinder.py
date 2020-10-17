import pandas as pd
from tqdm import tqdm
from collections import Counter
from lingpy import ipa2tokens
import codecs #to write utf-8 encoded files
from gensim.models import KeyedVectors
import os
from nltk.corpus import wordnet as wn #for geting synonymsfrom nltk.corpus import wordnet as wn #for geting synonyms
import matplotlib

datafolder=os.path.dirname(os.path.abspath(__file__))+r"\data"
os.chdir(datafolder) #change to folder "data"

dfgot=pd.read_csv("dfgot.csv",encoding="utf-8",usecols=["substi"])
dfgot.substi=dfgot.substi.str.split(", ")
dfgot=dfgot.explode("substi")

model=[]
semsimdict={}
dfallmatches=pd.DataFrame(columns=["substi","hun_idx","got_idx"])
dfuniqmatches=pd.DataFrame(columns=["substi","hun_idx","got_idx"])

cns="jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC"
vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅"

def flatten(mylist):
    return [item for sublist in mylist for item in sublist]

def word2struc(ipaword): #in: "baba" out: "CVCV"
    return "".join([(lambda x: "C" if (x in cns) else ("V" if (x in vow) else ""))(i) for i in ipa2tokens(ipaword, merge_vowels=False, merge_geminates=False)])

def ipa2clusters(ipaword): 
    return [j for j in "".join([(lambda x: "€"+x+"€" if x[0] in vow else x)(i) for i in ipa2tokens(ipaword, merge_vowels=True)]).split("€") if j]

def matchfinder(regexstr,index,method="allmatches"):
    dftotal=dfgot
    dftotal["hun_idx"]=dftotal["substi"].replace(regexstr,index,regex=True)
    dftotal["got_idx"]=dftotal.index
    dftotal=dftotal[dftotal.hun_idx.astype(str).str.isdigit()]
    
    if method=="allmatches":
        global dfallmatches
        dfallmatches=dfallmatches.append(dftotal)
    
    if method=="uniquematches":
        global dfuniqmatches
        dfuniqmatches=dfuniqmatches.append(dftotal.drop_duplicates("got_idx"))

def statistics(dfstat, caller, method, name=""):
    lendf=len(dfstat)
    subuniq=len(set(dfstat.substi))
    substruc=Counter(dfstat.substi.apply(word2struc)).most_common()
    subclust=Counter(flatten(dfstat.substi.apply(ipa2clusters).tolist())).most_common()
    subphons=Counter(flatten(dfstat.substi.apply(ipa2tokens).tolist())).most_common()
    subinit=Counter(flatten(dfstat.substi.apply(ipa2clusters).str.slice(stop=1).tolist())).most_common()
    subfin=Counter(flatten(dfstat.substi.apply(ipa2clusters).str.slice(start=-1).tolist())).most_common()
    submed=Counter(flatten(dfstat.substi.apply(ipa2clusters).str.slice(start=1,stop=-1).tolist())).most_common()

    hununiq=len(set(dfstat.hun_idx))
    hunmat=Counter(dfstat.hun_lemma)
    hunyear=Counter(dfstat.hun_year)
    hunpos=Counter(dfstat.hun_pos)
    hunorig=Counter(dfstat.hun_orig_en)
    hunsuff=Counter(dfstat.hun_suffix)
    
    gotuniq=len(set(dfstat.got_idx))
    gotmat=Counter(dfstat.got_lemma)
    gotpos=Counter(dfstat.got_pos)
    gotocc=Counter(dfstat.got_occurences)
    gotcert=Counter(dfstat.got_certainty)
    gotrecon=Counter(dfstat.got_reconstructedness)
    
    content="""
    Number of matches: {}\n\n
    Substitutes:\n
    \tUniques: {}\n
    \tStructure distribution: {}\n
    \tConsonant/Vowel Cluster distribution: {}\n
    \tConsonant/Vowel distribution: {}\n
    \tDistribution of word initial clusters: {}\n
    \tDistribution of word final clusters:  {}\n
    \tDistribution of medial clusters:  {}\n
    Hungarian words:\n
    \tUniques: {}\n
    \tDistribution of matches (index in zaicz.csv : nr of matches): {}\n
    \tDistribution of year of first appearance: {}\n
    \tDistribution of part of speach (nltk-tags: n=noun,v=verb,a=adjective,r=other): {}\n
    \tDistribution of origin: {}\n
    \tSuffixes: {}\n\n   
    Gothic words:\n
    \tUniques: {}\n
    \tDistribution of matches: {}\n
    \tUniques' part of speach (pos) distribution (nltk-tags: n=noun,v=verb,a=adjective,r=other): {}\n
    \tUniques' occurences distribution: {}\n
    \tUniques' certainty distribution: {}\n
    \tUniques' reconstructedness distribution: {}\n
    """.format(lendf,subuniq,substruc,subclust,subphons,subinit,subfin,submed,hununiq,hunmat,hunyear,hunpos,hunorig,hunsuff,gotuniq,gotmat,gotpos,gotocc,gotcert,gotrecon)
    
    f = codecs.open(datafolder+r"\results\metadata_"+method+"_"+caller+name+".txt", "w", "utf-8")
    f.write(content)
    f.close()

def semsim(hun_en, got_en, nvarhun='n,v,a,r', nvargot='n,v,a,r'):
    """
    "Download Vectors from:\nhttps://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit\n"\
    "More infos: https://code.google.com/archive/p/word2vec/")
    """
    topsim=-1 #score of the most similar word pair
    hwrd='KeyError gensim'
    gwrd='KeyError gensim'
    
    global model
    if model==[]:
        model = KeyedVectors.load_word2vec_format(datafolder+r"\GoogleNews-vectors-negative300.bin", binary=True)

    if isinstance(hun_en, float) or isinstance(got_en, float):
        return float(-1)
    
    if hun_en+', '+got_en in semsimdict:
        return semsimdict[hun_en+', '+got_en][0]
    else: 
        hun=hun_en.split(', ') #get names of synsets, if they match the wordtype, flatten list, remove duplicates
        got=got_en.split(', ')
        hun=list(dict.fromkeys(flatten([hun]+[y.lemma_names() for y in [wn.synsets(x) for x in hun][0] if y.pos() in nvarhun])))[:20]
        got= list(dict.fromkeys(flatten([got]+[y.lemma_names() for y in [wn.synsets(x) for x in got][0] if y.pos() in nvargot])))[:20]
       
        for j in hun:
            for i in got: #calculate semantic similarity of all pairs0
                try:
                    modsim=model.similarity(j, i)
                except KeyError: #if word is not in KeyedVecors of gensim continue
                    continue                                   
                if modsim>topsim: #replace topsim if word pair is more similar than the current topsim, 
                    topsim=modsim
                    gwrd=i
                    hwrd=j

        if hwrd !='KeyError gensim':
            semsimdict.update({hun_en+', '+got_en : [topsim,hwrd,gwrd]})
        return topsim
    
def findmatches(df1="zaicz.csv" ,method="allmatches", name=""):
    zaicz=pd.read_csv(df1,encoding="utf-8",usecols=["regexroot","orig_tag","hun_year"])
    zaiczinfo=pd.read_csv(df1,encoding="utf-8",usecols=["hun_lemma","hun_year","hun_orig_en","hun_pos","hun_suffix"])
    dfgotinfo=pd.read_csv(datafolder+"\dfgot.csv",encoding="utf-8",usecols=["got_lemma","got_pos", "got_occurences", "got_certainty","got_reconstructedness"])

    #filter zaicz.csv
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

    tqdm.pandas(desc="Searching for phonetic matches")
    zaicz.progress_apply(lambda x: matchfinder(x.regexroot, x.ind, method), axis=1)
    
    if method=="allmatches":
        dfstat=dfallmatches
    elif method=="uniquematches":
        dfstat=dfuniqmatches
        
    dfstat=dfstat.merge(zaiczinfo, left_on="hun_idx", right_index=True)
    dfstat=dfstat.merge(dfgotinfo, left_on="got_idx", right_index=True)      
    statistics(dfstat, caller="precutoff", method=method, name=name)
    
    dfstat=dfstat[["hun_idx","substi","got_idx"]] #keep only these two cols
    dfstat=dfstat.merge(pd.read_csv(datafolder+"\zaicz.csv",encoding="utf-8",usecols=["hun_pos","hun_en"]), left_on="hun_idx", right_index=True)
    dfstat=dfstat.merge(pd.read_csv(datafolder+"\dfgot.csv",encoding="utf-8",usecols=["got_pos","got_en"]), left_on="got_idx", right_index=True)

    tqdm.pandas(desc="Calculating semantic similarity")
    dfstat["semsim"]=dfstat.progress_apply(lambda x: semsim(x.hun_en, x.got_en, x.hun_pos, x.got_pos), axis=1)
    
    dfstat["semsim"].apply(lambda x: float(x) if (x!="") else float(0)).plot.kde().get_figure().savefig(datafolder+r"\results\semsimdistr_"+method+name+"_precutoff.pdf")
    
    dfstat=dfstat.sort_values(by="semsim",ascending=False).head(200000) #sort, cut off
    
    dfstat=dfstat.drop(["hun_pos", "got_pos"], axis=1)
    dfstat=dfstat.merge(zaiczinfo, left_on="hun_idx", right_index=True)
    dfstat=dfstat.merge(dfgotinfo, left_on="got_idx", right_index=True)    
    statistics(dfstat, "cutoff", method=method, name=name)
    dfstat["semsim"].apply(lambda x: float(x) if (x!="") else float(0)).plot.kde().get_figure().savefig(datafolder+r"\results\semsimdistr_"+method+name+"_cutoff.pdf")
    
    dfstat=dfstat.merge(pd.read_csv(datafolder+r"\dfgot.csv",encoding="utf-8",usecols=["got_links"]), left_on="got_idx", right_index=True)   
    dfstat.to_csv(datafolder+r"\results\cutoff_"+method+name+".csv",encoding="utf-8",index=False)