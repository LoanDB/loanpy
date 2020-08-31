#loanpy module. Main function loan()
#(c) Viktor Martinovic 08.08.2020
#import os
#os.chdir(r'C:\Users\Viktor\OneDrive\PhD cloud\Vorgehensweisen\loanpy3') #directory where python works
import pandas as pd #to work with dataframes
from lingpy import ipa2tokens #posy etc needs this to split and identify consonants and vowels
import itertools #for shuffle function mainly
try: #try/except because these dictionaries are to be written by qfysc() first. So qfysc has to work without them.
    from uraloEdict import dictex #dictionary of all examples for every sound change, created with qfysc()
    from uraloSEdict import dictse #dictionary with sum of examples for every sound change, created with qfysc()
except  ModuleNotFoundError:
    pass
import re #to clean the translations in the gothic dataframe. Should actually be part of preprocessing
from nltk.corpus import wordnet as wn #for geting synonyms
import epitran #just to occasionally transcribe Hun words to ipa, since loan requires ipa as input
epi = epitran.Epitran('hun-Latn')
import codecs #to write dictionaries in qfysc()
import functools #for tuple to string
import operator #for tuple to string
from itertools import combinations #to shuffle the substituted clusters
from collections import Counter #to Count clusters
from collections import defaultdict #for sufy
from itertools import count #for sufy

forbidden=['nan','0.0','∅',"0","<NA>"] #shuffle needs this variable. Add sounds to this variable that can't be...
ipa2uewdict={'¨':'ȣ̈','ɑ':'a','æ':'ä','ð':'δ','ðʲ':'δ́','ɣ':'γ','lʲ':'ĺ',\
             'nʲ':'ń','ʃ':'š','y':'ü','θ':'ϑ','t͡ʃ':'č','ʃʲ':'ś','t͡ʃʲ':'ć'}
#check uewscrape2ipa()
semsimdict={'KeyError gensim, KeyError gensim': 'KeyError gensim, KeyError gensim, -1'}
wikipos='Abkürzung,Adj.,Adv.,Art.,Buchstabe,F.,Interj.,Konj.,LN.,M.,N.,Num.,ON.,Partikel,PN.,Präp.,Pron.,Sb.,V.,Wort,nan'
wikikeys=wikipos.split(',')
wikivalues=['r','a','r','r','r','n','r','r','n','n','n','r','n','r','n','r','r','n','v','n','nvar']
wikidict = dict(zip(wikikeys, wikivalues))

subsdict={}
model=[]

def ipacsvfunction(column): #extracts soundtypes from ipa.csv (panphon) and ads them to a string (=faster)
    try:
        ipacsv= pd.read_csv("ipa_table.csv", encoding='utf-8',sep=';') #list from panphon to tell if an IPA sound is a vowel
    except FileNotFoundError:
        return 'ipa_table.csv not found, download ipa_table.csv via https://pypi.org/project/panphon/0.1/'
    listyes=''
    listno=''
    for i,row in ipacsv.iterrows():
        if row[column] == '+':
            listyes+=row['ipa']
        elif row[column] == '-':
            listno+=row['ipa']
    return listyes, listno

Cns='jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ð' \
    'd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎr'\
    'r̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC' #ipacsvfunction('cons')[0]
Vow='ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅' #ipacsvfunction('cons')[1]
back='wɡkqɠɢʛq͡χɢ͡ʁxħɣʁʕχŋɴʀɫɑɤouɒɔəɘɵɞɜɨɯʊɐʌʍɹȣ' #ipacsvfunction('back')[0]
front='jcɖɟʄʈbb͡ddd̪pp͡ttt̪ɓɗc͡çd͡ʒt͡ʃɟ͡ʝb͡vd͡zd͡ʑp͡ft͡st͡ɕçʂʃʐʒʝfss̪vzz̪ðɸβθɕʑɳmnn̪ɲʎhll̪ɦɘɞɵʉaeiyæøœɛɪɶʏʔɥɻ¨'#ipacsvfunction('back')[1]

def l2s(s):  
    str1 = ""   
    return (str1.join(s))

def convertTuple(tup): #for shuffling the clusters
    str = functools.reduce(operator.add, (tup)) 
    return str

def ipa2uew(word): #converts ipa strings to clean uew
    for i in ipa2tokens(word, merge_vowels=False, merge_geminates=False): #load the corresponding csv and transcribe according to table
        if i in ipa2uewdict:
            word=word.replace(i,ipa2uewdict[i])
    return word

def uewscrape2ipa(word):
    for i in trs['UEW_webscraped'].tolist(): #load the corresponding csv and transcribe according to table
        if isinstance(i,str) and i in word:
            word=word.replace(i, trs.iloc[trs[trs['UEW_webscraped'] == str(i)].index[0],0])
    return word

def snd2type(ipasound):
    if str(ipasound) in Cns:
        return 'C'
    elif str(ipasound) in Vow:
        return 'V'
    elif str(ipasound[0]) in Cns: #important to have an extra elif for this (ʷ/ʲ/aspiration etc. vs. t͡ʃ etc.)
        return 'C'
    elif str(ipasound[0]) in Vow:
        return 'V'
    elif str(ipasound) =='0':
        return ''
    else:
        return '?'

def word2struc(ipaword):
    return l2s([snd2type(i) for i in ipa2tokens(ipaword, merge_vowels=False, merge_geminates=False)])

def sndclusterdf(df,columnipa): #misleading name: this is about phonotactics (like CVCCVVCV)
    dflist=df[columnipa].tolist()
    allowedclusters=set([word2struc(item) for item in dflist])
    return allowedclusters

def clustercount(df,column): #counts actual sound clusters (not CVCV)
    posylist=[]
    for index, row in df.iterrows():
        posx=posy(row[column])
        posylist+=posx
    clusterlist=list(sum(Counter(posylist).most_common(), ()))
    cltdf=pd.DataFrame(clusterlist,columns=['allowed'])
    #cltdf.to_csv(outputnamewithoutdotcsv+".csv", encoding="utf-8", index=False) #list to df
    
    return cltdf #df, every second row is relevant 
    
def settimelayer(layer):
    global uralin
    uralin=pd.read_csv("uraloipa.csv", encoding='utf-8', sep=';') #read input wordlist
    uralin=uralin[uralin.Lan == layer] #keep only Uralic timelayer, dynamise later
    global allowedstruc
    allowedstruc=sndclusterdf(uralin, 'Old') #get set of allowed clusters
    global dfgot
    try:
        dfgot=pd.read_csv("dfgot"+layer+".csv", encoding='utf-8')
        dfgot=dfgot[dfgot['substi_struc'].isin(allowedstruc)] #keep only words with allowed structure in dfgot (ca.140K)
    except FileNotFoundError:
        print('dfgot'+layer+'.csv not found')
        pass
    try:
        global SCin
        SCin={}
        SCincsv = pd.read_csv("scingot"+layer+".csv", encoding='utf-8')
        for column in SCincsv:
            SCin[column]=[x for x in SCincsv[column] if str(x) not in forbidden]
    except FileNotFoundError:
        print("scingot"+layer+".csv not found")
        pass
        
    global timelayer
    timelayer=layer

#Wordinitial¹ Wordfinal² Medial³ sound
def posy(ipaword): #requires an ipastring as input, use epitran and uewscrape2ipa to transcribe your data
    ipaword=ipaword.replace('ː','') #this symbol can screw up everything. N.B: it's not a colon.
    splitipa=[]
    for i in ipa2tokens(ipaword): #tokenize the word
        ##print(i) #most errors appear here
        if str(i)[0] in Cns: #if it's a consonant,
            splitipa.append(i) #append it to the new list
        else:
            splitipa.append('%'+i+'%') #if it's not a consonant (=Vowel), append '%Vowel%' (% will do the split)
    splitipa=l2s(splitipa).split('%') #split so that consonant clusters stay together
    while("" in splitipa): 
        splitipa.remove("")  #remove empty elements from list
        
    splitipa[0]+='¹' #word initial/onset
    splitipa[-1]+='²' #word final/coda
    for idx,i in enumerate(splitipa):
        if idx !=0 and idx != len(splitipa)-1:
            splitipa[idx]+='³' #medial/nucleus
    
    if str(splitipa[0])[0] in Vow: #if first letter is vowel
        splitipa.insert(0, '0¹') #insert 0¹ at the beginning
    if str(splitipa[-1])[0] in Cns: #if last letter is consonant
        splitipa.append('∅²') #insert ∅² at the end
    return splitipa #return a list of vowel and consonant clusters with their position

def got2ipa(x): #in: word in Gothic orthography (as found at wikiling.de), out: ipa transcription
    x = x.lower()
    x = x.replace("ggw", "ŋɡw") #particularly important that it's chr(609) not chr(103)!
    x = x.replace("gg", "ŋɡ") #particularly important that it's chr(609) not chr(103)!
    x = x.replace("gk", "ŋk")
    x = x.replace("gq", "ŋkʷ")
    x = x.replace("aú", "ɔ")
    x = x.replace("áu", "au")
    x = x.replace("au", "au")
    x = x.replace("aí", "ɛ") #particularly important that it's chr(603) and NOT chr(949). They look the same.
    x = x.replace("ái", "aɪ")
    x = x.replace("ai", "aɪ")
    x = x.replace("ā", "a")
    x = x.replace("á", "a")
    x = x.replace("à", "a")
    x = x.replace("ē", "e")
    x = x.replace("ī", "i")
    x = x.replace("í", "i")
    x = x.replace("ì", "i")
    x = x.replace("ō", "o")
    x = x.replace("ū", "u")
    x = x.replace("i", "ɪ")
    x = x.replace("eɪ", "i")
    x = x.replace("q", "kʷ")
    x = x.replace("ƕ", "hʷ")
    x = x.replace("þ", "θ")
    x = x.replace("f", "ɸ")
    x = x.replace("b", "β") #exceptions defined below
    x = x.replace("d", "ð") #exceptions defined below
    x = x.replace("g", "ɣ") #exceptions defined below

    y=[]
    x=list(x)
    for idx,i in enumerate(x): #According to Braune2004 these word initial sounds were plosives
        if idx==0: #if word initial
            if i == 'β': 
                y.append('b')
            elif i == 'ð':
                y.append('d')
            elif i == 'ɣ':
                y.append('ɡ') #particularly important that it's chr(609) not chr(103)!
            elif i == 'x':
                y.append('h')
            else:
                y.append(i)
        else: #medially these were plosives after consonants (Braune2004)
            if i == 'β' and str(x[idx-1])[0] in Cns:
                y.append('b')
            elif i == 'ð' and str(x[idx-1])[0] in Cns:
                y.append('d')
            elif i == 'ɣ' and str(x[idx-1])[0] in Cns:
                y.append('ɡ') #particularly important that it's chr(609) not chr(103)!
            else:
                y.append(i)
    x=l2s(y)
    #correct wrongly transcribed geminates
    x = x.replace("βb", "bb")
    x = x.replace("ðd", "dd")
    x = x.replace("xh", "xx")

    return x

def gclean(graw): #input=df like G_raw_link.csv
    #manually remove 'lat.got., ' from line 1375 in G_raw_links.csv before running this code
    graw["Lemma"]=graw["Lemma"].str.split(", ") #bei Lemma stehen mehrere Grundformen durch ', ' separiert
    graw=graw.explode("Lemma").reset_index(drop=True) #diese Alternativformen in neue Reihen einfügen
    graw[['Lemma0','got_occurences']] = graw['Lemma'].str.split(' ', n=1, expand=True) #Zahl in extra Spalte
    graw['got_reconstructedness']=''
    graw['got_certainty']=''

    for idx,row in graw.iterrows(): #https://www.koeblergerhard.de/got/3A/got_vorwort.html (strg+f: 'Stern')
        try:
            if row['Lemma0'][-1]=='*' or row['Lemma0'][-2]=='*': #Wenn '*' am Wortende (wegen '?' manchmal vorletzte)
                graw.at[idx,'got_reconstructedness']='form' #dann ist die Grundform rekonstruiert
            elif row['Lemma0'][0]=='*': #Wenn '*' am Anfang, dann ist das Wort rekonstruiert
                graw.at[idx,'got_reconstructedness']='word' 
            else:
                graw.at[idx,'got_reconstructedness']='not reconstructed'

        except IndexError:
            continue #bei Wörtern von länge 1 ohne '?' kann er nicht das vorletzte Zeichen checken.

    for idx,row in graw.iterrows(): #Wenn '?' vorkommt 'uncertain' sonst 'certain' in extra spalte
        if '?' in row['Lemma0']:
            graw.at[idx,'got_certainty']='uncertain'
        else:
            graw.at[idx,'got_certainty']='certain'

    graw['Lemma']=graw['Lemma'].str.strip('*1?234,()56=789-0.#↑ ') #alle nicht-alphabetischen Zeichen entfernen
    for idx, row in graw.iterrows(): #Übriggebliebene '*',' ' (+alles danach) und 'Pl.' enfernen
        graw.at[idx,'Lemma']=row['Lemma'].split('*', 1)[0] 
    for idx, row in graw.iterrows():
        graw.at[idx,'Lemma']=row['Lemma'].split(' ', 1)[0]
    for idx, row in graw.iterrows():
        graw.at[idx,'Lemma']=row['Lemma'].replace('Pl.', '')
        
    graw=graw[graw['Lemma'].astype(bool)] #probably to drop rows where Lemma is nan

    for idx,row in graw.iterrows(): #transcribe to ipa and fill into column that turned unnecessary
        graw.at[idx, 'Lemma0']=got2ipa(row['Lemma'])
    graw=graw.rename({'Lemma0': 'got_ipa','Lemma':'got_lemma'}, axis=1) #rename column
    
    #clean translations, transcribe pos
    graw.insert(6, 'got_en', '')
    graw.insert(4, 'got_pos', '')
    for index,row in graw.iterrows():
        try:
            gothic3=row['Englische Bedeutung'].replace(', ',',').replace(' (','(').replace(' ','_')
            gothic3=re.sub(r" ?\([^)]+\)", "", gothic3) #remove parentheses and their content
            gothic3=re.sub(r'[^A-Za-z,_]+', '', gothic3) #keep only letters of English alphabet, commas, and underscores
            gothic3=gothic3.replace(',',', ') #looks nicer. csv can only store strings.
            graw.at[index,'got_en']= gothic3
        except AttributeError:
            graw.at[index,'got_en']= ''
        
        try:
            graw.at[index,'got_pos']= wikidict[row['Wortart']]
        except KeyError:
            graw.at[index,'got_pos']='n,v,a,r'
            
    gcln=graw.drop(['#', 'Sprachen'], axis=1)
    gcln.to_csv("gcln.csv", encoding="utf-8", index=False)
    gcln.to_excel("gcln.xlsx", encoding="utf-8", index=False)
    
    return gcln 

def getsubsti(gcln): #in: df created with gclean() out: Table1 to fill in substitutions, Table2: clusters
                     #TCLR (Prardis 1988) shows ideas how to automatise substitutions
    #write table for filling in sound substitutions manually
    posylist=[]
    for index, row in gcln.iterrows():
        posx=posy(got2ipa(row['got_lemma']))
        if len(posx)<=6: #limit only true for Uralic timelayer
            posylist+=posx
    mastercolumnlist=list(sum(Counter(posylist).most_common(), ()))
    cms1=[]
    cms2=[]
    msnd='ŋɡw,ŋɡ,ŋk,ŋkʷ,au,aɪ,ɪu,kʷ,hʷ'
    for idx,i in enumerate(mastercolumnlist):
        if isinstance(i, str): #otherwise getting errors since every 2nd element is an integer
            if i[:-1] in msnd or len(i)<=2:
                cms1.append(i)
                cms1.append(mastercolumnlist[idx+1])
            else:
                cms2.append(i)
                cms2.append(mastercolumnlist[idx+1])
    substi1 = pd.DataFrame(columns=cms1) #for i in columns, if vowel insert ɜ¨ȣ
    substi2 = pd.DataFrame(columns=cms2)
   
    substi1.to_excel("substi_single.xlsx", encoding="utf-8",index=False)
    substi2.to_excel("substi_clusters.xlsx", encoding="utf-8",index=False)
    substi1.to_csv("substi_single.csv", encoding="utf-8",index=False)
    substi2.to_csv("substi_clusters.csv", encoding="utf-8",index=False)
    #Excel: File/Options/Formula/Enable R1C1-Format for better overview
    return substi1,substi2 

#shuffle substitutions for consonant clusters
def gclustercng(substi1, substclt): #2 dfs: substi1&substi2 from gclean()
    dfsndy2=pd.DataFrame()
    allowed = [x[:-1] for x in clustercount(uralin,'Old').iloc[::2, :]['allowed'].tolist()] #get allowed clusters
    forbidden=['nan','0'] #otherwise the 0 from h will be in middle, since pos is lost
    substsnd=substi1.rename(columns= lambda s: s[:-1]) #remove 123 from cm names (!)
    substsnd = substsnd.loc[:,~substsnd.columns.duplicated()] #remove duplicate columns, else problems
    cltlst=list(substclt.columns) #convert the clumns to a list
    del cltlst[1::2] #delete every second element (=frequency)
    for i in cltlst:
        args=[]
        sndlst2=[]
        sndlst3=[]
        sndlst=ipa2tokens(i, merge_vowels=False, merge_geminates=False)
        for j in sndlst[:-1]: #dont take position into account (-last element)
            try:
                args.append([x for x in substsnd[j] if str(x) not in forbidden]) #[0] wichtig, wegen pos jede cm 3x da
            except KeyError:
                if j=='ŋ': #this sound appears only in gothic clusters, but not as a single sound, that's why keyerror
                    args.append('ŋ')
                else:
                    #print('other keyerror')
                    continue
                    
        if str(i[0])[0] in Cns: #if consonant, do combinations
            for i in [list(item) for item in combinations(args, 2)]: # for i in list of tuples of lists of strings
                sndlst2+=[convertTuple(item) for item in itertools.product(*i)] #shuffle
            for i in sndlst2: #throw out impossible clusters
                if i in allowed:
                    sndlst3.append(i)
            sndlst2=[item for sublist in sndlst2 for item in sublist]+sndlst3 #append clusters to single sounds
        else: #if vowel don't shuffle
            sndlst2=[item for sublist in args for item in sublist]
        
        if sndlst2 ==[]:
            sndlst2.append('') #wichtig, weil er sonst keine Spalte ans df anhängt
        sndlst2 = list(dict.fromkeys(sndlst2)) #remove duplicates
        dfsndy=pd.DataFrame(sndlst2) #list to df
        dfsndy2=pd.concat([dfsndy2,dfsndy], ignore_index=True, axis=1) #concat dfs, ignore length
    
    dfsndy2.columns=cltlst #rename columns correctly
    substi1=substi1[substi1.columns[::2]] #remove every second column
    dffinal=pd.concat([substi1,dfsndy2], axis=1) #concatenate with non-cluster df
    
    #add ¨,ȣ,ɜ to vowels
    for i in dffinal.columns:
        if i != '0¹': #otherwise keyerror
            if i[0]=='a': #a can be front OR back vowel
                dffinal[i]= pd.Series([x for x in dffinal[i].tolist() if str(x) != 'nan']+['ɜ','ȣ','¨'])
            elif str(i[0])[0] in Vow: #if vowel
                if str(i[0])[0] in back: #if back vowel
                    dffinal[i]= pd.Series([x for x in dffinal[i].tolist() if str(x) != 'nan']+['ɜ','ȣ']) #append ɜ,ȣ
                elif str(i[0])[0] in front: #if front vowel
                    dffinal[i]= pd.Series([x for x in dffinal[i].tolist() if str(x) != 'nan']+['ɜ','¨']) #append ɜ,¨
                              
    dffinal.to_excel("substituions12.xlsx",encoding='utf-8',index=False)
    dffinal.to_csv("substitutions12.csv",encoding='utf-8',index=False) #write an excel and a csv file
    return dffinal #df containing all possible sound substituions single sounds+clusters

def sufy(word): #in: word, out: list of poss. substitutions
    global subsdict
    if subsdict=={}:
        try:
            sbsin = pd.read_csv("substitutions12.csv", encoding='utf-8') #created by gclustercng()
            sbsin=sbsin.replace(0.0,'0') #sonst nervig
            subsdict={}
            for column in sbsin:
                subsdict[column]=[x for x in sbsin[column] if str(x) != 'nan']
        except FileNotFoundError:
            print("substitutions12.csv not found")

    subby=[]
    args=[]
    Wort=posy(word)
    try:
        #get unique indexes e.g. for 'ipipo' res=[0,1,0,1,2]
        d = defaultdict(count(1).__next__)
        res = [d[k]-1 for k in Wort]
        #keep types. Like set but a list in the correct order
        Wort = list(dict.fromkeys(Wort))
        #get substituion laws from csv
        for i in Wort:
                args.append(subsdict[i])
        #reconstruct the word from the combinations and unique indexes
        for subst in itertools.product(*args):
            Spiegel=[]  
            for i in res:
                Spiegel+=subst[i]
                Spiegel=''.join(Spiegel)
            subby.append(Spiegel)
        return subby
    except KeyError:
        print(word,' has soundclusters not in substitutions12.csv')
        pass

def dfsufy(gcln): #needs a dataframe such as "gcln3.csv" made by gclean() (=list of words in Gothic IPA, cleaned)
    gcln['substi']=gcln['got_ipa'].apply(sufy)
    gcln=gcln.explode("substi").reset_index(drop=True) #put every substituted form into own row
    gcln.fillna("",inplace=True)
    gcln['substi_struc']=gcln['substi'].apply(word2struc)
    gcln.to_csv("dfgot"+timelayer+".csv", encoding="utf-8",index=False) #write csv
    #don't write to excel b/c it will take for ever. Rather transform csv with excel later.
    return gcln


def SCE(dfwcipa): #sound change extractor. In: df with wordchanges: new word in clm0 and old word in clm1, all in ipa
    dfscefinal=pd.DataFrame(columns=['loansound','donorsound','soundchange','wordchange'])
    for idx,row in dfwcipa.iterrows():
        #four columns: loansound, donorsound, soundchange, wordchange
        #ipa is necessary to keep the code dynamic. Transcription is part of preprocessing. v=vowels, c=consonants     
        lw=posy(row['New'])
        dw=posy(row['Old'])
        dfsce=pd.DataFrame()
        #padding. insert 0 and ∅ until sets have same length
        while len(lw)<len(dw):
            lw += ['0','∅']
        while len(lw)>len(dw):
            dw += ['0', '∅'] 

        #remove positions from dw
        dw1=[]
        positions=['¹','²','³']
        for i in dw:
            if i[-1] in positions:
                dw1.append(i[:-1])
            else:
                dw1.append(i)
        sc=[]
        lwdw=[]
        for i in range(len(lw)):
            scstr=lw[i]+'<*'+dw1[i]
            lwdwstr=row['New']+'<*'+row['Old']
            sc.append(scstr)
            lwdw.append(lwdwstr)

        dfsce['loansound']=lw
        dfsce['donorsound']=dw1
        dfsce['soundchange']=sc
        dfsce['wordchange']=lwdw
        
        dfscefinal=dfscefinal.append(dfsce)
    
    dfscefinal=dfscefinal.reset_index()
    return dfscefinal

def qfysc(dfwcipa): #takes a dataframe with new words in cm0 and old word in cm1 as input, gives SCin as output
    global dictse
    dictse={}
    global dictex
    dictex={}
    dfsce=SCE(dfwcipa)
    sclist=dfsce['soundchange'].tolist() #transform column to list because you can't count uniques in a column later
    wclist=dfsce['wordchange'].tolist()
    for index,row in dfsce.iterrows(): #loop through rows
        scstr=row['soundchange'] #have to store these in a variable, otherwise the dict gives weird errors
        wcstr=row['wordchange']
        if scstr not in dictse: #dict defined on top of code, check if sound change in dict[keys]
            dictse[scstr]=sclist.count(scstr) #if not, count how often it appears and insert to value
        if scstr not in dictex:    
            allex=[] #allexamples, will be inserted to other dictionary
            for idx,j in enumerate(sclist): #loop through all scoundchanges again
                if j==scstr: #if it's the soundchange in question
                    if wclist[idx] not in allex: #and if the example in question is not in the allex-list yet
                        allex.append(wclist[idx]) #append the wordchange example to the allex-list
            dictex[scstr]=allex #insert list of all examples to value of dict[key]
    
    #write both dicts to txt-files
    file = codecs.open("uraloSEdict.py", "w", "utf-8")
    file.write('dictse='+str(dictse))
    file.close()
    file = codecs.open("uraloEdict.py", "w", "utf-8")
    file.write('dictex='+str(dictex))
    file.close()

    dfsce=dfsce.groupby('loansound')['donorsound'].apply(lambda x:list(set(x)))
    dfsce= dfsce.to_frame().T
    
    #This part looks nasty and is long, but I found no better way
    #Inserting Sum of Examples (SE) and Examples (E) from dictionaries
    cm1=dfsce.columns #Columns of old dataframe to list
    cm2=[] #Column names of SE
    cm3=[] #Column names of E
    cms=[] #Columns of the new dataframe
    for i in cm1:
        cm2.append(i+'_SE')
        cm3.append(i+'_E') #List of correct column names
    for i in range(1,len(cm1)): #Merge the three lists in correct order
        cms.append(cm1[i])
        cms.append(cm2[i])
        cms.append(cm3[i])
        
    scin=pd.DataFrame(columns=cms) #Create dataframe (will be returned at end) with the new column names
         
    for column in dfsce: #for every column of the old dataframe
        scl=[item for sublist in dfsce[column].tolist() for item in sublist] #column to list, flatten list
        scin[column]=pd.Series(scl) #insert list into new dataframe
        
        sel=[] #sum of example list
        for i in scl: #for every soundchange            
            try:
                sel.append(dictse[column+'<*'+i])   #get the right SE from dictionary
            except KeyError: #if it is in the dictionary
                continue #if not then just skip it
        scin[column+'_SE']=pd.Series(sel) #insert into next column of new dataframe
        
        exl=[] #example list
        for i in scl:             #for every sound change
            try:
                exl.append(dictex[column+'<*'+i])    #append the correct examples from dictionary
            except KeyError: #if it's not in the dictionary
                continue #then just skip it
        scin[column+'_E']=pd.Series(exl) #insert into the next column of new dataframe
        
    #scin.to_csv("scin.csv", encoding="utf-8") #write a csv (other functions use this as input)
    #scin.to_excel("scin.xlsx", encoding="utf-8") #write an Excel file, to be able to read it yourself
   
    return scin #return df with Soundchanges, Sum of Examples and Exampless

def qfyscgot(dfwcipa): #like qfysc() but excludes impossible sounds based on substitutions from goth.
    shwubby=pd.read_csv("substitutions12.csv", encoding='utf-8') #read substitutions, made with gclustercng()
    shwlist=[]
    for column in shwubby:
        shwlist+=shwubby[column].tolist() #put all clms into one list
    possiblesnd = set([x for x in shwlist if str(x) != 'nan']) #get all unique values = possible sounds
    scin=qfysc(dfwcipa) #returns the normal scin file
    scin2=pd.DataFrame(columns=scin.columns)
    for index,column in enumerate(scin):
        scl=scin[column].tolist()
        scl=[x for x in scl if str(x) !='nan']
        if index % 3 == 0: #wenn restlos durch 3 teilbar (=jede dritte spalte)
            scin2[column]=pd.Series([x for x in scin[column].tolist() if str(x) in possiblesnd],dtype=pd.StringDtype())
            scl=scin2[column].tolist()
            scl=[x for x in scl if str(x) !='nan']
            
            sel=[] #sum of example list
            for i in scl: #for every soundchange            
                try:
                    sel.append(dictse[column+'<*'+i])   #get the right SE from dictionary
                except KeyError: #if it is in the dictionary
                    continue #if not then just skip it
            scin2[column+'_SE']=pd.Series(sel,dtype=int) #insert into next column of new dataframe

            exl=[] #example list
            for i in scl:             #for every sound change
                try:
                    exl.append(dictex[column+'<*'+i])    #append the correct examples from dictionary
                except KeyError: #if it's not in the dictionary
                    continue #then just skip it
                scin2[column+'_E']=pd.Series(exl,dtype='object') #insert into the next column of new dataframe
                
    scin2.to_csv("scingot"+timelayer+".csv", encoding="utf-8") #write a csv (other functions use this as input)
    scin2.to_excel("scingot"+timelayer+".xlsx", encoding="utf-8") #write an Excel file, to be able to read it yourself
    global SCin
    SCin=scin2
    return scin2

def noder(word): #no derivative suffixes. Returns a list of possible wordforms without der. suff.
    Wort=posy(word)
    if Wort is not None: #if word is e.g. longer than 6 position() returns None
        nyd=[Wort] #noyesderivationalsuffixes: list of words with and without der. suff.
        if Wort[-1]== '∅²': #if last element is '∅²' remove it
            Wort=Wort[:-1] #has to be phrased this way because .pop() and del will change ALL elements of list
            try:
                if Wort[-1] in SCin: #to avoid KeyErrors
                    x=SCin[Wort[-1]] #remove derivational suffixes one after another in a while loop
                    while '0' in list(x) or '∅' in list(x): #'0' or '∅' being in SCin[Wort] means it's a possible der. suff.
                        Wort=Wort[:-1] #remove der.suff.
                        nyd.append(Wort) #append word without suff to list
                        try:
                            x=SCin[Wort[-1]] 
                        except KeyError:
                            break #no idea if this part is actually necessary but don't dare to modify it for now
            except TypeError:
                print(Wort[-1]," not from Gothic")
        return nyd #is a list of lists, namely the posy() of each word with and without der.suff. ("noyesdersuff") 

#gets you a list of possible protoforms, input is posy(word) or better i for i in noder(word)
def shuffle(splitipa):
    #check if all sounds can be old, if yes: do combinations, else: write which sounds can't be old
    if all(i in SCin for i in splitipa): #check if all sounds are in SCin.columns to avoid KeyErrors
        args=[] #all possible proto-sounds of a current sound
        global protolist #loan() will use this as input
        protolist=[]
        #get all possible protoforms for each sound & append to args, don't pick sounds from the forbidden-list
        for i in splitipa:
            args.append(SCin[i])
        if splitipa[0]=='0¹':
            args[0].append('0') #generally '0' is a forbidden sound, EXCEPT when it's word initial
        #print(args)
        for protoform in itertools.product(*args): #get all possible sound-combinations/pseudo-protoforms (=shuffle)
            protolist.append(l2s(protoform)) #turn list of sounds into words
        return protolist #loan() takes this list of possible protoforms as input
    else: #If sound not in SCin.columns tell which sound it is
        for i in splitipa:
            if i not in SCin:
                print(i, 'not from ', timelayer)

#Drop protoforms of illegal structure, in: output of shuffle()
def structure(protolist):  #replace the first part of this function with this word2struc()      
    pwlist=[]
    protolist2=[]
    if protolist is not None: #e.g. if a sound is not in SCin.Columns shuffle() returns None
        for idx,j in enumerate(protolist):
            pw='' #this will contain the structure of the word, has to be emptied every round
            for i in list(ipa2tokens(j)): #lingpy's tokenizer is useful here
                if str(i)!='0': #0 would give a KeyError because it's not in ipacsv and shouldn't be inserted there since 0 is not a C.
                    if str(i)[0] in Vow: #[0] after strin(i) necessary so geminates are interpreted as single C
                        #if sound is defined as a vowel in ipacsv, add a 'V' to pw
                        pw+='V'
                    if str(i)[0] in Cns:
                        pw+='C'
            if pw in allowedstruc:
                protolist2.append(j) #only append words to new list that have an allowed structure
    return protolist2 #input for loan(), like posy() but better

def NSE(lwipa,dwipa): #has to be dictionary based. Because you can't write the NSE manually into a csv
                      #NSE can only be calculated with this module, which means you'll always have a dictionary also
                      #NSE is only in SCin to view for the human eye
                      #it's 100 times easier to extract NSE from dictionary than from csv
    data = {'New': [lwipa], 'Old': [dwipa]} #since SCE takes df as input, we have to convert the input to df first
    df4sce=pd.DataFrame(data, columns=['New','Old'])
    E=[]
    SE=0
    NSE=0
    dfsce=SCE(df4sce) #get the sound changes of the two words
    scl=dfsce['soundchange'].tolist()
    for i in scl: #get the necessary info out of the dictionaries created with qfysc
        E.append(i+': '+str(dictex[i])) #if this throws a keyerror it's mostly true. Soundchange rules are strict
        SE+=dictse[i] #To test this don't use random words. Mostly they throw legitimatelly a keyerror
    NSE=SE/len(dfsce.index) #calculate the NSE based on the length of df (includes 0,∅, and padding)
    return NSE,SE,E

def semsim(hungarian, gothic, nvarhun='n,v,a,r', nvargot='n,v,a,r'): #nvar=noun/verb(adjective/adverb)
#hungarian means English translation of the hungarian word. Can't change the var name bc find and replace in jupyter...
#...is so bad. Also "gothic" is the English translation of the Gothic word.
    global model
    if model==[]:
        from gensim.models import KeyedVectors #to calculate semantic similarities
        model = KeyedVectors.load_word2vec_format(r'C:\Users\Viktor\Downloads\GoogleNews-vectors-negative300.bin', binary=True)
    try:
            #if isinstance(hungarian, str):
        if str(gothic)+', '+str(hungarian) in semsimdict:
            return semsimdict[str(gothic)+', '+str(hungarian)]
        else:
            hungarian=hungarian.replace(', ',',').replace(' ','_')
            hungarian=re.sub(r'[^A-Za-z,_]+', '', hungarian)
            hungarian=hungarian.split(',') #turns translations into clean list
            gothic=gothic.split(', ') #because you can't store lists in csvs, only strings

            hungarian=[hungarian]+[y.lemma_names() for y in [wn.synsets(x) for x in hungarian][0] if y.pos() in nvarhun]
            hungarian = list(dict.fromkeys([item for sublist in hungarian for item in sublist]))[:20]
            gothic= [gothic]+[y.lemma_names() for y in [wn.synsets(x) for x in gothic][0] if y.pos() in nvargot]
            gothic = list(dict.fromkeys([item for sublist in gothic for item in sublist]))[:20]

            topsim=-1 #score of the most similar word pair
            hwrd='KeyError gensim'
            gwrd='KeyError gensim'

            topsim=-1
            for j in hungarian:
                for i in gothic: #calculate semantic similarity of all pairs
                    try:
                        #####print(j,i,model.similarity(j, i))
                        if model.similarity(j, i)>topsim: #if word pair is more similar than the current topsim, replace topsim
                            topsim=model.similarity(j, i)
                            gwrd=i
                            hwrd=j
                    except KeyError: #if word is not in KeyedVecors of gensim continue
                        continue   
            semsimkey=str(gothic)+', '+str(hungarian)
            semsimval=str(topsim)+'; '+hwrd+'; '+gwrd
            semsimdict.update({semsimkey : semsimval})
            return topsim,hwrd,gwrd

    except TypeError:
        return 'en_transl missing'
    except AttributeError:
        return 'something is missing'

def loan(word, en='', de='', nvarhun='n,v,a,r', age='', etymology=''):
    strushuff=[]
    #print(type(word)) #str
    noderword=noder(word)
    if noderword is not None: #get alternativeforms() of the word
        for l in noderword:
            strushuff+=(structure(shuffle(l)))
    if strushuff is not None:
        compdf=pd.DataFrame({'proto':strushuff,'word':word, 'hun_en':en,'hun_de':de,'pos_hun':nvarhun,\
                            'appearance_year':age,'etymology':etymology})
        dfmatch = pd.merge(compdf, dfgot, how='inner', right_on=['substi'],left_on=['proto'])
           
    return dfmatch #pandas dataframe

def loanaddsemsim(df):
    df['semsim']=''
    for index,row in df.iterrows():
        #print(index)
        df.at[index,'semsim']=semsim(row['hun_en'],row['got_en'],row['pos_hun'],row['got_pos'])
    df.to_csv("loans"+timelayer+"2.csv", encoding="utf-8",index=False)
    df.to_excel("loans"+timelayer+"2.xlsx", encoding="utf-8",index=False)
    return df

def loandf(df,wordipa,en,de,pos_hun):
    dfout=pd.DataFrame()
    for index,row in df.iterrows():
        print(index)
        dfout=dfout.append(loan(str(row[wordipa]),str(row[en]),str(row[de]),str(row[pos_hun])))
    dfout['layer']=timelayer
    dfout.to_csv("loans"+timelayer+".csv", encoding="utf-8",index=False)
    return dfout

def getloans(layer):
    indf=pd.read_csv("unknown.csv", encoding='utf-8') #if keyerror, try sep=';' b/c excel often screws this up
    settimelayer(layer) #returns the correct uralin file and define global variable timelayer
    try:
        loans=pd.read_csv("loans"+timelayer+".csv", encoding='utf-8')
    except FileNotFoundError:
        qfyscgot(uralin) #qfy the correct layer every time
        loandf(indf,'IPA','en','de','type')
        loans=pd.read_csv("loans"+timelayer+".csv", encoding='utf-8')
        
    loanaddsemsim(loans)
