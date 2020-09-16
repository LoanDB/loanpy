#loanpy module. Main function loan()
#(c) Viktor Martinovic 01.09.2020
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
import requests
from bs4 import BeautifulSoup
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO
from pdfminer.pdfpage import PDFPage
import gc

forbidden=['nan','0.0','∅',"0","<NA>"] #shuffle needs this variable. Add sounds to this variable that can't be...
ipa2uewdict={'¨':'ȣ̈','ɑ':'a','æ':'ä','ð':'δ','ðʲ':'δ́','ɣ':'γ','lʲ':'ĺ',\
             'nʲ':'ń','ʃ':'š','y':'ü','θ':'ϑ','t͡ʃ':'č','ʃʲ':'ś','t͡ʃʲ':'ć'}
#check uewscrape2ipa()
semsimdict={'KeyError gensim, KeyError gensim': 'KeyError gensim, KeyError gensim, -1'}
wikipos='Abkürzung,Adj.,Adv.,Art.,Buchstabe,F.,Interj.,Konj.,LN.,M.,N.,Num.,ON.,Partikel,PN.,Präp.,'\
'Pron.,Sb.,V.,Wort,nan'
wikikeys=wikipos.split(',')
wikivalues=['r','a','r','r','r','n','r','r','n','n','n','r','n','r','n','r','r','n','v','n','nvar']
wikidict = dict(zip(wikikeys, wikivalues))

subsdict={}
model=[]

def ipacsvfunction(column): #extracts soundtypes from ipa.csv (panphon) and ads them to a string (=faster)
    try:
        ipacsv= pd.read_csv("ipa_table.csv", encoding='utf-8',sep=';') #list from panphon to tell if an
        #IPA sound is a vowel
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

#####CREATE INPUT FILE
#convert pdf to string (from youtube tutorial)
def get_pdf_file_content(path_to_pdf):
    resource_manager = PDFResourceManager(caching=True)
    out_text = StringIO()
    laParams = LAParams()
    text_converter = TextConverter(resource_manager, out_text,laparams=laParams)
    fp = open(path_to_pdf, 'rb')
    interpreter = PDFPageInterpreter(resource_manager,text_converter)
    for page in PDFPage.get_pages(fp,pagenos=set(),maxpages=0,password="",caching=True,check_extractable=True):
        interpreter.process_page(page)
    text= out_text.getvalue()
    
    fp.close()
    text_converter.close()
    out_text.close()
    
    return text
    
#webscrape hungarian-english dictionary, store as txt
def scrapedict_huen():
    for i in range(ord('a'), ord('z')+1): #dictionary entries from a-z
        if i != ord("q"): #only letter missing from dictionary's website is "q"
            URL='https://mek.oszk.hu/00000/00076/html/hun-eng/'+chr(i)+'.htm'
            page = requests.get(URL)
            text_file = open("szotar"+chr(i)+".txt", "w")
            text_file.write(page.text) #write the webscraped page to a txt file
            text_file.close()

#convert txt to dictionary
def getdict_huen():
    hunendict={}
    for i in range(ord('a'), ord('z')+1): #dictionary entries from a-z
        print(chr(i))
        if i != ord("q"): #only letter missing from dictionary's website is "q"
            hul=[]
            enl=[]
            subdict={}
            soup1=BeautifulSoup(open("szotar"+chr(i)+".txt").read())
            soup1=soup1.body #cut out trash from beignning and end
            for s in soup1.select('script'): #cut off anything between tag "script"(bottom)
                s.extract()
            for s in soup1.select('center'):  #cut off anything between tag "center" (top)
                s.extract() 
            zl= re.sub(r'\<.*?\>', '', str(soup1)) #remove tags <body> and <html> from top and bottom
            if i == ord("z"): #z has some extra strings in the end that cause errors
                zl=zl[1:-9] #cut off the troublesome strings
            zlsplit=zl.split("\n\n ")[1:-1] #cut off first and last char, they cause errors
            for j in zlsplit:
                wordpair=j.split(" -» ") #split into hu and en word
                hul.append(wordpair[0].replace("õ","ő").replace("û","ű"))#correct wrong encoding
                enl.append(wordpair[1])
            for index, j in enumerate(hul):
                if j in hunendict:
                    hunendict[j].append(enl[index]) #add meaning if already in dict
                else:
                    hunendict[j]=[enl[index]] #else create new entry
    hunendict="hunendict="+str(hunendict)
    with open('hunendict.py','w',encoding="utf-8") as data:
        data.write(hunendict)
    return hunendict

#key: word, value: origin, extracted from zaicz pdf
def getdictorig():   
    zaiczcsv=pd.DataFrame(columns=['word','year','info','disambiguated','suffix',"en"]) #dffinal
    #zaicz1: year, zaicz2: origin
    zaicz2=zaicz.split(' \n \n\n \n\n\x0cA SZAVAK EREDET SZERINTI CSOPORTOSÍTÁSA* \n\n \n \n \n \n \n \n',1)[1]
    dictorig={}
    zlist=zaicz2.split("\n \n")
    for index,i in enumerate(zlist):
        if index<101:
            para=i.split("\n",1)
            paratag=para[0]
            if len(para)>1:
                paratxt=para[1]
                for i in paratxt.split(", "):
                    if i[-1]=="?":
                        dictorig[i.replace("x0c","").replace("\n","").replace("?","")]=paratag+"?"
                    else:
                        dictorig[i.replace("x0c","").replace("\n","")]=paratag
        if index>=101 and (index % 2) ==0:
            for j in i.split(", "):
                if i[-1]=="?":
                    dictorig[j.replace("x0c","").replace("\n","")]=zlist[index-1]+"?"
                else:
                    dictorig[j.replace("x0c","").replace("\n","")]=zlist[index-1]
    dictorig="dictorig="+str(dictorig)
    with open('dictorig.py','w',encoding="utf-8") as data:
        data.write(dictorig)
    return dictorig

#read annex of zaicz pdf and tranform to csv (main input file)
def zaicz2csv():
    try:
        from hunendict import hunendict
    except:
        print("create dictorig with getdictorig()")
        
    try:
        from dictorig import dictorig
    except:
        print("create dictorig with getdict_huen()")
    zaiczcsv=pd.DataFrame(columns=['word','year','info','disambiguated','suffix',"en","orig","pos_hun",\
                                  "wordipa"]) #dffinal
    #zaicz1: year, zaicz2: origin
    path_to_pdf = r"C:\Users\Viktor\Desktop\TAMOP_annex.pdf"
    zaicz=get_pdf_file_content(path_to_pdf)
    zaicz1=zaicz.split(' \n \n\n \n\n\x0cA SZAVAK EREDET SZERINTI CSOPORTOSÍTÁSA* \n\n \n \n \n \n \n \n',1)[0]
    zaicz2=zaicz.split(' \n \n\n \n\n\x0cA SZAVAK EREDET SZERINTI CSOPORTOSÍTÁSA* \n\n \n \n \n \n \n \n',1)[1]

    #zaicz1 (year):
    for index,i in enumerate(zaicz1.split('[')): #list of year-word pairs
        if ':' in i: #otherwise error
            zaiczcsv.at[index,'word']=i.split(':')[1].replace(" ","").replace("\n","")\
            .replace("1951-től","").replace("1000-ig","") 
            zaiczcsv.at[index,'year']=re.sub("[^0-9]", "", i.split(':')[0].split(',')[-1]) #the sure year
            zaiczcsv.at[index,'info']=i.split(':')[0][:-1].replace("\n","") #all other info

    for index,row in zaiczcsv.iterrows():
        zaiczcsv.at[index,'word']=row['word'].split(',') #explode funktioniert nur mit listen, darum split()
    #explode, reset index,drop old index,remove rows with empty cells in column "word"
    zcsv=zaiczcsv.explode('word')
    zcsv=zcsv.reset_index()
    zcsv=zcsv.drop('index',axis=1)
    zcsv= zcsv[zcsv.word != '']

    for index,row in zcsv.iterrows():
        if len(row['year'])==2: #e.g. 20 ist left from "20.sz" (20th century) so we'll append "00"
            zcsv.at[index,'year']=row['year']+'00'
        if row['word'][-2:].isnumeric(): #remove headers (like "1001-1100")
            zcsv.at[index,'word']=row['word'][:-9] #4+4+1 (year1+hyphen+year2)
        zcsv.at[index,'disambiguated']=row['word']
        if row['word'][-1].isnumeric(): #disambiguation to other column
            zcsv.at[index,'word']=row['word'][:-1]
        zcsv.at[index,'word']=row['word'].replace("~","/").split("/") #explode needs list, so split()

    zcsv=zcsv.explode('word')
    zcsv=zcsv.reset_index() #reset index to 1,2,3,4,... again
    zcsv=zcsv.drop('index',axis=1) #drop old index 1,1,1,2,2,3,4,....

    for index,row in zcsv.iterrows():
        if row['word'][0]=="-": #remove hyphens
            zcsv.at[index,"word"]=row["word"][1:]
            zcsv.at[index,"suffix"]="+" #mark that they're a suffix in extra column
        #insert translations
        try:
            zcsv.at[index,"en"]=str(hunendict[row["word"]]).replace("[","").replace("]","").\
            replace("'","").replace('"','') #b/c semsim requires a string not a list
            #b/c you can not store lists in a csv, only strings
        except KeyError:
            pass
        try:
            zcsv.at[index,"orig"]=dictorig[row["disambiguated"]]
        except KeyError:
            pass
        zcsv.at[index,"wordipa"]=epi.transliterate(row["word"])
    zcsv.to_excel("zcsv.xlsx", encoding="utf-8", index=False)
    zcsv.to_csv("zcsv.csv", encoding="utf-8", index=False)

#works on python 3.5. and spacy 2.0.12 (Hungarian pos_tagger)
#create virtual environment via anaconda navigator (environments->play)
def getpos_hu():
    zcsv=pd.read_csv("zcsv.csv", encoding='utf-8')
    import hu_core_ud_lg
    nlp = hu_core_ud_lg.load()
    for index,row in zcsv.iterrows():
        doc=nlp(row["word"])
        for i in doc:
            zcsv.loc[[index],["pos_hun"]]=i.pos_ #df.at does not work (maybe bc older python version?)
            print(index)
    zcsv.to_csv("zaicz_in.csv", encoding="utf-8", index=False)
    return zcsv

#converts spacy's pos-tags to nltk's. https://spacy.io/api/annotation
def spacy2nltk_postags():
    zcsv=pd.read_csv("zaicz_in.csv",encoding="utf-8")
    hunposdict={"ADJ":"a","ADP":"r","ADV":"r","AUX":"v","CONJ":"r","CCONJ":"r","DET":"r","INTJ":"r","NOUN":"n",\
                "NUM":"r","PART":"r","PRON":"r","PROPN":"r","PUNCT":"r","SCONJ":"r","SYM":"r","VERB":"v","X":"r",\
                "SPACE":"r"}
    for index,row in zcsv.iterrows():
        zcsv.at[index,"pos_hun"]=hunposdict[row["pos_hun"]]
    zcsv.to_csv("zaicz_in.csv", encoding="utf-8", index=False)
    return zcsv
#####################

def l2s(s):  
    str1 = ""   
    return (str1.join(s))

def convertTuple(tup): #for shuffling the clusters
    str = functools.reduce(operator.add, (tup)) 
    return str

def ipa2uew(word): #converts ipa strings to clean uew
    for i in ipa2tokens(word, merge_vowels=False, merge_geminates=False): #load the corresponding 
        #csv and transcribe according to table
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
    try:
        global SCin
        SCin={}
        SCincsv = pd.read_csv("scingot"+layer+".csv", encoding='utf-8')
        for column in SCincsv:
            SCin[column]=[x for x in SCincsv[column] if str(x) not in forbidden]
    except FileNotFoundError:
        print("scingot"+layer+".csv not found")
    lenlist=[]
    for i in list(allowedstruc):
        lenlist.append(len(i))
    global maxlength
    maxlength=max(lenlist)
    global timelayer
    timelayer=layer
    global dfgot
    dfgot=pd.read_csv("dfgot"+timelayer+".csv", encoding='utf-8')

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
            gothic3=re.sub(r'[^A-Za-z,_]+', '', gothic3) #keep only letters of English alphabet, commas, 
            #and underscores
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
        if len(posx)<=6: #limit only true for Uralic timelayer, #dynamise later
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
    gcln.to_csv("dfgot"+timelayer+"_raw.csv", encoding="utf-8",index=False) #write csv
    #don't write to excel b/c it will take for ever. Rather transform csv with excel later.
    return gcln

def gotdf_keepallowedstruc():
    try:
        dfgot=pd.read_csv("dfgot"+timelayer+"_raw.csv", encoding='utf-8')
        dfgot=dfgot[dfgot['substi_struc'].isin(allowedstruc)] #keep only words with allowed 
        dfgot.to_csv("dfgot"+timelayer+".csv", encoding="utf-8",index=False)
    except FileNotFoundError:
        print('dfgot'+timelayer+'.csv not found, generate with dfsufy()')


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
                    while '0' in list(x) or '∅' in list(x): #'0' or '∅' being in SCin[Wort] means it's a 
                        #possible der. suff.
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
                if str(i)!='0': #0 would give a KeyError because it's not in ipacsv and shouldn't be 
                    #inserted there since 0 is not a C.
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

def semsim(hungarian, gothic, nvarhun='n,v,a,r', nvargot='n,v,a,r',\
           gensimpath=r'C:\Users\Viktor\Downloads\GoogleNews-vectors-negative300.bin'): 
    #nvar=noun/verb(adjective/adverb)
#hungarian means English translation of the hungarian word. Can't change the var name bc find and replace in jupyter...
#...is so bad. Also "gothic" is the English translation of the Gothic word.
    global model
    if model==[]:
        from gensim.models import KeyedVectors #to calculate semantic similarities
        model = KeyedVectors.load_word2vec_format(gensimpath, binary=True)
    try:
            #if isinstance(hungarian, str):
        if str(gothic)+', '+str(hungarian) in semsimdict:
            return semsimdict[str(gothic)+', '+str(hungarian)]
        if hungarian=="" or gothic=="":
            return "some translation missing"
        else:
            #make sure "hungarian"/"gothic" is a string with separator=", "
            hungarian=hungarian.replace(', ',',').replace(' ','_') #prepare to convert string to list
            hungarian=re.sub(r'[^A-Za-z,_]+', '', hungarian) #clean
            hungarian=hungarian.split(',') #turns string to list
            gothic=gothic.split(', ') #because you can't store lists in csvs, only strings

            hungarian=[hungarian]+[y.lemma_names() for y in [wn.synsets(x) for x in hungarian][0] if y.pos() in nvarhun]
            hungarian = list(dict.fromkeys([item for sublist in hungarian for item in sublist]))[:20]
            gothic= [gothic]+[y.lemma_names() for y in [wn.synsets(x) for x in gothic][0] if y.pos() in nvargot]
            gothic = list(dict.fromkeys([item for sublist in gothic for item in sublist]))[:20]
            #get names of synsets, if they match the wordtype, flatten list, remove duplicates

            topsim=-1 #score of the most similar word pair
            hwrd='KeyError gensim'
            gwrd='KeyError gensim'

            topsim=-1
            for j in hungarian:
                for i in gothic: #calculate semantic similarity of all pairs
                    try:
                        #####print(j,i,model.similarity(j, i))
                        if model.similarity(j, i)>topsim: #if word pair is more similar than the current topsim, 
                            #replace topsim
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

def loan(wordipa,word="",info="",disambiguated="",suffix="",en="",pos_hun="n,v,a,r",year="",origin=""):
    if len(ipa2tokens(wordipa))> maxlength:
        #print("word too long")
        dfmatch=pd.DataFrame()
        return dfmatch       
    else:
        strushuff=[]
        #print(type(word)) #str
        noderword=noder(wordipa)
        if noderword is not None: #get alternativeforms() of the word
            for l in noderword:
                strushuff+=(structure(shuffle(l)))
        if strushuff is not None:
            compdf=pd.DataFrame({"proto":strushuff,"wordipa":wordipa,"word":word,"info":info,\
                                 "disambiguated":disambiguated,"suffix":suffix,"hun_en":en,"pos_hun":pos_hun,\
                                 "year":year,"origin":origin})
            dfmatch = pd.merge(compdf, dfgot, how='inner', right_on=['substi'],left_on=['proto'])
           
    return dfmatch #pandas dataframe

def loanaddsemsim(df):
    semlist=[]
    try:
        for index,row in df.iterrows():
            semlist.append(semsim(row['hun_en'],row['got_en'],row['pos_hun'],row['got_pos']))
            if index % 1000 == 0:
                print(index)
        return semlist
    except:
        with open("sem.py", "w") as output:
            output.write("semlist="+str(semlist))

        return semlist

def loandf(df): #spped up by vectorizing!!
    dfout=pd.DataFrame()
    for index,row in df.iterrows():
        print(index)
        dfout=dfout.append(loan(wordipa=str(row["wordipa"]),word=str(row["word"]),info=str(row["info"]),\
                                disambiguated=str(row["disambiguated"]),suffix=str(row["suffix"]),\
                                en=str(row["en"]),pos_hun=str(row["pos_hun"]),\
                                year=str(row["year"]), origin=str(row["orig"])))
    dfout['layer']=timelayer
    dfout.to_csv("loans"+timelayer+".csv", encoding="utf-8",index=False)
    #dfout.to_excel("loans"+timelayer+".xlsx", encoding="utf-8",index=False)
    return dfout

def getloans(layer, filename):
    indf=pd.read_csv(filename, encoding='utf-8') #if keyerror, try sep=';' b/c excel often screws this up
    settimelayer(layer) #returns the correct uralin file and define global variable timelayer
    try:
        loans=pd.read_csv("loans"+timelayer+".csv", encoding='utf-8')
    except FileNotFoundError:
        qfyscgot(uralin) #qfy the correct layer every time
        loandf(df=indf,wordipa="wordipa",en="en",pos_hun="pos_hun",year="year",origin="orig",word="word",info="info",\
              disambiguated="disambiguated",suffix="suffix")
        loans=pd.read_csv("loans"+timelayer+".csv", encoding='utf-8')        
    loanaddsemsim(loans)

def loanbigdf(): #cut in chunks of 500 rows
    zcsv=pd.read_csv("zaicz_in.csv",encoding="utf-8")
    zcsvpre1600=zcsv[zcsv.year <= 1600] #ca. 4500 words
    for i in list(range(0,len(zcsvpre1600)))[0::500]: #cut into chunks of 500
        if i == 4000:
            chunky=zcsvpre1600[i:]
        else:
            chunky=zcsvpre1600[i:i+500]
        z=loandf(chunky)
        zai=pd.read_csv("loansU.csv",encoding="utf-8")
        zai.to_csv("loans"+timelayer+str(i)+".csv", encoding="utf-8",index=False) #rename chunk
        zai=() #empty variably so it doesn't eat up memory (?)
        gc.collect()
        
def semsimbigdf(): #cut in chunks of 500 rows
    zcsv=pd.read_csv("zaicz_in.csv",encoding="utf-8")
    zcsvpre1600=zcsv[zcsv.year <= 1600] #ca. 4500 words
    leny=len(zcsvpre1600)
    zcsvpre1600=()
    for i in list(range(0,leny))[0::500]: #cut into chunks of 500
            z_in=pd.read_csv("loans"+timelayer+str(i)+".csv",encoding="utf-8",low_memory=False,\
                             usecols=["hun_en","got_en","pos_hun","got_pos"])
            z_in=z_in.fillna("")
            #z_in=z_in.head(100)
            semlist=loanaddsemsim(z_in)
            z_in=""
            z_in2=pd.read_csv("loans"+timelayer+str(i)+".csv",encoding="utf-8",low_memory=False)
            z_in2["semsim"]=pd.Series(semlist)
            z_in2.to_csv("loans"+timelayer+str(i)+"a_sem.csv", encoding="utf-8",index=False)
            z_in=() #empty variably so it doesn't eat up memory (?)
            gc.collect()
            
def getcutoffval(): #takes 4min 25s
    listy=[]
    for i in range(0,4000)[0::500]:
        listy.append(str(i))
    listy.append("4000a")
    listy.append("4000b") #append parts of all filenames to list
    cutoff=pd.DataFrame(columns=["semsim"]) #new df only col "semsim"
    for i in listy: #open one chunk after another
        print(i)
        z_in=pd.read_csv("loans"+timelayer+str(i)+"_sem.csv",encoding="utf-8",low_memory=False,\
                                 usecols=["semsim"])
        cutoff=cutoff.append(z_in) #append col "semsim" to main df "cutoff"
    print("resetting index")
    cut=cutoff.reset_index() #cln
    print("dropping old index")
    cut=cut.drop("index",axis=1) #cln
    print("removing brackets")
    cut["semsim"]=cut["semsim"].str[1:-1] #remove brackets (=first and last char)
    print("splitting cols")
    cut[["semsim","words"]]=cut["semsim"].str.split(",",1,True) #nrs & wrds in sep cols
    print("sorting values")
    cut=cut.sort_values(by=["semsim"],ascending=False) #sort nrs by descending order
    print("removing non-floats")
    cut2=cut.tail(len(cut)-5415858) #first 5415858 elements miss a translation (manual evaluation)
    print("cutting top 200K")
    cut3=cut2.head(200000)
    print("writing to csv")
    cut3.to_csv("cut.csv", encoding="utf-8", index=False)
    cutoffval=cut3.iloc[199999,0]
    return cutoffval #top 200 000 elements, the last one is the cut-off value

def cut_off():
    listy=[]
    print("Creating filename parts")
    for i in range(0,4000)[0::500]:
        listy.append(str(i))
    listy.append("4000a")
    listy.append("4000b")
    for i in listy:
        print("reading loans"+timelayer+str(i)+"_sem.csv")
        cut=pd.read_csv("loans"+timelayer+str(i)+"_sem.csv",encoding="utf-8",low_memory=False)
        #cut=cut.head(30000)
        floatlist=[]
        print("removing brackets")
        cut["semsim"]=cut["semsim"].str[1:-1] #remove brackets (=first and last char)
        print("splitting cols")
        cut[["semsim","meanings"]]=cut["semsim"].str.split(",",1,True) #nrs & wrds in sep cols
        floatlist=[]
        print("Converting to float, nr of rows: "+str(len(cut)))
        for ind,r in cut.iterrows():
            
            if ind % 10000 == 0:
                print(ind)
            try:
                floatlist.append(float(r["semsim"]))
            except ValueError:
                floatlist.append(float(-1))
        print("Inserting floatlist")
        cut["semsim"]=floatlist
        print("Cutting off")
        cut=cut[cut.semsim >= float(0.46550298)]# calculated with getcutoffval()
        print("Writing csv")
        cut.to_csv("loans"+timelayer+str(i)+"_cut.csv", encoding="utf-8",index=False)
        
def mergeout():
    listy=[]
    print("Creating filename parts")
    for i in range(0,4000)[0::500]:
        listy.append(str(i))
    listy.append("4000a")
    listy.append("4000b")
    mainout=pd.DataFrame()
    for i in listy:
        print("reading loans"+timelayer+str(i)+"_cut.csv")
        cut=pd.read_csv("loans"+timelayer+str(i)+"_cut.csv",encoding="utf-8",low_memory=False)
        mainout=mainout.append(cut)
    mainout.to_csv("mainout.csv", encoding="utf-8", index=False)
