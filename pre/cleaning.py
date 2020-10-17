#I. cleangot(): clean dfgot from wikiling.de    
    #1. insert links()
    #2. every lemma() to own row
    #3. occurences() to own col
    #4. certainty() to own col
    #5. reconstructedness() to own col
    #6.a clean col lemma
    #6.b clean col lemma
    #6. translations()
    #7.a activate got-ipa transcription file
    #8 clean English translations
    #9.a activate dictionary for translating pos-tag-names from wikiling to nltk
    #9.b translate pos-tags from wikiling to nltk notation
    #11. write empty file to fill in substitutions
    #12. write clean dfgot.csv
    
#II. cleanuralonet(): clean uralonet_raw.csv from uralonet.nytud.hu
    #1. turn sound to C for consonant or V for vowel
    #2. get phonotactic profile of word
    #3. activate transcription files with copy2epitran if not already activated while cleaning dfgot
    #4. clean uralonet_raw.csv
    
#III. mine and clean zaicz.csv
    #1. mine pdf
    #2. create dictionary from webscraped txts of English-Hungarian dictionary (web-address: )
    #3. create dictionary of word-origin pairs from zaicz.pdf
    #4. read annex of zaicz pdf and tranform to csv (main input file)
    #5. add missing translations with google translate
    #6. add pos-tags with spacy
        #!works on python 3.5. and spacy 2.0.12 (Hungarian pos_tagger)
        #!create virtual environment via anaconda navigator for this function
    #7. converts spacy's pos-tags to nltk's. https://spacy.io/api/annotation
    #8.a translate origin-tags from Hungarian to English with google translate
    #8.b insert translated origin-tags to df
    #9.a convert origin to tags "U, FU, Ug"
    #9.b insert new origin-tags "U, FU, Ug"
    #10. remove brackets
    #11. translate info-col to English
    #11.b insert translated info-col to df
    
#imports for cleangot() and cleanuralonet()
import pandas as pd
import re #reconstr(), clemma(), ctransl()
import epitran #transcribe gothic to ipa
import os #copy2epitran()
import shutil #copy2epitran()
import itertools #for deletion()
from lingpy import ipa2tokens
from loanpy import word2struc

#imports for zaicz.csv
from bs4 import BeautifulSoup
import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO
from pdfminer.pdfpage import PDFPage
from googletrans import Translator
translator = Translator()
hu2en_origdict={}
origtagdict={}
posdict={}

cns="jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC"
vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅"

os.chdir(os.path.dirname(os.path.abspath(__file__))+r"\data\pre") #change to folder "data"

#I. Clean dfgot

#1. insert col links
def links():
    linkliste=[]
    for i in range(1,281): #number of pages
        linkliste.append(20*["https://koeblergerhard.de/wikiling/?query=&f=got&mod=0&page="+str(i)]) #20 entries per page
    linkliste=[item for sublist in linkliste for item in sublist][:5582] #flatten list
    return linkliste

#2. explode lemmas
def explemma(graw):
    graw.at[1373,"got_lemma"]=str(graw.iloc[1373,1]).replace("lat.got., ","") #clean line 1373
    graw["got_lemma"]=graw["got_lemma"].str.split(", ") #bei Lemma stehen mehrere Grundformen durch ', ' separiert
    graw=graw.explode("got_lemma").reset_index(drop=True) #diese Alternativformen in neue Reihen einfügen
    return graw

#3. occurences to own col
def occurences(entry):
    return re.findall("[0-9]+",entry)
    
#4. level of certainty to own col
def certainty(entry):
    if "?" in entry:
        return "uncertain"
    else:
        return "certain"
    
#5. reconstructedness to own col
def reconstr(entry): #https://www.koeblergerhard.de/got/3A/got_vorwort.html (ctrl+f: "Stern")
    if re.search(r"\*\?? ?[0-9]* ?$",entry) is not None:
        return "form" #other forms documented, basic form reconstructed
    elif re.search(r"^\*",entry) is not None:
        return "word" #other forms not documented, word itself reconstructed
    else:
        return ""

#6.a clean col "got_lemma"
def helpclean(filename,column):
    chars=[]
    df=pd.read_csv(filename,encoding="utf-8")
    df=df.fillna("")
    for i in df[column].tolist():
        chars+=i
    return set(chars)

#6.b cluean col "got_lemma"
def clemma(entry):
    entry.replace("Pl.","")
    entry.lower()
    return re.sub(r"[^a-zA-ZÀ-ÿþāēīōūƕ]+", "", entry) #use helpclean() to find out what to keep

#7. copy files to epitran\data\map and epitran\data\post to piggyback epitran
def copy2epitran():
    epipath=epitran.__file__[:-(len("\epitran.py"))]+r"data"
    dstmap = epipath+r"\map"
    dstpost = epipath+r"\post"
    
    srcgotmap = os.getcwd()+r"\got-translit.csv"
    srcgotpost = os.getcwd()+r"\got-translit.txt"
    srcuew = os.getcwd()+r"\uew-scrape.csv"
    
    shutil.copy(srcgotmap,dstmap)
    shutil.copy(srcgotpost,dstpost) #special rules go to folder "post"
    shutil.copy(srcuew,dstmap)

#8 clean English translations
def ctransl(entry):
    entry=re.sub(r" ?\([^)]+\)", "", entry) #remove parentheses and their content
    entry=entry.replace(', ',',').replace(' (','(').replace(' ','_')
    entry=re.sub(r"[^0-9A-Za-z,_äéþōƕ]+", "", entry) #use helpclean() to find out what to keep
    entry=entry.replace(",", ", ")
    return entry

#9.a activate dictionary of wikiling-pos-tags to nltk-pos-tags
def getposdict():
    poskeys="Abkürzung,Adj.,Adv.,Art.,Buchstabe,F.,Interj.,Konj.,LN.,M.,N.,Num.,ON.,Partikel,PN.,Präp.,"\
    "Pron.,Sb.,V.,Wort," #last comma important
    posvalues=["r","a","r","r","r","n","r","r","n","n","n","r","n","r","n","r","r","n","v","n","nvar"]
    global posdict
    posdict = dict(zip(poskeys.split(','), posvalues))

#9.b translate wikiling pos-tags to nltk-pos tags
def nltktags(entry):
    if posdict=={}:
        getposdict()
    nltktags=""
    for i in entry.split(", "):
        try:
            nltktags+=posdict[i]
        except KeyError:
            return "nvar"
    return nltktags

#11. write fillitout.csv
def fillitout(column): #automate this function later
    fillout = pd.DataFrame({"to_substitute" : sorted(list(set([i for s in column.apply(ipa2tokens, merge_vowels=False, merge_geminates=False).tolist() for i in s])))})
    fillout["substitution"]=""
    fillout.to_csv("fillitout.csv",encoding="utf-8",index=False)
    
def cleangot(filename): #e.g. g_raw.csv
    graw=pd.read_csv(filename, encoding="utf-8")
    graw=graw.rename({"Lemma":"got_lemma"}, axis=1) #rename column
    graw=graw.drop(['#', 'Sprachen'], axis=1)
    graw=graw.fillna("") #else problems with nans
    graw["links"] = links()
    graw=explemma(graw)
    graw["occurences"]=graw["got_lemma"].apply(occurences)
    graw["got_certainty"]=graw["got_lemma"].apply(certainty)
    graw["got_reconstructedness"]=graw["got_lemma"].apply(reconstr)
    graw["got_lemma"]=graw["got_lemma"].apply(clemma)
    graw=graw[graw["got_lemma"].astype(bool)] #remove rows where lemma turned empty after cleaning
    copy2epitran() #copy files to epitran-folder
    graw["got_ipa"]=graw["got_lemma"].apply(epitran.Epitran("got-translit").transliterate)
    graw["got_en"]=graw["Englische Bedeutung"].apply(ctransl)
    graw["got_pos"]=graw["Wortart"].apply(nltktags)
    gotclean=graw
    gotclean.to_csv("dfgot.csv",encoding="utf-8",index=False)
    return gotclean

#################################################################################################################

#II. Clean uralonet_raw.csv

    #1. turn sound to C for consonant or V for vowel
    #2. get phonotactic profile of word
    #3. activate transcription files with copy2epitran if not already activated while cleaning dfgot
    #4. clean uralonet_raw.csv

def cleanuralonet(filename): #in: uralonet_raw.csv
    df=pd.read_csv(filename,encoding="utf-8")
    copy2epitran()
    df["New"]=df.New_orth.apply(epitran.Epitran('hun-Latn').transliterate)
    df["Old"]=df.Old_orth.apply(epitran.Epitran('uew-scrape').transliterate)
    df["old_struc"]=df.Old.apply(word2struc)
    df.to_csv("uralonet.csv",encoding="utf-8",index=False)
    return df

###################################################################################################################

#III mine and clean zaicz.csv
    #1. mine pdf
    #2. create dictionary from webscraped txts of English-Hungarian dictionary (web-address: )
    #3. create dictionary of word-origin pairs from zaicz.pdf
    #4. read annex of zaicz pdf and tranform to csv (main input file)
    #5. add missing translations with google translate
    #6. add pos-tags with spacy
        #!works on python 3.5. and spacy 2.0.12 (Hungarian pos_tagger)
        #!create virtual environment via anaconda navigator for this function
    #7. converts spacy's pos-tags to nltk's. https://spacy.io/api/annotation
    #8.a translate origin-tags from Hungarian to English with google translate
    #8.b insert translated origin-tags to df
    #9.a convert origin to tags "U, FU, Ug"
    #9.b insert new origin-tags "U, FU, Ug"
    #10. remove brackets
    #11. translate info-col to English
    #11.b insert translated info-col to df

#1. mine pdf 
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

#2. create dictionary from webscraped txts of English-Hungarian dictionary (web-address: )
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

#3. create dictionary of word-origin pairs from zaicz.pdf
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
                        dictorig[i.replace("x0c","").replace("\n","").replace("?","").replace(" ","")]=paratag+"?"
                    else:
                        dictorig[i.replace("x0c","").replace("\n","").replace(" ","")]=paratag
        if index>=101 and (index % 2) ==0:
            for j in i.split(", "):
                if i[-1]=="?":
                    dictorig[j.replace("x0c","").replace("\n","").replace(" ","")]=zlist[index-1]+"?"
                else:
                    dictorig[j.replace("x0c","").replace("\n","").replace(" ","")]=zlist[index-1]
    dictorig="dictorig="+str(dictorig)
    with open('dictorig.py','w',encoding="utf-8") as data:
        data.write(dictorig)
    return dictorig

#4. read annex of zaicz pdf and tranform to csv (main input file)
def zaicz2csv():
    try:
        from hunendict import hunendict
    except:
        print("create hunendict with getdict_huen()")
        
    try:
        from dictorig import dictorig
    except:
        print("create dictorig with getdict_huen()")
    zaiczcsv=pd.DataFrame(columns=['word','year','info','disambiguated','suffix',"en","orig","pos_hun",                                  "wordipa"]) #dffinal
    #zaicz1: year, zaicz2: origin
    path_to_pdf = r"C:\Users\Viktor\OneDrive\PhD cloud\Vorgehensweisen\loanpy6\szotar\TAMOP_annex.pdf"
    zaicz=get_pdf_file_content(path_to_pdf)
    zaicz=zaicz.replace("Valószínűleg ősi szavak","Valószínűleg ősi szavak\n") #correct typo in dictionary
    zaicz1=zaicz.split(' \n \n\n \n\n\x0cA SZAVAK EREDET SZERINTI CSOPORTOSÍTÁSA* \n\n \n \n \n \n \n \n',1)[0]
    zaicz2=zaicz.split(' \n \n\n \n\n\x0cA SZAVAK EREDET SZERINTI CSOPORTOSÍTÁSA* \n\n \n \n \n \n \n \n',1)[1]

    #zaicz1 (year):
    for index,i in enumerate(zaicz1.split('[')): #list of year-word pairs
        if ':' in i: #otherwise error
            zaiczcsv.at[index,'word']=i.split(':')[1].replace(" ","").replace("\n","")            .replace("1951-től","").replace("1000-ig","") 
            zaiczcsv.at[index,'year']=re.sub("[^0-9]", "", i.split(':')[0].split(',')[-1]) #the sure year
            zaiczcsv.at[index,'info']=i.split(':')[0][:-1].replace("\n","") #all other info

    for index,row in zaiczcsv.iterrows():
        zaiczcsv.at[index,'word']=row['word'].split(',') #explode funktioniert nur mit listen, darum split()
    #explode, reset index,drop old index,remove rows with empty cells in column "word"
    zcsv=zaiczcsv.explode('word')
    zcsv=zcsv.reset_index(drop=True)
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
    zcsv=zcsv.reset_index(drop=True) #reset index to 1,2,3,4,... again

    for index,row in zcsv.iterrows():
        if row['word'][0]=="-": #remove hyphens
            zcsv.at[index,"word"]=row["word"][1:]
            zcsv.at[index,"suffix"]="+" #mark that they're a suffix in extra column
        #insert translations
        try:
            zcsv.at[index,"en"]=str(hunendict[row["word"]]).replace("[","").replace("]","").replace("'","").replace('"','').replace(" ","_").\
            replace(",_",", ").replace("to_","")
            #b/c semsim requires a string not a list
            #b/c you can not store lists in a csv, only strings
        except KeyError:
            pass
        try:
            zcsv.at[index,"orig"]=dictorig[row["disambiguated"]]
        except KeyError:
            pass
        zcsv.at[index,"wordipa"]=epi.transliterate(row["word"])
    
    zcsv.to_csv("zcsv.csv", encoding="utf-8", index=False)

#5. add missing translations with google translate
def addgoogletrans():
    zcsv=pd.read_csv("zcsv.csv",encoding="utf-8")
    zcsv["en"]=zcsv["en"].fillna('0')
    for index,row in zcsv.iterrows():
        print(index)
        if row["en"] =='0':
            try:
                zcsv.at[index,"en"]=translator.translate(row["word"], src='hu', dest='en').text
            except:
                zcsv.to_csv("zcsv.csv", encoding="utf-8", index=False)
                sys.exit("googletrans fail")
    zcsv.to_csv("zcsv.csv", encoding="utf-8", index=False)    

#6. add pos-tags with spacy
#works on python 3.5. and spacy 2.0.12 (Hungarian pos_tagger)
#create virtual environment via anaconda navigator for this function
def getpos_hu(path): 
    zcsv=pd.read_csv(path, encoding='utf-8')
    import hu_core_ud_lg #https://github.com/oroszgy/spacy-hungarian-models
    nlp = hu_core_ud_lg.load()
    for index,row in zcsv.iterrows():
        doc=nlp(row["word"])
        for i in doc:
            zcsv.loc[[index],["pos_hun"]]=i.pos_ #df.at does not work (maybe bc older python version?)
            print(index)
    zcsv.to_csv("zaicz_in.csv", encoding="utf-8", index=False)
    return zcsv

#7. converts spacy's pos-tags to nltk's. https://spacy.io/api/annotation
def spacy2nltk_postags(path):
    zcsv=pd.read_csv(path,encoding="utf-8")
    hunposdict={"ADJ":"a","ADP":"r","ADV":"r","AUX":"v","CONJ":"r","CCONJ":"r","DET":"r","INTJ":"r","NOUN":"n",                "NUM":"r","PART":"r","PRON":"r","PROPN":"r","PUNCT":"r","SCONJ":"r","SYM":"r","VERB":"v","X":"r",                "SPACE":"r"}
    for index,row in zcsv.iterrows():
        zcsv.at[index,"pos_hun"]=hunposdict[row["pos_hun"]]
    zcsv.to_csv("zin3.csv", encoding="utf-8", index=False)
    return zcsv

#8.a translate origin-tags from Hungarian to English with google translate
def orig_hu2en(path):
    zin=pd.read_csv(path,encoding="utf-8")
    for i in list(set(zin['orig'].tolist())):
        if i==i: #exclude nans
            hu2en_origdict[i]=translator.translate(i).text

#8.b insert translated origin-tags to df
def addorig_huen(path):
    zin=pd.read_csv(path,encoding="utf-8")
    zin.insert(zin.columns.get_loc("orig")+1,"orig_en",zin['orig'].map(hu2en_origdict))
    zin.to_csv("zin5.csv", encoding="utf-8", index=False)

#9.a convert origin to tags "U, FU, Ug"
def getorigtagdict(path):
    zin=pd.read_csv(path,encoding="utf-8")
    for i in list(set(zin['orig_en'].tolist())):
        if i==i and "?" not in i:
            if "Uralic" in i: #exclude nans
                origtagdict[i]="U"
            elif "Finno-Ugric" in i:
                origtagdict[i]="FU"
            elif "Ugric" in i and "Finno-Ugric" not in i:
                origtagdict[i]="Ug"
        elif i==i and "?" in i:
            if "Uralic" in i: #exclude nans
                origtagdict[i]="U?"
            elif "Finno-Ugric" in i:
                origtagdict[i]="FU?"
            elif "Ugric" in i and "Finno-Ugric" not in i:
                origtagdict[i]="Ug?"

#9.b insert new origin-tags "U, FU, Ug"
def addorigtags(path):
    getorigtagdict(path)
    zin=pd.read_csv(path,encoding="utf-8")
    zin.insert(zin.columns.get_loc("en")+1,"orig_tag",zin['orig_en'].map(origtagdict))
    zin.to_csv("zin6.csv", encoding="utf-8", index=False)

#10. remove brackets
def rembrack(path):
    zin=pd.read_csv(path,encoding="utf-8")
    zin.rename(columns={'en':'en_human'}, inplace=True)
    zin.insert(zin.columns.get_loc("en_human")+1,"en","")
    #remove all "[","]", "'" etc
    zin['en'] = zin['en_human'].str.replace(r'[\[\]\']', '').str.replace(r'to ', '').str.replace(r' ', '_').str.replace(r',_', ', ')
    zin.to_csv("zin7.csv", encoding="utf-8", index=False)

#11. translate info-col to English
def infotrsl(row):
    mainstr=""
    for string in row.split(","):
        if r"k." in string:
            mainstr+="ca. "+re.sub(r"k.","",string)+", "
        elif "u." in string:
            mainstr+="after "+re.sub(r"u.","",string)+", "  
        elif "e." in string:
            mainstr+="before "+re.sub(r"e.","",string)+", "
        else:
            mainstr+=string+", "
        if "vége" in string:
            mainstr="end of "+re.sub(r"vége","",string)+", "
        if "eleje" in string:
            mainstr="beginning of "+re.sub(r"eleje","",string)+", "
    mainstr=re.sub(r"tn.","placename",mainstr)
    mainstr=re.sub(r"sz.","century",mainstr)
    mainstr=mainstr[:-2] #remove the ", " from the end
    return mainstr

#11.b insert translated info-col to df
def info_hu2en(path):
    zin=pd.read_csv(path,encoding="utf-8")
    zin.insert(zin.columns.get_loc("info")+1,"info_en",zin["info"].apply(infotrsl))
    zin.to_csv("zin8.csv", encoding="utf-8", index=False)
    return zin
    #zin['info_en'] = zin['info'].str.replace(r'\d{3|4}', '').str.replace(r'to ', '').str.replace(r' ', '_').str.replace(r',_', ', ')
    #zin.to_csv("zin7.csv", encoding="utf-8", index=False)

#manually replace year of "fut" (130013200) with 1300