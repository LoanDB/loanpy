# LOANPY

[![loanpy](https://github.com/martino-vic/Framework-for-computer-aided-borrowing-detection/blob/master/white_logo_transparent_background.png)](https://github.com/martino-vic/Framework-for-computer-aided-borrowing-detection)  
[![Build Status](https://about.zenodo.org/static/img/logos/zenodo-gradient-square.svg)](https://zenodo.org/record/4009627#.X26Z_2gzaUk)

loanpy is a tool for historical linguists. It extracts sound changes and constraints from etymological dictionaries, generates pseudo-roots of L1, pseudo- sound substituted forms of L2, searches for matches and ranks them according to semantic similarity.

### Installation

Type _cmd_ into the search bar, open Command Prompt and type:

```sh
$ python -m pip install loanpy
```

### Getting started
Change your working directory:
```
>>> import os
>>> os.chdir(lp.__file__[:-(len("\loanpy.py"))]+r"\data")
```
Get the path of your new working directory:
```
>>> import os
>>> print(str(os.getcwd())[:-(len("\loanpy.py"))]+r"\data")
```
Download [pretrained vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (3 Gigabyte) and save them to your new working directory. This step is needed to enable sorting phonetic matches according to their semantics. You can use your own vectors as well but make sure to name them "GoogleNews-vectors-negative300.bin".

Start python and type:

```sh
>>> from loanpy import loanpy as lp
>>> import pandas as pd
```
Define your input file.  
loanpy comes with a csv generated from the annex of [Gábor Zaicz's Hungarian etymological dictionary from 2006](https://regi.tankonyvtar.hu/hu/tartalom/tinta/TAMOP-4_2_5-09_Etimologiai_szotar/adatok.html)  
If you use your own csv make sure to keep the same column names as in zaicz_in.csv 
```
>>> myinput = pd.read_csv("zaicz_in.csv",encoding="utf-8")
```

Filter your input according to year and/or origin.
E.g. if you want to keep only words of unknown origin that appear in texts before 1600 tpye:
```
>>> myinput=myinput[myinput.year <= 1600]
>>> myinput=myinput[myinput.origin == "G) ISMERETLEN EREDETŰ SZAVAK "]
```

Type in the timelayer in which you want to search for loans ("U" for Proto-Uralic, "FU" for Proto-Finno-Ugric or "Ug" for Proto-Ugric) and view your results in bestof[timelayer].csv (e.g. bestofUg.csv)

```
>>> lp.loandf("Ug",myinput)
```
### Other functions
Get phonetic matches and their semantic similarity score for a single word. Output is not sorted.  
(Use "n" for nouns, "v" for verbs, "a" for adjectives and "r" for everything else)
```
>>> import epitran
>>> epi = epitran.Epitran('hun-Latn')
>>> lp.loan(layer="Ug",wordipa=epi.transliterate("bor"),en="wine",word="bor",pos_hun="n")
```
Get phonetic matches without a semantic similarity score.  
```
>>> import epitran
>>> epi = epitran.Epitran('hun-Latn')
>>> lp.loan_nosem(layer="Ug",wordipa=epi.transliterate("bor"),en="wine",word="bor",pos_hun="n")
```

Get semantic similarity of two words.
```
>>> lp.semsim("wine","beer","n","n")
```

Delete duplicate word pairs by keeping only the ones with the most likely reconstructed root
```
>>> inputfile=pd.read_csv("bestofUg.txt",encoding="utf-8")
>>> lp.delbynse(inputfile)
```
Add all other words from the etymological dictionary that exhibit the same sound changes
```
>>> inputfile=pd.read_csv("bestofUg.txt",encoding="utf-8")
>>> lp.addexamples(inputfile)
```
Get a list of pseudo-roots
```
>>> import epitran
>>> epi = epitran.Epitran('hun-Latn')
>>> lp.settimelayer("U")
>>> lp.structure(lp.shuffle(lp.posy(epi.transliterate("bor"))))
```


License
----

Academic Free License (AFL) (Creative Commons Attribution 4.0 International)