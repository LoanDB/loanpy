from lingpy import ipa2tokens

cns="jwʘǀǃǂǁk͡pɡ͡bcɡkqɖɟɠɢʄʈʛbb͡ddd̪pp͡ttt̪ɓɗb͡βk͡xp͡ɸq͡χɡ͡ɣɢ͡ʁc͡çd͡ʒt͡ʃɖ͡ʐɟ͡ʝʈ͡ʂb͡vd̪͡z̪d̪͡ðd̪͡ɮ̪d͡zd͡ɮd͡ʑp͡ft̪͡s̪t̪͡ɬ̪t̪͡θt͡st͡ɕt͡ɬxçħɣʁʂʃʐʒʕʝχfss̪vzz̪ðɸβθɧɕɬɬ̪ɮʑɱŋɳɴmnn̪ɲʀʙʟɭɽʎrr̪ɫɺɾhll̪ɦðʲt͡ʃʲnʲʃʲC"
vow="ɑɘɞɤɵʉaeiouyæøœɒɔəɘɵɞɜɛɨɪɯɶʊɐʌʏʔɥɰʋʍɹɻɜ¨ȣ∅" 

def flatten(mylist):
    return [item for sublist in mylist for item in sublist]

def word2struc(ipaword): #in: "baba" out: "CVCV"
    return "".join([(lambda x: "C" if (x in cns) else ("V" if (x in vow) else ""))(i) for i in ipa2tokens(ipaword, merge_vowels=False, merge_geminates=False)])

def ipa2clusters(ipaword): 
    return [j for j in "".join([(lambda x: "€"+x+"€" if x[0] in vow else x)(i) for i in ipa2tokens(ipaword, merge_vowels=True)]).split("€") if j]

def list2regex(sclist):
    return "" if sclist == ["0"] else ("("+"".join([i+"|" for i in sclist])[:-1].replace("|0","").replace("0|","")+")"+"?" if "0" in sclist and sclist != ["0"]\
                                       else "("+"".join([i+"|" for i in sclist])[:-1]+")")