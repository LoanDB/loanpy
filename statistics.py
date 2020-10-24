def statistics(dfstat):
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

    return content