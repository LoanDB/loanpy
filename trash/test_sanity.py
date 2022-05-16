from loanpy_development import sanity

def validate():
#1. öffne dfargs.
#2. lies die oberste Zeile
#3. hol dir die Arguemente von dort
#4. initiiere die Klasse ROC_ad mit diesen Argumenten
#5. Schau ob ROC_ad.out ident ist mit der Version auf Github
#6. Wenn ja: Meldung + löschen
#7. Wenn nein: Meldung + beibehalten

    o = pd.DataFrame("dfargs.csv").head(1)
    res = sanity.ROC_ad(ptct_thresh=o.iloc[0,6], hm_struc_ceiling=o.iloc[0,11], hm_paths_ceiling=o.iloc[0,12],
                 only_documented_clusters=o.iloc[0,16], sort_by_nse=o.iloc[0,17], struc_filter=o.iloc[0,18])
                 
    pass