IPA2TIPA = {  # tipa encodings of ipa chars that threw an error
"˩": r"\textipa{\tone{11}}",
"˨": r"\textipa{\tone{22}}",
"˧": r"\textipa{\tone{33}}",
"˦": r"\textipa{\tone{44}}",
"˥": r"\textipa{\tone{55}}",
"ː": r"\textipa{:}",
"t͡ʃ": r"\t{t\textipa{S}}",
"ɣ": r"\textipa{G}",
"ɟ": r"\textipa{\textbardotlessj}",
"ɒ": r"\textipa{6}",
"ɛ": r"\textipa{E}",
"ʋ̥": r"\textsubring{\textipa{V}}",
"ʔ": r"\textipa{P}"
}

def process_tex_file(input_filename, output_filename):
    """
    see https://jon.dehdari.org/tutorials/tipachart_mod.pdf
    and https://ptmartins.info/tex/tipacheatsheet.pdf
    """
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        content = input_file.read()

    content = content.replace(  # declare tipa package
        r'\usepackage{babel}',
        r'\usepackage{babel}' + '\n' + r'\usepackage[tone]{tipa}')
    for i in IPA2TIPA:
        content = content.replace(i, IPA2TIPA[i])

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(content)


if __name__ == "__main__":
    process_tex_file('latex/loanpy.tex', 'latex/loanpy2.tex')
