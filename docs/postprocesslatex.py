"""

#. Add tipa package import to preamble
#. Replace badges with plain links
#. Replace IPA characters that throw errors in latex2pdf conversion
with commands for the tipa LaTeX-package.

"""
TIPAPREAMBLE = {
r'\usepackage{babel}':
r'\usepackage{babel}' + '\n' + r'\usepackage[tone]{tipa}'
}

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

REPLACEBADGES = {  # plain links, no badges

r"\sphinxhref{https://pypi.org/project/loanpy/}" +
r"{\sphinxincludegraphics{{/home/viktor/Documents/GitHub/loanpy/docs" +
r"/doctrees/images/a7198e3990cc11cb2969741e8a1e7e70a75f7e35/loanpy}.svg}}":

r"https://pypi.org/project/loanpy/",

r"\sphinxhref{https://doi.org/10.5281/zenodo.7893906}" +
r"{\sphinxincludegraphics{{/home/viktor/Documents/GitHub/loanpy/docs" +
r"/doctrees/images/7dd55bbb890ac04fada8200c15238c911bd9bcd5/zenodo.7893906}" +
r".svg}}":

r"https://doi.org/10.5281/zenodo.7893906",

r"\sphinxhref{https://loanpy.readthedocs.io/en/latest/?badge=latest}" +
r"{\sphinxincludegraphics{{/home/viktor/Documents/GitHub/loanpy/docs/" +
r"doctrees/images/" +
r"d5ccc73d2179a12dec92e6b0895418318e6fa182/c0d4a69ef485d07c560444afafbd0caf82a6e90d}" +
r".svg}}":

r"https://loanpy.readthedocs.io/en/latest/",

r"\sphinxhref{https://coveralls.io/github/LoanpyDataHub/loanpy}" +
r"{\sphinxincludegraphics{{/home/viktor/Documents/GitHub/loanpy/docs" +
r"/doctrees/images/17e79bc5e6c30d3b2e1c75d1dc2ab5099ef310ea/badge}.svg}}":

r"https://coveralls.io/github/LoanpyDataHub/loanpy",

r"\sphinxhref{https://dl.circleci.com/status-badge/redirect/gh/" +
r"LoanpyDataHub/loanpy/tree/main}{\sphinxincludegraphics" +
r"{{/home/viktor/Documents/GitHub/loanpy/" +
r"docs/doctrees/images/3a88e66b9a00f3e37881a996a46b2eff1384467b/main}.svg}}":

r"https://dl.circleci.com/status-badge/redirect/gh/" +
r"LoanpyDataHub/loanpy/tree/main"
}

def process_tex_file(input_filename, output_filename):
    """
    #. Read the file from specified path
    #. Apply changes from dictionaries defined on top
    #. Write file to specified path

    For ipa2tipa cheat sheets see
    https://jon.dehdari.org/tutorials/tipachart_mod.pdf
    and https://ptmartins.info/tex/tipacheatsheet.pdf
    """
    with open(input_filename, 'r', encoding='utf-8') as input_file:
        content = input_file.read()

    for dictionary in [TIPAPREAMBLE, IPA2TIPA, REPLACEBADGES]:
        for key in dictionary:
            content = content.replace(key, dictionary[key])

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(content)


if __name__ == "__main__":
    process_tex_file('latex/loanpy.tex', 'latex/loanpy2.tex')
