# Welcome to the development version of loanpy

The purpose of this branch is to continuously update, share and improve code and data related to my dissertation project [loanpy](https://pypi.org/project/loanpy/). This section is under constant change. To see the last stable versioin visit branch "master". Results can be viewed under [tests/sanity](https://github.com/martino-vic/loanpy/tree/development/sanity_check).
## Content

- data: the local input files loanpy is working with
- sanity_checks: results of the sanity checks, will later be migrated to [another repository](https://github.com/martino-vic/results_loanpy)

	- Checks done so far:

		- Predicting loanword adaptation from
			- [English into Maori](https://github.com/martino-vic/loanpy/tree/development/sanity_check/wiktionary) (Wiktionary, datadriven)
			- [English into Maori](https://github.com/martino-vic/loanpy/tree/development/sanity_check/wiktionary/heuristics) (Wiktionary, heuristics)
			- [English into Maori](https://github.com/martino-vic/loanpy/tree/development/sanity_check/Duval1995) (Duval1995, datadriven)
			- [English into Maori](https://github.com/martino-vic/loanpy/tree/development/sanity_check/Duval1995/heuristic), (Duval1995, heuristics)

	- Upcoming checks:

		- English loans in the world's languages ([GLAD](https://project.nhh.no/Anglicisms/HTMLClient/default.htm) vs Wiktionary, data vs. heuristics driven)
		- Predicting reconstructions from [uralonet](uralonet.nytud.hu): PFU, U, Ug, datadriven (no heuristics possible with historical sound changes)
		- Predicting adaptation of Proto-Indo-Iranian loanwords into Proto-Hungarian ([Holopainen 2019](https://helda.helsinki.fi/handle/10138/307582))
		- Reconstructing modern Hungarian words of Proto-Indo-Iranian origin into Proto-Hungrian ([Holopainen 2019](https://helda.helsinki.fi/handle/10138/307582))
		- Check how info about English loanwords in Hungarian influences the prediction of German ones and vice versa
		- Try to detect older Germanic loanowrds in Finnish (sources: [LÄGLOS](https://brill.com/view/title/30051?language=de), [Bedlan](https://github.com/lexibank/uralex))
		- Try to detect Gothic loanwords in Hungarian
		
	- Other tasks in the pipeline:
		- update loanfinder.py (inherit from classes)
		- instead of ROC curve like now take new criterion: How well does loanfinder.py distinguish loans from non-loans?
		- compare performance to AI (evaluation method: Levenshtein distance instead of false positive rate)
		- baseline tests: How many loanwords are found where there clearly aren't any?
		- Refactor, stylecheck, add docstrings, write tests, upload the next stable version
		
	- Outlook:
		- other proto languages jff (e.g. pgm+hun)
		- simulate lw "if they had been incorporated back then they'd sound like this". For more info see [this Tweet](https://twitter.com/martino_vik/status/1471702889483706369?s=20)
		- Try to get [Te Aka](https://maoridictionary.co.nz/) and [UESZ](http://uesz.nytud.hu/) as csvs somehow and use them as data sources

## Notes:

- Add Badges to readme: "Made with Python", "Open in Collab", "PyTorch", "Zenodo", "cldf", "orcid", "tests passing", "pypiversion" like [here](https://github.com/Rohith04MVK/AI-Art-Generator) and [here](https://github.com/cldf/pycldf)
- rename function "editops" to "edit script", see https://hal.archives-ouvertes.fr/hal-01360482/file/LATA2016.pdf
- Impossible to say how Levenshtein distance translates to false positive rate because there's no algorithm to calculate the k-neigborhood. Related Papers:  https://hal.archives-ouvertes.fr/hal-01360482/file/LATA2016.pdf, https://ir.cwi.nl/pub/30171/LIPIcs-CPM-2020-10.pdf

Visit [tests/sanity](https://github.com/martino-vic/loanpy/tree/development/sanity_check) to read more about the input and output