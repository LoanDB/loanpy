Tutorial
========

This tutorial will walk you through the process of using loanpy to
discover old loanwords.

Here is an illustration of the full workflow with a minimal example:

.. figure:: images/workflow.png
   :alt: The image shows a workflow chart with a turquoise bubble on top
         saying "kiki < gigi ← gege". Two arrows point away and towards it
         both on its left and on its right. The ones pointing away say "mine",
         and the ones pointing towards it "evaluate". The ones pointing
         away point towards a green box each. The box on the left reads
         "k<g, i<i" and the box on the right "g<g, i<e". There are two
         bigger turquoise 3x3-tables underneath the green boxes. The one on the
         left looks like this: The left column
         reads "Candidate Recipient Form, ikki, iikk", the middle column
         "Reconstruct hypothetical past form, iggi, iigg", and the right
         "Meaning, cat, cold". There's a yellow curved arrow
         above it, going from the left column up towards the green box and
         bending down and pointing to the second column. It says "apply" in
         the middle of its arch. The other 3x3 table is a mirrored version of
         this. Its left column reads "Meaning, hot, dog", its middle one
         "Adapt hypothetical recipient form, igig, iggi" and its right
         "Candidate Donor Form, egeg, egge". The yellow-apply arrow points
         from the right column to the middle one. There is a big yellow curved
         arrow on the bottom too, pointing from the middle column of the left
         3x3 table, to the middle column of the right 3x3 table. Above its
         arch it says "Find new etymology: ikki “cat” < iggi ← egge “dog”"

   The overall workflow with a minimal example: Mine sound correspondences
   from an etymological dictionary, evaluate their predictive power,
   input them to a sound change applier to generate pseudo-adapted and
   pseudo-proto-forms, search for phono-semantic matches between them.

Step 1: Mine sound correspondences
----------------------------------

Grab an etymological dictionary and mine information of how sounds
and phonotactic patterns changed during horizontal and vertical transfers.
Find a detailed guide in `Part 3 (steps 1-4) of ronataswestoldturkic's
documentation
<https://ronataswestoldturkic.readthedocs.io/en/latest/mkloanpy.html>`_.
In the minimal example, our dictionary contains only one etymology, namely
a horizontal transfer "gigi ← gege" and a vertical one "kiki < gigi".
If we mine the sound correspondences we get the rule "g from g, i from e"
in horizontal transfers and "k from g, i from i" in vertical ones.
In terms of phonotactics, we can mine "CVCV from CVCV" both horizontally
and vertically.

Step 2: Apply sound correspondences
-----------------------------------

Take the information mined from the etymological dictionary,
combine it with heuristics, and apply it to unseen words.
Create hypothetical proto-Hungarian forms by simulating their historical
changes based on modern Hungarian words as input. Simulate loandword
adaptation of Gothic words into proto-Hungarian.

For a detailed guide on predicting vertical (historical) transfers see
`gerstnerhungarian's documentation
<https://gerstnerhungarian.readthedocs.io/en/latest/?badge=latest>`_ and for
horizontal transfers (loanwords) see
`koeblergothic's documentation
<https://koeblergothic.readthedocs.io/en/latest/?badge=latest>`_.

Step 3: Find old loanwords
--------------------------

To find old loanwords by searching for phonetic and semantic overlaps
within the predicted forms, follow the `GothicHungarian's documentation
<https://gothichungarian.readthedocs.io/en/latest/?badge=latest>`_.

Conclusion
----------

Congratulations, you've completed this tutorial on loanpy! You should now
have a good understanding of how to use loanpy to find old loanwords.

If you have any questions or feedback, please don't hesitate to reach out
to me, e.g. via `e-mail <mailto:viktor_martinovic@$removethis$eva.mpg.de>`_ or
`Twitter <https://twitter.com/martino_vik>`_.

Further Reading
---------------

LoanPy was part of my dissertation-project at the Finno-Ugric department
of the University of Vienna. A link to the monograph will be made public
as soon as available.
