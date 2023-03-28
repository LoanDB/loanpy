.. _tutorial:

Tutorial
========

This tutorial will walk you through the process of using loanpy to analyze loanword adaptation patterns.

Step 1: Installing loanpy
--------------------------

Before you can use loanpy, you'll need to install it on your computer. You can do this using pip:

.. code-block:: console

   pip install loanpy

Step 2: Importing loanpy
------------------------

Once you've installed loanpy, you can import it into your Python script using the `import` statement:

.. code-block:: python

   import loanpy

Step 3: Loading data
---------------------

To analyze loanword adaptation patterns, you'll need some data to work with. You can load data from a CSV file using the `load_data` function:

.. code-block:: python

   data = loanpy.load_data('mydata.csv')

Step 4: Analyzing data
-----------------------

Now that you have some data loaded into memory, you can analyze it using loanpy's functions. For example, you can use the `analyze_adaptation` function to compute statistics on loanword adaptation patterns:

.. code-block:: python

   results = loanpy.analyze_adaptation(data)

Step 5: Visualizing results
----------------------------

Finally, you can visualize your results using loanpy's plotting functions. For example, you can use the `plot_results` function to create a bar chart of the adaptation statistics:

.. code-block:: python

   loanpy.plot_results(results)

Conclusion
----------

Congratulations, you've completed this tutorial on loanpy! You should now have a good understanding of how to use loanpy to analyze loanword adaptation patterns in your own data. If you have any questions or feedback, please don't hesitate to reach out to the loanpy development team.

Further Reading
---------------

For more information on loanpy and its functions, please refer to the API documentation and user guide on the loanpy website: https://loanpy.org/

