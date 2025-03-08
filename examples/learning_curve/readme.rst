Evaluation with learning curve
------------------------------

These examples demonstrate how to make evaluations using only a subset of
available example. For example, if you consider a dataset with 100 trials for
each class, you could evaluate several pipelines by using only a fraction of
these trials. To ensure the robustness of the results, you need to specify the
number of permutations. If you use 10 trials per class and 20 permutations,
each pipeline will be evaluated on a subset of 10 trials chosen randomly, that
will be repeated 20 times with different trial subsets.

.. toctree::
   :hidden:
