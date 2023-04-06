.. _overview:

Documentation overview
======================

.. note::

   If you haven't already installed MOABB, please take a look
   at our `installation guides <install/install>`__. Please also kindly find some
   resources for `learn_python <https://mne.tools/stable/overview/learn_python.html>`__
   from MNE if you need to, and the basic module from
   `NeuroMatch Academy <https://deeplearning.neuromatch.io/tutorials/W1D1_BasicsAndPytorch/student/W1D1_Tutorial1.html>`__.

The documentation for MOABB is divided into three main sections:

1. The :doc:`Getting Started<../auto_tutorials/index>` provide a sequential tutorial to start to
   use MOABB. They are designed to be read in **order**, and provide detailed
   explanations, sample code, and expected output for the most common MOABB
   analysis tasks. The emphasis is on thorough explanations that get new users
   up to speed quickly, at the expense of covering only a limited number of
   topics.

2. MOABB comes with working code samples that exhibit various modules and techniques.
   The examples are categorized into the four categories:
   :ref:`Simple <../auto_examples/index.html#simple-examples>`,
   :ref:`Advanced <../auto_examples/index.html#advanced-examples>`,
   :ref:`External <../auto_examples/index.html#external-examples>`,
   and :ref:`Evaluation <../auto_examples/index.html#evaluation-with-learning-curve>`.
   While these examples may not provide the same level of descriptive explanations as
   tutorials, they are a beneficial resource for discovering novel ideas for analysis
   or plotting. Moreover, they illustrate how to use MOABB to implement specific module.

3. The :doc:`API reference <api>` that provides documentation for
   the classes, functions and methods in the MOABB codebase. This is the
   same information that is rendered when running
   :samp:`help(moabb.{<function_name>})` in an interactive Python session, or
   when typing :samp:`moabb.{<function_name>}?` in an IPython session or Jupyter
   notebook.


The rest of the MOABB documentation pages are shown in the navigation menu,
including the :ref:`list of example datasets<data_summary>`, information about
the `MOABB license <https://github.com/NeuroTechX/moabb/blob/develop/LICENSE>`__,
`how to contribute <CONTRIBUTING.html>`__ to MOABB,
:doc:`how to cite MOABB <cite>`, and explanations of the
external library dependencies that MOABB uses, including Deep Learning, Code Carbon,
Docs and others.

.. toctree::
    :hidden:

    Getting Started <auto_tutorials/index>
    Gallery <auto_examples/index>
    API <api>
