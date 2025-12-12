:html_theme.sidebar_secondary.remove:

.. title:: MOABB - Mother of all BCI Benchmarks

.. The page title must be in rST for it to show in next/prev page buttons.
   Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">
    h1 {display:none;}
    /* Hide the first paragraph from README (the big logo/title) since we have hero section */
    .bd-article-container > section > p:first-of-type {display:none;}
    </style>

MOABB Homepage
==============

.. include:: README.md
   :parser: myst

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 10
   :caption: Main classes of MOABB:
   :titlesonly:

   The largest EEG benchmark <paper_results>
   Datasets <dataset_summary>
   Installation <install/install>
   Examples <auto_examples/index>
   API <api>
   Citation <cite>
   Release notes <whats_new>
