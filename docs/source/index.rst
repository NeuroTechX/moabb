:html_theme.sidebar_secondary.remove:

.. title:: MOABB - Mother of all BCI Benchmarks

.. The page title must be in rST for it to show in next/prev page buttons.
   Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">
    /* Keep the page H1 for next/prev buttons, but visually hide it on the homepage. */
    section#moabb-homepage > h1 {
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }

    /* Hide the GitHub README header block on the docs homepage (we have a hero above). */
    .moabb-readme-header {display:none;}
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
