.. meta::
   :description: The largest EEG-based BCI reproducibility benchmark results. Compare within-session, cross-session, and cross-subject accuracy across motor imagery, P300, and SSVEP paradigms.
   :keywords: BCI benchmark results, EEG benchmark, motor imagery accuracy, P300 accuracy, SSVEP accuracy, within-session evaluation
   :hide_sidebar: true

:html_theme.sidebar_secondary.remove:
:html_theme.sidebar_primary.remove:

.. _paper_results:
.. raw:: html

   <!-- Must import jquery before the datatables css and js files. -->
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.3.1/css/dataTables.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.3.1/js/dataTables.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/fixedcolumns/5.0.4/css/fixedColumns.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/fixedcolumns/5.0.4/js/dataTables.fixedColumns.js"></script>
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/fixedcolumns/5.0.4/js/fixedColumns.dataTables.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/3.0.2/css/buttons.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/3.0.2/js/dataTables.buttons.js"></script>
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/buttons/3.0.2/js/buttons.html5.js"></script>

   <style>
    table.dataTable {
      font-size: 1.05rem;
    }
    div.dt-scroll-head div table.dataTable thead th {
      padding-right: 5px !important;
      font-size: 1.05rem;
    }
    table.dataTable td {
      text-align: left;
    }
    html[data-theme="dark"] .dtfc-fixed-left,
    html[data-theme="dark"] .dtfc-fixed-start {
      background-color: #1e1e1e !important;
      color: #ffffff !important;
    }

    /* DataTables button styling */
    .dt-buttons .dt-button {
      display: inline-flex !important;
      align-items: center;
      gap: 6px;
      padding: 6px 16px !important;
      border-radius: 999px !important;
      border: 1px solid var(--pst-color-border, #ccc) !important;
      background: transparent !important;
      color: var(--pst-color-text-base, #333) !important;
      font-size: 0.875rem !important;
      font-weight: 500 !important;
      cursor: pointer;
      transition: border-color 0.2s, transform 0.15s, box-shadow 0.2s !important;
      line-height: 1.4 !important;
    }
    .dt-buttons .dt-button:hover {
      border-color: #007CBA !important;
      color: #007CBA !important;
      transform: translateY(-1px);
      box-shadow: 0 2px 6px rgba(0, 124, 186, 0.15);
      background: transparent !important;
    }
    .dt-buttons .dt-button:active {
      transform: scale(0.96) !important;
      box-shadow: none !important;
    }
    .dt-buttons .dt-button:focus {
      outline: 2px solid #007CBA !important;
      outline-offset: 2px !important;
    }
    .dt-buttons .dt-button svg {
      width: 16px;
      height: 16px;
      flex-shrink: 0;
      fill: currentColor;
    }

    /* Dark mode overrides */
    html[data-theme="dark"] .dt-buttons .dt-button {
      border-color: rgba(255, 255, 255, 0.2) !important;
      color: #e0e0e0 !important;
    }
    html[data-theme="dark"] .dt-buttons .dt-button:hover {
      border-color: #3db5e6 !important;
      color: #3db5e6 !important;
      box-shadow: 0 2px 6px rgba(61, 181, 230, 0.2);
    }
   </style>

.. currentmodule:: moabb.datasets

The largest EEG-based Benchmark for Open Science
================================================

We report the results of the benchmark study performed in:
`The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark <https://universite-paris-saclay.hal.science/hal-04537061v1/file/MOABB-arXiv.pdf>`_

.. figure:: images/datasets_largest.png
   :alt: MOABB Benchmark datasets
   :align: center

   Visualization of the MOABB datasets, with Motor Imagery (MI) in green, ERP in pink/purple and SSVEP in orange/brown. The size of the circle is proportional to the number of subjects and the contrast depends on the number of electrodes.

This study conducts an extensive Brain-computer interfaces (BCI) reproducibility analysis on open electroencephalography datasets,
aiming to assess existing solutions and establish open and reproducible benchmarks for effective comparison within the field. Please note that the results are obtained using `Within-Session evaluation <https://moabb.neurotechx.com/docs/generated/moabb.evaluations.WithinSessionEvaluation.html>`_.
The results are reported regarding mean accuracy and standard deviation across all folds for all sessions and subjects.

If you use the same evaluation procedure, you should expect similar results if you use the same pipelines and datasets, with some minor variations due to the randomness of the cross-validation procedure.

**You can copy and use the table in your work**, but please `**cite the paper** <http://moabb.neurotechx.com/docs/cite.html>`_ if you do so.

Motor Imagery
=============

Motor Imagery is a BCI paradigm where the subject imagines performing a movement.
Each imagery task is associated with a different class, and each task has its difficulty level related to how the brain generates the signal.

Here, we present three different scenarios for Motor Imagery classification:

#. **Left vs Right Hand**: We use only the classes Left Hand and Right Hand.
#. **Right Hand vs Feet**: We use only Right Hand and Feet classes.
#. **All classes**: We use all the classes in the dataset, when there are more than classes that are not Left Hand and Right Hand.

All the results here are for **within-session evaluation**, a 5-fold cross-validation, over the subject's session.


Motor Imagery - Left vs Right Hand
===================================

**Left vs Right Hand**: We use only Left Hand and Right Hand classes.

.. raw:: html
   :file: results/within_session_mi_left_vs_right_hand.html

.. raw:: html

   <hr>

Motor Imagery - Right Hand vs Feet
==================================

**Right Hand vs Feet**: We use only Right Hand and Feet classes.

.. raw:: html
   :file: results/within_session_mi_right_hand_vs_feet.html

.. raw:: html

   <hr>

Motor Imagery - All classes
===========================

**All classes**: We use all the classes in the dataset, when there are more classes that are not Left Hand and Right Hand.

.. raw:: html
   :file: results/within_session_mi_all_classes.html

.. raw:: html

   <hr>

SSVEP (All classes)
===================

Here, we have the results of the within-session evaluation, a 5-fold cross-validation, over the subject's session.
We use all the classes available in the dataset.

.. raw:: html
   :file: results/within_session_ssvep_all_classes.html

.. raw:: html

   <hr>

P300/ERP (All classes)
======================

Here, we have the results of the within-session evaluation, a 5-fold cross-validation, over the subject's session.
We use all the classes available in the dataset.

.. raw:: html
   :file: results/within_session_erp_p300_all_classes.html

.. raw:: html

   <hr>

  <script type="text/javascript">
     $(document).ready(function() {
       var copyIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>';
       var checkIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>';
       var csvIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>';

       $(".sortable").each(function() {
         const $table = $(this);

         $table.DataTable({
           fixedColumns: true,
           order: [[1, "desc"]],
           bJQueryUI: true,
           scrollX: true,
           paging: false,
           scrollCollapse: true,
           info: false,
           searching: false,
           layout: {
             topStart: {
               buttons: [
                 {
                   extend: 'copy',
                   text: copyIcon + ' Copy Table',
                   action: function(e, dt, node, config) {
                     var self = this;
                     $.fn.dataTable.ext.buttons.copyHtml5.action.call(self, e, dt, node, config);
                     var $btn = $(node);
                     var originalText = $btn.html();
                     $btn.html(checkIcon + ' Copied!');
                     setTimeout(function() {
                       $btn.html(originalText);
                     }, 2000);
                   }
                 },
                 {
                   extend: 'csv',
                   text: csvIcon + ' Export CSV',
                   title: 'moabb_benchmark'
                 }
               ],
             },
           },
         });
       });
     });
   </script>

.. toctree::
   :glob:
   :hidden:
   :caption: MOABB Results
   :titlesonly:
