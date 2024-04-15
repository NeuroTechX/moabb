:html_theme.sidebar_secondary.remove:
.. _paper_results:

.. currentmodule:: moabb.datasets

Benchmark
===================================

Text to introduce the benchmarking results. Create the citation for the benchmarking paper.


Motor Imagery - All classes
=============================

.. csv-table:: Motor Imagery - All classes
   :header: Pipelines,:class:`AlexMI`,:class:`BNCI2014_001`,:class:`PhysionetMI`,`HighGamma`_,:class:`Weibo2014`,:class:`Zhou2016`
   :class: sortable, datatable

    `ACM+TS+SVM`_,69.37±15.07,77.82±12.23,55.44±14.87,82.50±10.20,63.89±11.01,85.25±4.06
    `CSP+LDA`_,61.04±17.22,65.99±15.47,47.73±14.35,72.97±10.42,39.45±11.87,82.96±5.20
    `CSP+SVM`_,62.92±16.89,66.88±15.22,48.52±14.62,75.89±10.55,44.08±11.95,83.08±5.33
    `DLCSPauto+shLDA`_,60.63±17.91,66.31±15.36,46.85±14.65,72.82±10.44,38.84±11.97,82.06±5.57
    `DeepConvNet`_,37.71±4.56,35.29±8.26,27.68±3.91,56.78±18.11,24.17±9.80,55.69±5.61
    `EEGITNet`_,36.04±3.43,35.55±6.35,26.15±4.95,70.44±14.68,25.78±8.00,50.68±16.27
    `EEGNeX`_,37.71±9.64,45.62±15.29,26.69±5.64,67.56±14.15,30.22±11.02,56.42±11.29
    `EEGNet_8_2`_,43.96±8.62,60.46±20.20,29.04±7.03,76.99±13.05,35.35±14.05,83.34±3.58
    `EEGTCNet`_,34.17±1.86,41.65±13.73,25.79±3.85,71.11±11.96,17.95±3.88,37.19±2.57
    `FilterBank+SVM`_,65.00±17.56,66.53±12.05,45.49±12.54,75.94±8.59,45.21±10.05,81.99±4.65
    `FgMDM`_,65.63±15.63,70.14±15.13,55.04±14.17,82.97±10.08,56.94±9.26,83.07±4.96
    `MDM`_,60.62±13.69,61.60±14.20,42.96±12.98,52.03±10.11,33.41±8.67,76.05±7.10
    `ShallowConvNet`_,50.00±12.94,72.47±16.50,41.87±12.50,85.13±9.57,48.94±10.36,85.02±3.78
    `TS+EL`_,69.79±13.75,72.38±14.85,59.93±14.07,85.53±9.40,63.84±8.77,84.54±4.93
    `TS+LR`_,69.17±14.79,71.97±15.46,58.55±14.06,84.60±9.28,62.76±8.39,84.88±4.63
    `TS+SVM`_,67.92±12.74,70.76±15.08,58.46±15.15,84.41±9.56,61.47±9.62,83.66±4.55

Motor Imagery - Left vs Right Hand
===================================

.. csv-table:: Motor Imagery - Left vs Right Hand
   :header: Pipelines,:class:`BNCI2014_001`,:class:`BNCI2014_004`,:class:`Cho2017`,`Grosse2009`_,:class:`Lee2019_MI`,:class:`PhysionetMI`,`HighGamma`_,:class:`Shin2017A`,:class:`Weibo2014`,:class:`Zhou2016`
   :class: sortable

    `ACM+TS+SVM`_,91.71±10.30,82.67±15.33,73.56±14.54,86.60±15.12,83.05±13.97,63.55±21.24,85.82±13.98,68.97±23.45,84.78±13.33,95.03±4.76
    `CSP+LDA`_,82.34±17.26,80.10±14.93,71.38±14.54,76.44±20.95,76.88±17.41,65.75±17.37,77.23±18.43,72.30±21.79,80.72±15.29,93.15±6.88
    `CSP+SVM`_,83.07±16.53,79.27±15.68,71.92±14.25,77.81±21.27,77.27±16.73,65.71±17.90,79.24±20.07,70.11±22.19,79.84±15.86,92.96±7.86
    `DLCSPauto+shLDA`_,82.75±16.69,79.87±15.11,71.16±14.53,76.40±20.83,76.69±17.23,65.07±17.68,77.02±18.48,70.34±23.30,80.16±15.23,92.56±7.21
    `DeepConvNet`_,82.07±15.52,72.36±18.53,71.67±12.91,82.38±15.39,70.65±15.76,59.57±16.77,81.23±17.39,56.03±19.18,73.64±15.78,94.42±6.21
    `EEGITNet`_,75.27±16.37,65.10±15.32,57.20±12.21,72.19±14.71,59.17±11.72,52.71±11.11,74.66±20.52,52.18±16.78,59.35±14.06,69.41±14.66
    `EEGNeX`_,66.28±13.22,66.53±17.10,53.28±10.60,57.00±7.52,55.12±10.05,51.20±10.63,68.58±19.37,49.02±17.58,57.97±15.65,61.56±14.60
    `EEGNet_8_2`_,77.15±19.33,69.50±19.50,66.79±16.34,83.02±18.08,65.67±16.43,59.55±15.95,80.20±18.13,57.99±17.28,66.46±21.78,94.84±2.83
    `EEGTCNet`_,67.46±20.81,69.70±19.55,58.34±12.63,68.45±16.27,55.68±12.75,55.90±12.74,75.62±22.33,51.26±16.77,63.16±18.32,82.24±9.40
    `FilterBank+SVM`_,84.44±16.00,80.39±16.05,67.91±15.63,79.65±18.63,75.07±16.97,58.45±13.93,81.44±17.89,65.63±21.64,76.81±18.88,92.64±5.01
    `FgMDM`_,86.53±12.14,79.28±15.25,72.90±12.70,87.02±13.20,81.34±13.93,68.46±19.06,86.71±13.79,70.86±23.36,78.41±14.85,92.54±6.67
    `LogVariance+LDA`_,77.96±15.09,78.51±15.25,64.49±10.08,78.71±11.69,66.21±12.06,61.94±14.41,78.44±13.76,61.78±22.77,74.13±10.40,88.39±8.57,
    `LogVariance+SVM`_,75.86±16.45,78.30±15.18,65.46±11.71,81.73±12.40,73.83±13.85,62.35±16.87,79.42±13.66,61.38±22.68,74.85±11.33,88.47±8.50
    `MDM`_,81.69±14.94,77.66±15.78,63.39±13.69,64.29±8.04,70.23±13.87,54.76±16.79,61.53±16.41,62.99±21.25,58.80±16.13,90.70±7.11
    `ShallowConvNet`_,86.17±13.74,72.36±18.05,73.84±14.95,86.53±13.00,75.83±15.04,65.19±15.80,84.82±15.29,60.80±19.27,79.10±12.63,95.65±5.55
    `TRCSP+LDA`_,79.84±16.28,79.78±15.22,71.85±13.84,78.29±16.66,76.26±15.41,67.24±17.23,79.14±15.91,67.30±23.19,79.33±14.43,93.53±6.38
    `TS+EL`_,86.44±13.20,79.75±15.44,76.23±14.21,89.25±12.00,84.74±13.19,67.91±20.03,88.65±12.98,68.68±23.64,85.29±12.10,94.35±6.04
    `TS+LR`_,87.41±12.58,80.09±15.01,75.01±13.71,87.60±13.20,83.09±13.46,67.28±19.19,87.22±13.83,69.31±23.06,83.62±13.88,94.16±6.33
    `TS+SVM`_,86.48±13.58,79.41±15.26,74.62±14.19,88.08±13.58,83.57±14.08,68.18±19.92,87.64±13.48,68.45±24.25,83.72±14.28,93.37±6.30


Motor Imagery - Right Hand vs Feet
==================================

.. csv-table:: Motor Imagery - Right Hand vs Feet
   :header: Pipelines,:class:`AlexMI`,:class:`BNCI2014_001`,:class:`BNCI2014_002`,:class:`BNCI2015_001`,:class:`BNCI2015_004`,:class:`PhysionetMI`,`HighGamma`_,:class:`Weibo2014`,:class:`Zhou2016`
   :class: sortable

    `ACM+TS+SVM`_,86.56±12.26,97.32±3.35,88.60±10.71,93.01±8.09,62.60±14.62,93.33±8.46,98.67±3.06,93.25±4.12,97.18±3.00
    `CSP+LDA`_,77.19±17.58,91.52±10.39,80.98±14.79,88.52±10.75,54.02±11.33,86.41±13.96,97.02±5.17,88.59±6.36,95.20±3.17
    `CSP+SVM`_,78.59±20.14,91.04±10.35,81.21±15.30,89.19±10.08,52.08±11.05,88.04±12.57,97.50±4.90,88.64±5.90,94.95±3.53
    `DLCSPauto+shLDA`_,77.03±18.93,91.54±10.37,80.45±15.52,88.87±10.42,53.02±10.75,86.81±13.34,96.95±5.22,88.48±6.53,94.43±3.41
    `DeepConvNet`_,61.88±19.05,88.27±12.19,87.56±11.25,88.12±13.19,57.08±12.29,71.49±15.88,95.90±7.14,79.29±12.63,95.92±3.66
    `EEGITNet`_,47.50±9.46,75.98±13.09,70.90±17.50,71.95±16.76,51.41±6.40,54.69±11.97,96.04±8.62,62.54±12.32,80.40±17.12
    `EEGNeX`_,52.34±14.81,64.36±13.49,69.95±20.12,72.34±19.83,53.02±9.69,51.77±12.06,89.49±16.91,60.18±11.70,64.80±16.64
    `EEGNet_8_2`_,64.22±16.01,88.55±14.92,83.93±16.31,90.43±11.75,54.20±8.20,73.78±15.59,96.50±8.07,78.15±14.46,94.58±3.21
    `EEGTCNet`_,61.09±22.06,75.21±18.53,73.92±19.02,77.21±18.55,51.22±5.84,57.03±13.25,97.15±7.70,62.37±12.42,85.46±16.42
    `FilterBank+SVM`_,80.78±18.86,93.55±6.29,80.39±16.83,91.57±7.66,52.51±9.82,83.97±12.43,97.40±4.18,88.27±7.91,94.63±3.94
    `FgMDM`_,79.84±17.80,93.52±8.18,84.77±11.26,90.18±9.77,58.31±12.63,89.67±10.65,98.48±3.45,88.56±4.63,96.04±2.67
    `MDM`_,74.22±21.19,89.13±10.38,77.48±14.11,86.20±12.99,48.45±9.62,81.78±11.64,84.67±13.13,65.18±9.75,92.21±4.31
    `ShallowConvNet`_,64.22±18.33,93.00±8.05,87.60±12.05,91.41±10.88,57.23±12.36,74.75±14.98,98.06±4.35,88.70±5.60,97.06±1.86
    `TS+EL`_,81.41±21.36,94.45±6.74,85.98±11.38,91.19±8.49,58.70±13.37,94.09±7.17,98.56±3.01,92.32±3.98,96.59±2.82
    `TS+LR`_,83.75±17.47,94.45±7.06,85.86±11.01,91.09±8.71,61.01±14.22,93.15±7.40,98.60±3.08,91.53±4.53,96.76±2.58
    `TS+SVM`_,82.66±18.16,94.01±7.60,86.19±11.50,90.81±8.95,62.55±15.30,94.27±7.19,98.72±2.92,91.84±4.25,96.11±2.99

P300/ERP (All classes)
======================

.. csv-table:: P300/ERP (All classes)
   :header: Pipelines,:class:`BNCI2014_008`,:class:`BNCI2014_009`,:class:`BNCI2015_003`,:class:`BI2012`,:class:`BI2013a`,:class:`BI2014a`,:class:`BI2014b`,:class:`BI2015a`,:class:`BI2015b`,:class:`Cattan2019_VR`,:class:`EPFLP300`,:class:`Huebner2017`,:class:`Huebner2018`,:class:`Lee2019_ERP`,:class:`Sosulski2019`
   :class: sortable

    `ERPCov+MDM`_,74.30±9.77,81.16±10.13,76.79±10.95,78.77±10.32,80.59±9.36,71.62±11.17,78.57±12.36,80.02±10.07,75.04±15.85,80.76±10.07,71.97±10.88,94.47±8.26,95.15±3.72,74.43±13.26,68.17±13.59
    `ERPCov(svd_n=4)+MDM`_,75.42±9.91,84.52±8.83,76.93±11.26,79.02±10.53,82.07±8.46,72.11±11.64,76.48±12.83,77.92±10.33,77.09±15.81,80.67±9.47,71.44±10.20,96.21±6.50,96.61±1.89,82.47±12.56,70.63±13.79
    `XDAWN+LDA`_,82.24±5.26,64.03±3.91,78.62±7.19,64.41±4.14,76.74±7.16,66.60±7.54,83.73±10.62,76.02±10.46,77.22±13.73,67.16±6.11,62.98±5.38,97.74±2.84,97.54±1.58,96.45±3.93,67.49±7.44
    `XDAWNCov+MDM`_,77.62±9.81,92.04±5.97,83.08±7.55,88.22±5.90,90.97±5.52,80.88±11.01,91.58±10.02,92.57±5.03,83.48±12.05,88.53±7.34,83.20±9.05,98.07±2.09,97.78±1.04,97.70±2.68,86.07±7.15
    `XDAWNCov+TS+SVM`_,85.61±4.43,93.43±5.11,82.95±8.57,90.99±4.79,92.71±4.92,85.77±9.75,91.88±9.94,93.05±4.98,84.56±12.09,90.68±6.29,84.29±8.53,98.69±1.78,98.47±0.97,98.41±2.03,87.28±6.92

.. raw:: html

   <h1 align="center">SSVEP (All classes)</h1>


.. csv-table:: SSVEP (All classes) part 1
   :header: Methods,:class:`Kalunga2016`,:class:`Lee2019_SSVEP`,:class:`MAMEM1`,:class:`MAMEM2`,:class:`MAMEM3`
   :class: sortable

    `SSVEP_CCA`_,25.40±2.51,23.86±3.72,19.17±5.01,23.60±4.10,13.80±7.47
    `SSVEP_MsetCCA`_,22.67±4.23,25.10±3.81,20.50±2.37,22.08±1.76,27.60±3.01
    `SSVEP_MDM`_,70.89±13.44,75.38±18.38,27.31±11.64,23.12±6.29,34.40±9.96
    `SSVEP_TS+LR`_,70.86±11.64,89.44±13.84,53.71±24.25,39.36±12.06,42.10±14.33
    `SSVEP_TS+SVM`_,68.95±13.73,88.58±14.47,50.58±23.34,34.80±11.76,40.20±14.41
    `SSVEP_TRCA`_,24.84±7.24,64.01±15.27,24.24±6.65,24.24±2.93,23.70±3.49

.. csv-table:: SSVEP (All classes) part 2
   :header: Methods,:class:`Nakanishi2015`,:class:`Wang2016`
   :class: sortable

    `SSVEP_CCA`_,8.15±0.74,2.48±1.01
    `SSVEP_MsetCCA`_,7.10±1.50,4.00±1.10
    `SSVEP_MDM`_,78.77±19.06,54.77±21.95
    `SSVEP_TS+LR`_,87.22±15.96,67.52±20.04
    `SSVEP_TS+SVM`_,86.30±15.88,59.58±20.57
    `SSVEP_TRCA`_,83.21±10.80,2.79±1.03

.. toctree::
   :glob:
   :hidden:
   :caption: MOABB Results
   :titlesonly:

.. raw:: html

   <!-- Must import jquery before the datatables css and js files. -->
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.min.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.0.3/js/dataTables.min.js"></script>


   <table id="voted_issues_table" class="hover row-border order-column" style="width:100%">
      <thead>
         <tr>
            <th>👍</th>
            <th>Issue</th>
            <th>Author</th>
            <th>Title</th>
         </tr>
      </thead>
   </table>

   <!-- JS to enable the datatable features: sortable, paging, search etc
           https://datatables.net/reference/option/
           https://datatables.net/  -->

   <script type="text/javascript">
        $(document).ready(function() {
           $('#voted_issues_table').DataTable( {
              <!-- "ajax": 'voted-issues.json', -->
              "ajax": 'https://raw.githubusercontent.com/scitools/voted_issues/main/voted-issues.json',
              "lengthMenu": [10, 25, 50, 100],
              "pageLength": 10,
              "order": [[ 0, "desc" ]],
              "bJQueryUI": true,
           } );
        } );
   </script>
   <p></p>

.. _Grosse2009: http://moabb.neurotechx.com/docs/generated/moabb.datasets.GrosseWentrup2009.html
.. _HighGamma: http://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html#
.. _SSVEP_CCA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CCA-SSVEP.yml
.. _SSVEP_MsetCCA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/MsetCCA-SSVEP.yml
.. _SSVEP_MDM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/MDM-SSVEP.yml
.. _SSVEP_TS+LR: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR-SSVEP.yml
.. _SSVEP_TS+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml
.. _SSVEP_TRCA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TRCA-SSVEP.yml
.. _XDAWN+LDA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/xDAWN_LDA.yml
.. _XDAWNCov+MDM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/XdawnCov_MDM.yml
.. _XDAWNCov+TS+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/XdawnCov_TS_SVM.yml
.. _ERPCov+MDM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/ERPCov_MDM.yml
.. _ERPCov(svd_n=4)+MDM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/ERPCov_MDM.yml
.. _ACM+TS+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/AUG_TANG_SVM_grid.yml
.. _CSP+LDA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP.yml
.. _CSP+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/CSP_SVM_grid.yml
.. _DLCSPauto+shLDA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/regCSP%2BshLDA.yml
.. _DeepConvNet: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/Keras_DeepConvNet.yml
.. _EEGITNet: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/Keras_EEGITNet.yml
.. _EEGNeX: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/Keras_EEGNeX.yml
.. _EEGNet_8_2: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/Keras_EEGNet_8_2.yml
.. _EEGTCNet: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/Keras_EEGITNet.yml
.. _FilterBank+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/FBCSP.py
.. _FgMDM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/FgMDM.yml
.. _LogVariance+LDA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/LogVar_grid.yml
.. _LogVariance+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/LogVar_grid.yml#L7
.. _MDM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/MDM.yml
.. _ShallowConvNet: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/Keras_ShallowConvNet.yml
.. _TRCSP+LDA: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/WTRCSP.py
.. _TS+EL: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/EN_grid.yml
.. _TS+LR: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSLR.yml
.. _TS+SVM: https://github.com/NeuroTechX/moabb/blob/develop/pipelines/TSSVM_grid.yml



.. raw:: html

   <script type="text/javascript" src="https://cdn.datatables.net/v/bm/dt-1.13.4/datatables.min.js"></script>
   <script type="text/javascript">
    $(document).ready(function() {
    $('.sortable').DataTable({
      "paging": false,
      "searching": false,
      "info": false

    });
    });
   </script>
