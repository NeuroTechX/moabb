:html_theme.sidebar_secondary.remove:
.. _paper_results:
.. raw:: html

   <!-- Must import jquery before the datatables css and js files. -->
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.0.3/js/dataTables.js"></script>
   <div style="font-size: 20px;">

.. currentmodule:: moabb.datasets

The largest EEG-based Benchmark for Open Science
=================================================

We report the results of the benchmark study performed in:
`The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark <https://universite-paris-saclay.hal.science/hal-04537061v1/file/MOABB-arXiv.pdf>`_

This study conducts an extensive Brain-computer interfaces (BCI) reproducibility analysis on open electroencephalography datasets,
aiming to assess existing solutions and establish open and reproducible benchmarks for effective comparison within the field. Please note that the results are obtained using `Within-Session evaluation <http://moabb.neurotechx.com/docs/generated/moabb.evaluations.WithinSessionEvaluation.html>`_.
The results are reported regarding mean accuracy and standard deviation across all folds for all sessions and subjects.

If you use the same evaluation procedure, you should expect similar results if you use the same pipelines and datasets, with some minor variations due to the randomness of the cross-validation procedure.

You can copy and use the table in your work, but please `**cite the paper** <http://moabb.neurotechx.com/docs/cite.html>`_ if you do so.

Motor Imagery - All classes
=============================
Motor Imagery is a BCI paradigm where the subject imagines performing a movement.
Each imagery task is associated with a different class, and each task has its difficulty level related to how the brain generates the signal.

Here, we present three different scenarios for Motor Imagery classification:

#. **All classes**: We use all the classes in the dataset.
#. **Left vs Right Hand**: We use only the classes Left Hand and Right Hand.
#. **Right Hand vs Feet**: We use only Right Hand and Feet classes.

All the results here are for **within-session evaluation**, a 5-fold cross-validation, over the subject's session.

.. csv-table:: Motor Imagery - All classes
   :header: Pipelines,:class:`AlexMI`,:class:`BNCI2014_001`,:class:`PhysionetMI`,:class:`Schirrmeister2017`,:class:`Weibo2014`,:class:`Zhou2016`
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
   :header: Pipelines,:class:`BNCI2014_001`,:class:`BNCI2014_004`,:class:`Cho2017`,:class:`GrosseWentrup2009`,:class:`Lee2019_MI`,:class:`PhysionetMI`,:class:`Schirrmeister2017`,:class:`Shin2017A`,:class:`Weibo2014`,:class:`Zhou2016`
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
   :header: Pipelines,:class:`AlexMI`,:class:`BNCI2014_001`,:class:`BNCI2014_002`,:class:`BNCI2015_001`,:class:`BNCI2015_004`,:class:`PhysionetMI`,:class:`Schirrmeister2017`,:class:`Weibo2014`,:class:`Zhou2016`
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



SSVEP (All classes)
======================

Here, we have the results of the within-session evaluation, a 5-fold cross-validation, over the subject's session.
We use all the classes available in the dataset.


.. raw:: html
    <table id="ssvep" class="hover row-border order-column" style="width:100%">
        <thead>
        <tr class="row-odd"><th class="head"><p>Pipelines</p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Kalunga2016.html#moabb.datasets.Kalunga2016" title="moabb.datasets.Kalunga2016"><code class="xref py py-class docutils literal notranslate"><span class="pre">Kalunga2016</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Lee2019_SSVEP.html#moabb.datasets.Lee2019_SSVEP" title="moabb.datasets.Lee2019_SSVEP"><code class="xref py py-class docutils literal notranslate"><span class="pre">Lee2019_SSVEP</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.MAMEM1.html#moabb.datasets.MAMEM1" title="moabb.datasets.MAMEM1"><code class="xref py py-class docutils literal notranslate"><span class="pre">MAMEM1</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.MAMEM2.html#moabb.datasets.MAMEM2" title="moabb.datasets.MAMEM2"><code class="xref py py-class docutils literal notranslate"><span class="pre">MAMEM2</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.MAMEM3.html#moabb.datasets.MAMEM3" title="moabb.datasets.MAMEM3"><code class="xref py py-class docutils literal notranslate"><span class="pre">MAMEM3</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Nakanishi2015.html#moabb.datasets.Nakanishi2015" title="moabb.datasets.Nakanishi2015"><code class="xref py py-class docutils literal notranslate"><span class="pre">Nakanishi2015</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Wang2016.html#moabb.datasets.Wang2016" title="moabb.datasets.Wang2016"><code class="xref py py-class docutils literal notranslate"><span class="pre">Wang2016</span></code></a></p></th>
        </tr>
        </thead>
    </table>


   <script type="text/javascript">
        $(document).ready(function() {
           $('#ssvep').DataTable( {
              "ajax": 'https://raw.githubusercontent.com/bruAristimunha/moabb/table_results/results/within_session_ssvep_all_classes.json',
              "order": [[ 0, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "searching": false,
           } );
        } );
   </script>



P300/ERP (All classes)
======================

Here, we have the results of the within-session evaluation, a 5-fold cross-validation, over the subject's session.
We use all the classes available in the dataset.

.. raw:: html

    <table id="p300" class="hover row-border order-column" style="width:100%">
        <thead>
        <tr class="row-odd"><th class="head"><p>Pipelines</p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_008.html#moabb.datasets.BNCI2014_008" title="moabb.datasets.BNCI2014_008"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_008</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_009.html#moabb.datasets.BNCI2014_009" title="moabb.datasets.BNCI2014_009"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_009</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2015_003.html#moabb.datasets.BNCI2015_003" title="moabb.datasets.BNCI2015_003"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2015_003</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BI2012.html#moabb.datasets.BI2012" title="moabb.datasets.BI2012"><code class="xref py py-class docutils literal notranslate"><span class="pre">BI2012</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BI2013a.html#moabb.datasets.BI2013a" title="moabb.datasets.BI2013a"><code class="xref py py-class docutils literal notranslate"><span class="pre">BI2013a</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BI2014a.html#moabb.datasets.BI2014a" title="moabb.datasets.BI2014a"><code class="xref py py-class docutils literal notranslate"><span class="pre">BI2014a</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BI2014b.html#moabb.datasets.BI2014b" title="moabb.datasets.BI2014b"><code class="xref py py-class docutils literal notranslate"><span class="pre">BI2014b</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BI2015a.html#moabb.datasets.BI2015a" title="moabb.datasets.BI2015a"><code class="xref py py-class docutils literal notranslate"><span class="pre">BI2015a</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BI2015b.html#moabb.datasets.BI2015b" title="moabb.datasets.BI2015b"><code class="xref py py-class docutils literal notranslate"><span class="pre">BI2015b</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Cattan2019_VR.html#moabb.datasets.Cattan2019_VR" title="moabb.datasets.Cattan2019_VR"><code class="xref py py-class docutils literal notranslate"><span class="pre">Cattan2019_VR</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.EPFLP300.html#moabb.datasets.EPFLP300" title="moabb.datasets.EPFLP300"><code class="xref py py-class docutils literal notranslate"><span class="pre">EPFLP300</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Huebner2017.html#moabb.datasets.Huebner2017" title="moabb.datasets.Huebner2017"><code class="xref py py-class docutils literal notranslate"><span class="pre">Huebner2017</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Huebner2018.html#moabb.datasets.Huebner2018" title="moabb.datasets.Huebner2018"><code class="xref py py-class docutils literal notranslate"><span class="pre">Huebner2018</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Lee2019_ERP.html#moabb.datasets.Lee2019_ERP" title="moabb.datasets.Lee2019_ERP"><code class="xref py py-class docutils literal notranslate"><span class="pre">Lee2019_ERP</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Sosulski2019.html#moabb.datasets.Sosulski2019" title="moabb.datasets.Sosulski2019"><code class="xref py py-class docutils literal notranslate"><span class="pre">Sosulski2019</span></code></a></p></th>
        </tr>
        </thead>
    </table>


   <script type="text/javascript">
        $(document).ready(function() {
           $('#p300').DataTable( {
              "ajax": 'https://raw.githubusercontent.com/bruAristimunha/moabb/table_results/results/within_session_erp_p300_all_classes.json',
              "order": [[ 0, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "searching": false,
           } );
        } );
   </script>
   <p></p>


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

.. toctree::
   :glob:
   :hidden:
   :caption: MOABB Results
   :titlesonly:
