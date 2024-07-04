:html_theme.sidebar_secondary.remove:
.. _paper_results:
.. raw:: html

   <!-- Must import jquery before the datatables css and js files. -->
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.0.3/js/dataTables.js"></script>
   <div style="font-size: 1em;">

.. currentmodule:: moabb.datasets

The largest EEG-based Benchmark for Open Science
=================================================

We report the results of the benchmark study performed in:
`The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark <https://universite-paris-saclay.hal.science/hal-04537061v1/file/MOABB-arXiv.pdf>`_

This study conducts an extensive Brain-computer interfaces (BCI) reproducibility analysis on open electroencephalography datasets,
aiming to assess existing solutions and establish open and reproducible benchmarks for effective comparison within the field. Please note that the results are obtained using `Within-Session evaluation <http://moabb.neurotechx.com/docs/generated/moabb.evaluations.WithinSessionEvaluation.html>`_.
The results are reported regarding mean accuracy and standard deviation across all folds for all sessions and subjects.

If you use the same evaluation procedure, you should expect similar results if you use the same pipelines and datasets, with some minor variations due to the randomness of the cross-validation procedure.

**You can copy and use the table in your work**, but please `**cite the paper** <http://moabb.neurotechx.com/docs/cite.html>`_ if you do so.

Motor Imagery
=============================
Motor Imagery is a BCI paradigm where the subject imagines performing a movement.
Each imagery task is associated with a different class, and each task has its difficulty level related to how the brain generates the signal.

Here, we present three different scenarios for Motor Imagery classification:

#. **Left vs Right Hand**: We use only the classes Left Hand and Right Hand.
#. **Right Hand vs Feet**: We use only Right Hand and Feet classes.
#. **All classes**: We use all the classes in the dataset, when there are more than classes that are not Left Hand and Right Hand.

All the results here are for **within-session evaluation**, a 5-fold cross-validation, over the subject's session.


Motor Imagery - Left vs Right Hand
===================================
.. raw:: html
   <hr>

**Left vs Right Hand**: We use only the classes Left Hand and Right Hand.


.. raw:: html
  </div>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.0.3/js/dataTables.js"></script>

    <table id="mileftvsright" class="hover row-border order-column" style="width:100%">
        <thead>
        <tr class="row-odd"><th class="head"><p>Pipelines</p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001" title="moabb.datasets.BNCI2014_001"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_001</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_004.html#moabb.datasets.BNCI2014_004" title="moabb.datasets.BNCI2014_004"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_004</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Cho2017.html#moabb.datasets.Cho2017" title="moabb.datasets.Cho2017"><code class="xref py py-class docutils literal notranslate"><span class="pre">Cho2017</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.GrosseWentrup2009.html#moabb.datasets.GrosseWentrup2009" title="moabb.datasets.GrosseWentrup2009"><code class="xref py py-class docutils literal notranslate"><span class="pre">GrosseWentrup2009</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Lee2019_MI.html#moabb.datasets.Lee2019_MI" title="moabb.datasets.Lee2019_MI"><code class="xref py py-class docutils literal notranslate"><span class="pre">Lee2019_MI</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.PhysionetMI.html#moabb.datasets.PhysionetMI" title="moabb.datasets.PhysionetMI"><code class="xref py py-class docutils literal notranslate"><span class="pre">PhysionetMI</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Schirrmeister2017.html#moabb.datasets.Schirrmeister2017" title="moabb.datasets.Schirrmeister2017"><code class="xref py py-class docutils literal notranslate"><span class="pre">Schirrmeister2017</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Shin2017A.html#moabb.datasets.Shin2017A" title="moabb.datasets.Shin2017A"><code class="xref py py-class docutils literal notranslate"><span class="pre">Shin2017A</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Weibo2014.html#moabb.datasets.Weibo2014" title="moabb.datasets.Weibo2014"><code class="xref py py-class docutils literal notranslate"><span class="pre">Weibo2014</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016" title="moabb.datasets.Zhou2016"><code class="xref py py-class docutils literal notranslate"><span class="pre">Zhou2016</span></code></a></p></th>
        </tr>
        </thead>
    </table>
   <script type="text/javascript">
        $(document).ready(function() {
           $('#mileftvsright').DataTable( {
              "ajax": 'https://raw.githubusercontent.com/NeuroTechX/moabb/develop/results/within_session_mi_left_vs_right_hand.json',
              "order": [[ 1, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "info": false,
              "searching": false,
           } );
        } );
   </script>
   <hr>




Motor Imagery - Right Hand vs Feet
==================================

**Right Hand vs Feet**: We use only Right Hand and Feet classes.

.. raw:: html
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.0.3/js/dataTables.js"></script>

    <table id="mirightvsfeet" class="hover row-border order-column" style="width:100%">
        <thead>
        <tr class="row-odd"><th class="head"><p>Pipeline</p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.AlexMI.html#moabb.datasets.AlexMI" title="moabb.datasets.AlexMI"><code class="xref py py-class docutils literal notranslate"><span class="pre">AlexMI</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001" title="moabb.datasets.BNCI2014_001"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_001</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_002.html#moabb.datasets.BNCI2014_002" title="moabb.datasets.BNCI2014_002"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_002</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2015_001.html#moabb.datasets.BNCI2015_001" title="moabb.datasets.BNCI2015_001"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2015_001</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2015_004.html#moabb.datasets.BNCI2015_004" title="moabb.datasets.BNCI2015_004"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2015_004</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.PhysionetMI.html#moabb.datasets.PhysionetMI" title="moabb.datasets.PhysionetMI"><code class="xref py py-class docutils literal notranslate"><span class="pre">PhysionetMI</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Schirrmeister2017.html#moabb.datasets.Schirrmeister2017" title="moabb.datasets.Schirrmeister2017"><code class="xref py py-class docutils literal notranslate"><span class="pre">Schirrmeister2017</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Weibo2014.html#moabb.datasets.Weibo2014" title="moabb.datasets.Weibo2014"><code class="xref py py-class docutils literal notranslate"><span class="pre">Weibo2014</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016" title="moabb.datasets.Zhou2016"><code class="xref py py-class docutils literal notranslate"><span class="pre">Zhou2016</span></code></a></p></th>
        </tr>
        </thead>
    </table>


   <script type="text/javascript">
        $(document).ready(function() {
           $('#mirightvsfeet').DataTable( {
              "ajax": 'https://raw.githubusercontent.com/NeuroTechX/moabb/develop/results/within_session_mi_right_hand_vs_feet.json',
              "order": [[ 1, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "info": false,
              "searching": false,
           } );
        } );
   </script>



Motor Imagery - All classes
===================================
.. raw:: html
   <p></p>
   <hr>



**All classes**: We use all the classes in the dataset, when there are more than classes that are not Left Hand and Right Hand.

.. raw:: html
   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/2.0.3/css/dataTables.dataTables.css">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/2.0.3/js/dataTables.js"></script>

    <table id="mi-all" class="hover row-border order-column" style="width:100%">
        <thead>
        <tr class="row-odd"><th class="head"><p>Pipelines</p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.AlexMI.html#moabb.datasets.AlexMI" title="moabb.datasets.AlexMI"><code class="xref py py-class docutils literal notranslate"><span class="pre">AlexMI</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001" title="moabb.datasets.BNCI2014_001"><code class="xref py py-class docutils literal notranslate"><span class="pre">BNCI2014_001</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.PhysionetMI.html#moabb.datasets.PhysionetMI" title="moabb.datasets.PhysionetMI"><code class="xref py py-class docutils literal notranslate"><span class="pre">PhysionetMI</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Schirrmeister2017.html#moabb.datasets.Schirrmeister2017" title="moabb.datasets.Schirrmeister2017"><code class="xref py py-class docutils literal notranslate"><span class="pre">Schirrmeister2017</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Weibo2014.html#moabb.datasets.Weibo2014" title="moabb.datasets.Weibo2014"><code class="xref py py-class docutils literal notranslate"><span class="pre">Weibo2014</span></code></a></p></th>
        <th class="head"><p><a class="reference internal" href="generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016" title="moabb.datasets.Zhou2016"><code class="xref py py-class docutils literal notranslate"><span class="pre">Zhou2016</span></code></a></p></th>
        </tr>
        </thead>
    </table>
    <script type="text/javascript">
        $(document).ready(function() {
           $('#mi-all').DataTable( {
              "ajax": 'https://raw.githubusercontent.com/NeuroTechX/moabb/develop/results/within_session_mi_all_classes.json',
              "order": [[ 1, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "searching": false,
              "info": false,
           } );
        } );
    </script>



SSVEP (All classes)
======================

Here, we have the results of the within-session evaluation, a 5-fold cross-validation, over the subject's session.
We use all the classes available in the dataset.

.. raw:: html

   <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
   <link href="https://cdn.datatables.net/v/dt/dt-2.0.4/b-3.0.2/b-html5-3.0.2/datatables.min.css" rel="stylesheet">
   <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/v/dt/dt-2.0.4/b-3.0.2/b-html5-3.0.2/datatables.min.js"></script>

    <table id="ssvep" class="hover row-border order-column" style="width:100%">
        <thead>
        <tr class="row-odd"><th class="head"><p>Pipeline</p></th>
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
              "ajax": 'https://raw.githubusercontent.com/NeuroTechX/moabb/develop/results/within_session_ssvep_all_classes.json',
              "order": [[ 1, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "searching": false,
              "info": false,
			  "buttons": ["copyHtml5","csvHtml5"],
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
              "ajax": 'https://raw.githubusercontent.com/NeuroTechX/moabb/develop/results/within_session_erp_p300_all_classes.json',
              "order": [[ 1, "desc" ]],
              "bJQueryUI": true,
              "scrollX": true,
              "paging": false,
              "searching": false,
              "info": false,
			  "buttons": ["copyHtml5","csvHtml5"],
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
