.. _data_summary:

Data Summary
======================

MOABB gather many datasets, here is list summarizing important information. Most of the
datasets are listed here but this list not complete yet, check API for complete
documentation.

Do not hesitate to help us complete this list. It is also possible to add new datasets,
there is a tutorial explaining how to do so, and we welcome warmly any new contributions!

See also `Datasets-Support <https://github.com/NeuroTechX/moabb/wiki/Datasets-Support>`__ for supplementary
detail on datasets (class name, size, licence, etc.)

Motor Imagery
======================

.. csv-table::
   :header: Dataset, #Subj, #Chan, #Classes, #Trials, len, Sampling rate, #Sessions, #Trials*#Sessions
   :class: sortable

   AlexMI,8,16,3,20,3s,512Hz,1,20
   BNCI2014001,10,22,4,144,4s,250Hz,2,288
   BNCI2014002,15,15,2,80,5s,512Hz,1,80
   BNCI2014004,10,3,2,360,4.5s,250Hz,5,1800
   BNCI2015001,13,13,2,200,5s,512Hz,2,400
   BNCI2015004,10,30,5,80,7s,256Hz,2,160
   Cho2017,53,64,2,100,3s,512Hz,1,100
   Lee2019_MI,55,62,2,100,4s,1000Hz,2,200
   MunichMI,10,128,2,150,7s,500Hz,1,150
   Schirrmeister2017,14,128,4,120,4s,500Hz,1,120
   Ofner2017,15,61,7,60,3s,512Hz,1,60
   PhysionetMI,109,64,4,23,3s,160Hz,1,23
   Shin2017A,29,30,2,30,10s,200Hz,3,90
   Shin2017B,29,30,2,30,10s,200Hz,3,90
   Weibo2014,10,60,7,80,4s,200Hz,1,80
   Zhou2016,4,14,3,160,5s,250Hz,3,480

P300/ERP
======================

.. csv-table::
   :header: Dataset, #Subj, #Chan, #Trials / class, Trials length, Sampling rate, #Sessions
   :class: sortable

   BNCI2014008, 8, 8, 3500 NT / 700 T, 1s, 256Hz, 1
   BNCI2014009, 10, 16, 1440 NT / 288 T, 0.8s, 256Hz, 3
   BNCI2015003, 10, 8, 1500 NT / 300 T, 0.8s, 256Hz, 1
   bi2012, 25, 16, 6140 NT / 128 T, 1s, 512Hz, 2
   bi2013a, 24, 16, 3200 NT / 640 T, 1s, 512Hz, 8 for subjects 1-7, else 1
   bi2014a, 71, 16, , 1s, 512Hz, up to 3
   bi2014b, 38, 32, , 1s, 512Hz, 3
   bi2015a, 50, 32, , 1s, 512Hz, 3
   bi2015b, 44, 32, , 1s, 512Hz, 2
   VirtualReality, 24, 16, 600 NT / 120 T, 1s, 512Hz, 2
   EPFLP300, 8, 32, 2753 NT / 551 T, 1s, 2048Hz, 4
   Lee2019_ERP, 54, 62, 6900 NT / 1380 T, 1s, 1000Hz, 2

SSVEP
======================


.. csv-table::
   :header: Dataset, #Subj, #Chan, #Classes, #Trials / class, Trials length, Sampling rate, #Sessions
   :class: sortable

   Lee2019_SSVEP,24,16,4,25,1s,1000Hz,1
   SSVEPExo,12,8,4,16,2s,256Hz,1
   MAMEM1,10,256,5,12-15,3s,250Hz,1
   MAMEM2,10,256,5,20-30,3s,250Hz,1
   MAMEM3,10,14,4,20-30,3s,128Hz,1
   Nakanishi2015,9,8,12,15,4.15s,256Hz,1
   Wang2016,32,62,40,6,5s,250Hz,1



Submit a new dataset
~~~~~~~~~~~~~~~~~~~~

you can submit a new dataset by mentioning it to this
`issue <https://github.com/NeuroTechX/moabb/issues/1>`__. The datasets
currently on our radar can be seen `here <https://github.com/NeuroTechX/moabb/wiki/Datasets-Support>`__,
but we are open to any suggestion.

If you want to actively contribute to inclusion of one new dataset, you can follow also this tutorial
`tutorial <https://neurotechx.github.io/moabb/auto_tutorials/tutorial_4_adding_a_dataset.html>`__.

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
