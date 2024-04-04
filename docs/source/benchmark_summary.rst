.. _data_summary:

.. automodule:: moabb.benchmark
    :members:
    :undoc-members:
    :show-inheritance:

.. currentmodule:: moabb.benchmark


What are the states of art for BCI?
===================================

 text to introduce the benchmarking results. Create the citation for the benchmarking paper.


Motor Imagery
======================

.. csv-table::
   :header: Dataset, #Subj, #Chan, #Classes, #Trials, Trial length, Freq, #Session, #Runs, Total_trials
   :class: sortable

        :class:`AlexMI`,8,16,3,20,3s,512Hz,1,1,480
        :class:`BNCI2014_001`,9,22,4,144,4s,250Hz,2,6,62208
        :class:`BNCI2014_002`,14,15,2,80,5s,512Hz,1,8,17920
        :class:`BNCI2014_004`,9,3,2,360,4.5s,250Hz,5,1,32400
        :class:`BNCI2015_001`,12,13,2,200,5s,512Hz,3,1,14400
        :class:`BNCI2015_004`,9,30,5,80,7s,256Hz,2,1,7200
        :class:`Cho2017`,52,64,2,100,3s,512Hz,1,1,9800
        :class:`Lee2019_MI`,54,62,2,100,4s,1000Hz,2,1,11000
        :class:`GrosseWentrup2009`,10,128,2,150,7s,500Hz,1,1,3000
        :class:`Schirrmeister2017`,14,128,4,120,4s,500Hz,1,2,13440
        :class:`Ofner2017`,15,61,7,60,3s,512Hz,1,10,63000
        :class:`PhysionetMI`,109,64,4,23,3s,160Hz,1,1,69760
        :class:`Shin2017A`,29,30,2,30,10s,200Hz,3,1,5220
        :class:`Shin2017B`,29,30,2,30,10s,200Hz,3,1,5220
        :class:`Weibo2014`,10,60,7,80,4s,200Hz,1,1,5600
        :class:`Zhou2016`,4,14,3,160,5s,250Hz,3,2,11496

P300/ERP
======================

.. csv-table::
   :header: Dataset, #Subj, #Chan, #Trials / class, Trials length, Sampling rate, #Sessions
   :class: sortable

   :class:`BNCI2014_008`, 8, 8, 3500 NT / 700 T, 1s, 256Hz, 1
   :class:`BNCI2014_009`, 10, 16, 1440 NT / 288 T, 0.8s, 256Hz, 3
   :class:`BNCI2015_003`, 10, 8, 1500 NT / 300 T, 0.8s, 256Hz, 1
   :class:`BI2012`, 25, 16, 640 NT / 128 T, 1s, 128Hz, 2
   :class:`BI2013a`, 24, 16, 3200 NT / 640 T, 1s, 512Hz, 8 for subjects 1-7 else 1
   :class:`BI2014a`, 64, 16, 990 NT / 198 T, 1s, 512Hz, up to 3
   :class:`BI2014b`, 38, 32, 200 NT / 40 T, 1s, 512Hz, 3
   :class:`BI2015a`, 43, 32, 4131 NT / 825 T, 1s, 512Hz, 3
   :class:`BI2015b`, 44, 32, 2160 NT / 480 T, 1s, 512Hz, 1
   :class:`Cattan2019_VR`, 21, 16, 600 NT / 120 T, 1s, 512Hz, 2
   :class:`Huebner2017`, 13, 31, 364 NT / 112 T, 0.9s, 1000Hz, 3
   :class:`Huebner2018`, 12, 31, 364 NT / 112 T, 0.9s, 1000Hz, 3
   :class:`Sosulski2019`, 13, 31, 75 NT / 15 T, 1.2s, 1000Hz, 3
   :class:`EPFLP300`, 8, 32, 2753 NT / 551 T, 1s, 2048Hz, 4
   :class:`Lee2019_ERP`, 54, 62, 6900 NT / 1380 T, 1s, 1000Hz, 2


SSVEP
======================


.. csv-table::
   :header: Dataset, #Subj, #Chan, #Classes, #Trials / class, Trials length, Sampling rate, #Sessions
   :class: sortable

   :class:`Lee2019_SSVEP`,54,62,4,50,4s,1000Hz,2
   :class:`Kalunga2016`,12,8,4,16,2s,256Hz,1
   :class:`MAMEM1`,10,256,5,12-15,3s,250Hz,1
   :class:`MAMEM2`,10,256,5,20-30,3s,250Hz,1
   :class:`MAMEM3`,10,14,4,20-30,3s,128Hz,1
   :class:`Nakanishi2015`,9,8,12,15,4.15s,256Hz,1
   :class:`Wang2016`,34,62,40,6,5s,250Hz,1


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
