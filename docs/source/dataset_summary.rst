.. _data_summary:

.. automodule:: moabb.datasets

.. currentmodule:: moabb.datasets


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
   :file: ../build/summary_imagery.csv
   :header-rows: 1
   :class: sortable


P300/ERP
======================

.. csv-table::
   :file: ../build/summary_p300.csv
   :header-rows: 1
   :class: sortable


SSVEP
======================

.. csv-table::
   :file: ../build/summary_ssvep.csv
   :header-rows: 1
   :class: sortable


c-VEP
======================

Include neuro experiments where the participant is presented with psuedo-random noise-codes,
such as m-sequences, Gold codes, or any arbitrary "pseudo-random" code. Specifically, the
difference with SSVEP is that SSVEP presents periodic stimuli, while c-VEP presents
non-periodic stimuli. For a review of c-VEP BCI, see:

Martínez-Cagigal, V., Thielen, J., Santamaria-Vazquez, E., Pérez-Velasco, S., Desain, P.,&
Hornero, R. (2021). Brain–computer interfaces based on code-modulated visual evoked
potentials (c-VEP): A literature review. Journal of Neural Engineering, 18(6), 061002.
DOI: https://doi.org/10.1088/1741-2552/ac38cf

.. csv-table::
   :file: ../build/summary_cvep.csv
   :header-rows: 1
   :class: sortable

Resting States
======================

Include neuro experiments where the participant is not actively doing something.
For example, recoding the EEG of a subject while s/he is having the eye closed or opened
is a resting state experiment.

.. csv-table::
    :file: ../build/summary_rstate.csv
    :header-rows: 1
    :class: sortable

Compound Datasets
======================

.. automodule:: moabb.datasets.compound_dataset

.. currentmodule:: moabb.datasets.compound_dataset

Compound Datasets are datasets compounded with subjects from other datasets.
It is useful for merging different datasets (including other Compound Datasets),
select a sample of subject inside a dataset (e.g. subject with high/low performance).

.. csv-table::
   :header: Dataset, #Subj, #Original datasets
   :class: sortable

   :class:`BI2014a_Il`,17,BI2014a
   :class:`BI2014b_Il`,11,BI2014b
   :class:`BI2015a_Il`,2,BI2015a
   :class:`BI2015b_Il`,25,BI2015b
   :class:`Cattan2019_VR_Il`,4,Cattan2019_VR
   :class:`BI_Il`,59,:class:`BI2014a_Il` :class:`BI2014b_Il` :class:`BI2015a_Il` :class:`BI2015b_Il` :class:`Cattan2019_VR_Il`


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
