"""Brodu2012: OpenViBE motor imagery dataset #1 (Brodu, Lotte & Lecuyer, 2012)."""

import bz2

import mne
import numpy as np
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)


# Base URL for the current (v2.0) merged signal+labels CSV files hosted on the
# OpenViBE download server. The original nicolas.brodu.net link is dead (404);
# this Inria mirror is the working source. Each record is
# ``<NN>-signal.csv.bz2`` with NN in 01..14.
OPENVIBE_BASE_URL = "https://openvibe.inria.fr/private/datasets/dataset-1/"

# The 14 records of "Dataset #1 - Motor Imagery of Hands" all come from the
# single OpenViBE / INRIA subject of Brodu et al. (2012): "560 trials of motor
# imagery (280 trials per class) were recorded over a 2 week period"
# (14 records x 40 trials). They are exposed as 14 runs of one session.
N_RECORDS = 14

# OpenViBE / GDF stimulation codes for the Graz protocol (decimal).
CODE_LEFT = 769  # OVTK_GDF_Left  (0x301)
CODE_RIGHT = 770  # OVTK_GDF_Right (0x302)

# EEG channels in CSV column order. Records 01-04 store the nasion reference as
# ``Ref_Nose``, records 05-14 as its standard 10-10 name ``Nz``.
_CHANNELS = "C3 C4 Nz FC3 FC4 C5 C1 C2 C6 CP3 CP4".split()


class Brodu2012(BaseDataset):
    """Motor imagery dataset of Brodu et al. (2012), OpenViBE dataset #1 [1]_.

    **Dataset description**

    Recorded for Brodu, Lotte and Lecuyer (2012) [1]_ on band-power features for
    motor-imagery BCIs ("OpenViBE / INRIA data"): 14 records of left- versus
    right-hand imagined movement following the Graz protocol (40 trials each,
    20 per hand), acquired with a Mindmedia NeXus32B amplifier at 512 Hz in
    common-average-reference mode over 11 channels. The paper states that the
    data "comprises EEG signals from a single subject for which 560 trials of
    motor imagery (280 trials per class) were recorded over a 2 week period"
    (the download page: "on three different days of the same month"), so the
    14 records are exposed as the 14 runs of one session of a single subject;
    the record-to-day assignment is not documented. The paper describes a
    "nose reference electrode", while the download page states the channels
    "are recorded in common average mode and Nz can be used as a reference".

    Notes
    -----
    The original ``nicolas.brodu.net`` URL is dead (HTTP 404); this loader uses
    the Inria OpenViBE mirror and the current (v2.0) bzip2-compressed CSVs, in
    which the stimulations are merged into the signal file. The ``Event Id``
    column carries the Graz codes ``OVTK_GDF_Left`` (769) and ``OVTK_GDF_Right``
    (770), verified to yield 40 cues (20 + 20) for record 01, so the separate
    ``dataset-1_dep`` label files are not used. The nasion reference column
    (``Ref_Nose`` in records 01-04) is exposed as ``Nz``.

    References
    ----------
    .. [1] Brodu, N., Lotte, F., & Lecuyer, A. (2012). Exploring two novel
       features for EEG-based brain-computer interfaces: Multifractal
       cumulants and predictive complexity. Neurocomputing, 79, 87-94.
       DOI: https://doi.org/10.1016/j.neucom.2011.10.010
       Data: https://openvibe.inria.fr/datasets-downloads/ (Dataset #1,
       recorded 2008/03, uploaded 2010/03).
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            channel_types={"eeg": 11},
            montage="10-10",
            hardware="Mindmedia NeXus32B amplifier",
            reference="common average (nose electrode Nz recorded)",
            ground=None,
            sensors=list(_CHANNELS),
        ),
        participants=ParticipantMetadata(n_subjects=1, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 280, "right_hand": 280},
            events={"left_hand": CODE_LEFT, "right_hand": CODE_RIGHT},
            study_design="Graz University motor imagery protocol: 14 records "
            "of 40 trials (20 left-hand, 20 right-hand imagined movements) from "
            "one subject over a two-week period.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neucom.2011.10.010",
            description="OpenViBE motor imagery dataset (left vs right hand, "
            "Graz protocol, 11 channels, 512 Hz; one subject, 14 records).",
            investigators=["Nicolas Brodu", "Fabien Lotte", "Anatole Lecuyer"],
            institution="INRIA Rennes-Bretagne Atlantique",
            country="FR",
            license="free for research use",
            repository="OpenViBE (Inria)",
            data_url=OPENVIBE_BASE_URL,
            publication_year=2012,
        ),
        sessions_per_subject=1,
        runs_per_session=N_RECORDS,
    )

    def __init__(self):
        super().__init__(
            subjects=[1],
            sessions_per_subject=1,
            events={"left_hand": CODE_LEFT, "right_hand": CODE_RIGHT},
            code="Brodu2012",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1016/j.neucom.2011.10.010",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the 14 record files (``01-signal.csv.bz2`` .. ``14-...``)."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        return [
            dl.data_dl(
                f"{OPENVIBE_BASE_URL}{record:02d}-signal.csv.bz2",
                self.code,
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
            for record in range(1, N_RECORDS + 1)
        ]

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {run: Raw}}`` with one run per record file."""
        return {
            "0": {
                str(run): self._read_record(path)
                for run, path in enumerate(self.data_path(subject))
            }
        }

    @staticmethod
    def _read_record(path):
        with bz2.open(path, "rt") as fobj:
            df = pd.read_csv(fobj, low_memory=False)
        df = df.rename(columns={"Ref_Nose": "Nz"})

        # Signal channels (11), converted from microvolts to volts.
        info = mne.create_info(_CHANNELS, sfreq=512.0, ch_types="eeg")
        signal = df[_CHANNELS].to_numpy(dtype=np.float64).T * 1e-6
        raw = mne.io.RawArray(signal, info, verbose=False)

        # Event cells carry one or more ':'-separated stimulation codes; keep
        # only the left/right imagery cues.
        mapping = {CODE_LEFT: "left_hand", CODE_RIGHT: "right_hand"}
        events = []
        for idx, cell in df["Event Id"].dropna().items():
            for tok in str(cell).split(":"):
                try:
                    code = int(float(tok))
                except ValueError:
                    continue
                if code in mapping:
                    events.append([idx, 0, code])
        if events:
            raw.set_annotations(
                mne.annotations_from_events(
                    np.array(events, dtype=int), 512.0, mapping, verbose=False
                )
            )
        raw.set_montage("colin27_1005", on_missing="ignore", verbose=False)
        return raw
