"""OpenViBE motor imagery dataset (Brodu, Lotte & Lecuyer, 2012)."""

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

# OpenViBE / GDF stimulation codes for the Graz protocol (decimal).
CODE_LEFT = 769  # OVTK_GDF_Left  (0x301)
CODE_RIGHT = 770  # OVTK_GDF_Right (0x302)

# Channel names as they appear in the CSV header (the reference/nose channel is
# stored as ``Ref_Nose``; it is the nasion ``Nz`` position).
_CSV_CHANNELS = [
    "C3",
    "C4",
    "Ref_Nose",
    "FC3",
    "FC4",
    "C5",
    "C1",
    "C2",
    "C6",
    "CP3",
    "CP4",
]
# Rename to standard montage names for MNE (Ref_Nose -> Nz).
_CH_RENAME = {"Ref_Nose": "Nz"}
_CHANNELS = [_CH_RENAME.get(c, c) for c in _CSV_CHANNELS]


class OpenViBE(BaseDataset):
    """Motor imagery dataset from the OpenViBE project [1]_.

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class     Trials len    Sampling rate    #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        OpenViBE        14       11           2                 20            varies            512            1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    This EEG dataset was recorded for the study of Brodu, Lotte and Lecuyer
    (2012) [1]_ on band-power feature extraction for motor imagery based
    brain-computer interfaces. It contains 14 motor imagery records of
    left-hand versus right-hand imagined movements following the Graz
    University protocol. Each record contains 40 trials (20 left hand, 20
    right hand). The records were acquired over three different days of the
    same month.

    The signals were recorded with a Mindmedia NeXus32B amplifier at 512 Hz
    in a common-average-reference mode over 11 channels: C3, C4, Nz (nasion
    reference, stored as ``Ref_Nose`` in the CSV), FC3, FC4, C5, C1, C2, C6,
    CP3 and CP4.

    The data are distributed as bzip2-compressed CSV files produced by the
    OpenViBE ``CSVFileWriter`` box. In the current (v2.0) format, the
    stimulation labels are merged into the signal file (``Event Id`` /
    ``Event Date`` / ``Event Duration`` columns). The left- and right-hand
    imagery cues use the GDF stimulation codes ``OVTK_GDF_Left`` (769) and
    ``OVTK_GDF_Right`` (770).

    Notes
    -----
    The original ``nicolas.brodu.net`` download URL is no longer available
    (HTTP 404). This loader uses the working Inria OpenViBE mirror at
    ``openvibe.inria.fr/private/datasets/dataset-1/``.

    The trial labels are embedded in the signal CSV itself: the ``Event Id``
    column carries the Graz stimulation codes, and the left/right cues use
    ``OVTK_GDF_Left`` (769) and ``OVTK_GDF_Right`` (770). This was verified by
    decompressing ``dataset-1/01-signal.csv.bz2`` and parsing the column
    directly, which yields exactly 40 cue events per record (20 code 769 and
    20 code 770), matching the 20-left / 20-right Graz design. The signal file
    is therefore the authoritative source; the separate ``dataset-1_dep``
    variant (``NN-signal_d.csv.bz2`` + ``NN-labels_d.csv.bz2``) is not needed
    and is not used here.

    References
    ----------
    .. [1] Brodu, N., Lotte, F., & Lecuyer, A. (2012). Exploring two novel
       features for EEG-based brain-computer interfaces: Multifractal
       cumulants and predictive complexity. Neurocomputing, 79, 87-94.
       DOI: https://doi.org/10.1016/j.neucom.2011.10.010
    """

    nemar_id = "EXEMPT"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=11,
            channel_types={"eeg": 11},
            montage="10-10",
            hardware="Mindmedia NeXus32B amplifier",
            reference="common average reference",
            ground=None,
            sensors=list(_CHANNELS),
        ),
        participants=ParticipantMetadata(n_subjects=14, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 20, "right_hand": 20},
            events={"left_hand": CODE_LEFT, "right_hand": CODE_RIGHT},
            study_design="Graz University motor imagery protocol: 40 trials "
            "per record (20 left-hand, 20 right-hand imagined movements).",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neucom.2011.10.010",
            description="OpenViBE motor imagery dataset (left vs right hand, "
            "Graz protocol, 11 channels, 512 Hz).",
            investigators=["Nicolas Brodu", "Fabien Lotte", "Anatole Lecuyer"],
            country="FR",
            license="free for research use",
            repository="OpenViBE (Inria)",
            data_url=OPENVIBE_BASE_URL,
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14 + 1)),
            sessions_per_subject=1,
            events={"left_hand": CODE_LEFT, "right_hand": CODE_RIGHT},
            code="OpenViBE",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1016/j.neucom.2011.10.010",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path to a single subject's signal file.

        Parameters
        ----------
        subject : int
            The subject (record) number, 1..14.
        path : None | str
            Location for the data storage folder. If None, the MNE default is
            used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Whether to update the MNE config path.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of str
            A one-element list with the path to the subject's signal file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        url = f"{OPENVIBE_BASE_URL}{subject:02d}-signal.csv.bz2"
        local_path = dl.data_dl(
            url, self.code, path=path, force_update=force_update, verbose=verbose
        )
        return [local_path]

    def _get_single_subject_data(self, subject):
        """Return the data for a single subject.

        Parameters
        ----------
        subject : int
            The subject (record) number, 1..14.

        Returns
        -------
        dict
            ``{"0": {"0": mne.io.Raw}}`` mapping session -> run -> raw.
        """
        file_path = self.data_path(subject)[0]

        with bz2.open(file_path, "rt") as fobj:
            df = pd.read_csv(fobj, low_memory=False)

        # Records 01--04 name the nasion reference ``Ref_Nose``, while
        # records 05--14 use its standard 10-10 name ``Nz``.  Normalize this
        # one source-header alias before selecting the fixed channel order.
        if "Ref_Nose" not in df.columns and "Nz" in df.columns:
            df = df.rename(columns={"Nz": "Ref_Nose"})

        # Signal channels (11), converted from microvolts to volts.
        signal = df[_CSV_CHANNELS].to_numpy(dtype=np.float64).T * 1e-6

        # Build the raw object with a dedicated stim channel.
        info = mne.create_info(
            ch_names=list(_CHANNELS), sfreq=512.0, ch_types=["eeg"] * len(_CHANNELS)
        )
        raw = mne.io.RawArray(signal, info, verbose=False)

        # Parse the merged event column: rows carry one or more ':'-separated
        # stimulation codes. Keep only the left/right imagery cues.
        events = []
        event_col = df["Event Id"]
        for idx, cell in event_col.items():
            if pd.isna(cell):
                continue
            for tok in str(cell).split(":"):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    code = int(float(tok))
                except ValueError:
                    continue
                if code in (CODE_LEFT, CODE_RIGHT):
                    events.append([int(idx), 0, code])

        mapping = {CODE_LEFT: "left_hand", CODE_RIGHT: "right_hand"}
        if events:
            events_arr = np.array(events, dtype=int)
            annotations = mne.annotations_from_events(
                events_arr, sfreq=raw.info["sfreq"], event_desc=mapping, verbose=False
            )
            raw.set_annotations(annotations)

        try:
            raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
        except Exception:
            pass

        return {"0": {"0": raw}}
