"""Mobile BCI dataset with SSVEP and ERP paradigms during motion.

Lee et al. (2021), Scientific Data.
DOI: 10.1038/s41597-021-01094-4
"""

from functools import partialmethod
from pathlib import Path

import mne

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


OSF_CODE = "r7s9b"


class Lee2021Mobile(BaseDataset):
    """Mobile BCI dataset with scalp- and ear-EEG during standing/walking/running.

    Dataset from [1]_.

    This dataset contains 73-channel recordings (32 scalp-EEG + 14 ear-EEG +
    27 IMU) from 24 healthy subjects (14 men, 10 women, mean age 24.5) during
    different motion conditions: standing (0 m/s), slow walking (0.8 m/s),
    fast walking (1.6 m/s), and slight running (2.0 m/s).

    Two BCI paradigms were used:
    - **SSVEP**: 3 target frequencies (5.45, 8.57, 12.0 Hz), 60 trials/session
    - **ERP**: Target ('OOO') vs non-target ('XXX'), 300 trials/session

    Data is in BrainVision format (.eeg/.vhdr/.vmrk) organized as BIDS on OSF.

    Sessions correspond to motion conditions:
    - Session '02': Standing (0 m/s)
    - Session '03': Slow walking (0.8 m/s)
    - Session '04': Fast walking (1.6 m/s)
    - Session '05': Slight running (2.0 m/s) [subjects 1-18 only]

    For ERP, session '01' is also available (training, standing only).

    Warnings
    --------
    Not all subjects have session '05' (2.0 m/s running). Subjects 19-24
    only have sessions 01-04.

    The BIDS events.tsv files have onset in samples (not seconds as BIDS
    requires), so data is loaded via MNE's BrainVision reader directly
    instead of mne_bids.

    References
    ----------
    .. [1] Y.-E. Lee, G.-H. Shin, M. Lee, and S.-W. Lee, "Mobile BCI
       dataset of scalp- and ear-EEGs with ERP and SSVEP paradigms while
       standing, walking, and running," Scientific Data, vol. 8, p. 315,
       2021. DOI: 10.1038/s41597-021-01094-4
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1005",
            hardware="BrainAmp (Brain Product GmbH)",
            line_freq=60.0,
            sensor_type="Ag/AgCl",
            electrode_material="Ag/AgCl",
            reference="FCz",
            ground="Fpz",
            impedance_threshold_kohm=50,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=4,
                eog_type=["vertical", "horizontal"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=24,
            health_status="healthy",
            gender={"male": 14, "female": 10},
            age_mean=24.5,
            age_min=19,
            age_max=32,
            age_std=2.9,
            # Per-subject demographics from BIDS participants.tsv
            ages=[
                28,
                24,
                23,
                24,
                24,
                22,
                28,
                19,
                27,
                22,
                29,
                21,
                26,
                27,
                24,
                24,
                22,
                23,
                32,
                26,
                24,
                21,
                25,
                23,
            ],
            sexes=[
                "female",
                "female",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "male",
                "female",
                "male",
                "female",
            ],
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=3,
            trial_duration=5.0,
            stimulus_type="visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            study_design="BCI during motion (standing/walking/running)",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-021-01094-4",
            investigators=[
                "Young-Eun Lee",
                "Gi-Hwan Shin",
                "Minji Lee",
                "Seong-Whan Lee",
            ],
            senior_author="Seong-Whan Lee",
            institution="Korea University",
            country="KR",
            repository="OSF",
            data_url="https://osf.io/r7s9b/",
            license="CC BY 4.0",
            publication_year=2021,
            ethics_approval=[
                "Institutional Review Board of Korea University, KUIRB-2019-0194-01"
            ],
            funding=[
                "IITP No. 2017-0-00451",
                "IITP No. 2015-0-00185",
                "IITP No. 2019-0-00079",
            ],
            keywords=["SSVEP", "ERP", "mobile BCI", "ear-EEG", "locomotion"],
        ),
        bci_application=BCIApplicationMetadata(
            environment="mobile",
            online_feedback=False,
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        sessions_per_subject=4,
        file_format="BrainVision",
    )

    # BrainVision annotation -> frequency mapping for SSVEP
    _SSVEP_MARKER_MAP = {
        "Stimulus/S 11": "5.45",
        "Stimulus/S 12": "8.57",
        "Stimulus/S 13": "12.0",
    }

    # BrainVision annotation -> label mapping for ERP (P300)
    _ERP_MARKER_MAP = {
        "Stimulus/S  2": "Target",
        "Stimulus/S  1": "NonTarget",
    }

    def __init__(
        self,
        paradigm,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        if paradigm.lower() == "ssvep":
            code_suffix = "SSVEP"
            events = {"5.45": 11, "8.57": 12, "12.0": 13}
            interval = [0, 5]
            paradigm_type = "ssvep"
            n_sessions = 4  # ses-02 through ses-05 (variable per subject)
        elif paradigm.lower() in ("erp", "p300"):
            code_suffix = "ERP"
            events = {"Target": 2, "NonTarget": 1}
            interval = [0, 1.0]
            paradigm_type = "p300"
            n_sessions = 5  # ses-01 through ses-05
        else:
            raise ValueError(f"Unknown paradigm '{paradigm}'. Use 'ssvep' or 'erp'.")

        self._task_name = code_suffix

        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=n_sessions,
            events=events,
            code=f"Lee2021Mobile-{code_suffix}",
            interval=interval,
            paradigm=paradigm_type,
            doi="10.1038/s41597-021-01094-4",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across motion sessions.

        Reads BrainVision .vhdr files directly (not via mne_bids) because
        the BIDS events.tsv has onset in samples rather than seconds.
        Annotations from the .vmrk marker file are used instead, with
        stimulus markers renamed to frequency strings for the SSVEP paradigm.
        """
        vhdr_files = self.data_path(subject)

        sessions = {}
        for vhdr_path in vhdr_files:
            vhdr_path = Path(vhdr_path)

            # Extract session from filename: sub-01_ses-02_task-SSVEP_eeg.vhdr
            parts = vhdr_path.stem.split("_")
            session = None
            for p in parts:
                if p.startswith("ses-"):
                    session = p.replace("ses-", "")
                    break
            if session is None:
                session = "0"
            else:
                # Canonicalize BIDS ses-02 -> "2" so selected_sessions=[2] matches.
                try:
                    session = str(int(session))
                except ValueError:
                    session = str(session)

            raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

            # Rename stimulus markers to match event names
            if self._task_name == "SSVEP":
                raw.annotations.rename(self._SSVEP_MARKER_MAP)
            elif self._task_name == "ERP":
                raw.annotations.rename(self._ERP_MARKER_MAP)

            sessions[session] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code.split("-")[0]
        bids_root = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        sub_dir = bids_root / f"sub-{subject:02d}"

        # Find .vhdr files for this subject and task
        task = self._task_name
        vhdr_files = sorted(sub_dir.rglob(f"*task-{task}_eeg.vhdr"))

        if vhdr_files and not force_update:
            return [str(f) for f in vhdr_files]

        # Download from OSF if not found locally
        self._download_from_osf(subject, bids_root, force_update, verbose)

        vhdr_files = sorted(sub_dir.rglob(f"*task-{task}_eeg.vhdr"))
        if not vhdr_files:
            raise FileNotFoundError(
                f"No .vhdr files found for subject {subject}, task {task}. "
                f"Download the OSF archive from https://osf.io/{OSF_CODE}/ "
                f"and extract to {bids_root}"
            )
        return [str(f) for f in vhdr_files]

    def _download_from_osf(self, subject, bids_root, force_update, verbose):
        """Download subject files from OSF per-file API."""
        bids_root.mkdir(parents=True, exist_ok=True)

        metainfo = self._get_metainfo()

        # Download root BIDS files
        for fname in [
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
        ]:
            local_path = bids_root / fname
            if not local_path.exists() or force_update:
                matches = metainfo[metainfo["filename"] == fname]
                if not matches.empty:
                    url = matches.iloc[0]["url"]
                    dl.download_if_missing(str(local_path), url, warn_missing=False)

        # Download subject files
        sub_str = f"sub-{subject:02d}"
        sub_files = metainfo[metainfo["filename"].str.startswith(sub_str)]

        for _, row in sub_files.iterrows():
            fname = row["filename"]
            url = row["url"]
            local_path = self._filename_to_bids_path(fname, bids_root)
            if not local_path.exists() or force_update:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                dl.download_if_missing(str(local_path), url, warn_missing=False)

    @staticmethod
    def _filename_to_bids_path(filename, bids_root):
        """Convert a flat BIDS filename to the full directory structure."""
        parts = filename.split("_")
        sub = parts[0]  # sub-XX
        ses = parts[1] if len(parts) > 1 and parts[1].startswith("ses-") else None

        if ses:
            return bids_root / sub / ses / "eeg" / filename
        return bids_root / sub / "eeg" / filename

    _metainfo_cache = None

    @classmethod
    def _get_metainfo(cls):
        """Get or cache the OSF file listing."""
        if cls._metainfo_cache is None:
            cls._metainfo_cache = dl.create_metainfo_osf(OSF_CODE)
        return cls._metainfo_cache


class Lee2021Mobile_SSVEP(Lee2021Mobile):
    """SSVEP paradigm of the Mobile BCI dataset.

    See :class:`Lee2021Mobile` for full documentation.
    """

    __init__ = partialmethod(Lee2021Mobile.__init__, "SSVEP")


class Lee2021Mobile_ERP(Lee2021Mobile):
    """ERP paradigm of the Mobile BCI dataset.

    See :class:`Lee2021Mobile` for full documentation.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1005",
            hardware="BrainAmp (Brain Product GmbH)",
            line_freq=60.0,
            sensor_type="Ag/AgCl",
            electrode_material="Ag/AgCl",
            reference="FCz",
            ground="Fpz",
            impedance_threshold_kohm=50,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=4,
                eog_type=["vertical", "horizontal"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=24,
            health_status="healthy",
            gender={"male": 14, "female": 10},
            age_mean=24.5,
            age_min=19,
            age_max=32,
            age_std=2.9,
            ages=[
                28,
                24,
                23,
                24,
                24,
                22,
                28,
                19,
                27,
                22,
                29,
                21,
                26,
                27,
                24,
                24,
                22,
                23,
                32,
                26,
                24,
                21,
                25,
                23,
            ],
            sexes=[
                "female",
                "female",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "male",
                "female",
                "male",
                "male",
                "male",
                "female",
                "male",
                "female",
            ],
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=2,
            trial_duration=1.0,
            stimulus_type="visual oddball",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            study_design="BCI during motion (standing/walking/running)",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-021-01094-4",
            investigators=[
                "Young-Eun Lee",
                "Gi-Hwan Shin",
                "Minji Lee",
                "Seong-Whan Lee",
            ],
            senior_author="Seong-Whan Lee",
            institution="Korea University",
            country="KR",
            repository="OSF",
            data_url="https://osf.io/r7s9b/",
            license="CC BY 4.0",
            publication_year=2021,
            ethics_approval=[
                "Institutional Review Board of Korea University, KUIRB-2019-0194-01"
            ],
            funding=[
                "IITP No. 2017-0-00451",
                "IITP No. 2015-0-00185",
                "IITP No. 2019-0-00079",
            ],
            keywords=["SSVEP", "ERP", "mobile BCI", "ear-EEG", "locomotion"],
        ),
        bci_application=BCIApplicationMetadata(
            environment="mobile",
            online_feedback=False,
        ),
        tags=Tags(
            pathology=["healthy"],
            modality=["visual"],
            type=["perception"],
        ),
        sessions_per_subject=5,
        file_format="BrainVision",
    )

    __init__ = partialmethod(Lee2021Mobile.__init__, "ERP")
