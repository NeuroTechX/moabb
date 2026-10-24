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
    CrossValidationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
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
    The OSF project holds each recording twice under the same file name: the
    BIDS copy (``/sub-XX/``, 73 channels at 100 Hz, EEG merged with the IMU)
    and the original recording (``/sourcedata/sub-XX/``, 50 channels at
    500 Hz: 32 scalp-EEG, 4 EOG and 14 ear-EEG; the header types HEOGL/HEOGR
    as EOG and VEOGU/VEOGL as EEG). MOABB always downloads and loads the
    original 500 Hz recording; the IMU channels are not loaded.

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
            channel_types={"eeg": 48, "eog": 2},
            montage="standard_1005",
            hardware="BrainAmp (Brain Product GmbH)",
            line_freq=60.0,
            sensor_type="Ag/AgCl",
            electrode_material="Ag/AgCl",
            reference="FCz",
            ground="Fpz",
            impedance_threshold_kohm=50,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True, eog_channels=4, eog_type=["vertical", "horizontal"]
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=23,
            health_status="healthy",
            gender={"male": 13, "female": 10},
            age_mean=24.5,
            age_min=19,
            age_max=32,
            age_std=2.9,
            # Per-subject demographics from BIDS participants.tsv
            # Subject 24 excluded: no SSVEP data on OSF
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
            ],
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            events={"5.45": 11, "8.57": 12, "12.0": 13},
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
        signal_processing=SignalProcessingMetadata(
            classifiers=["rLDA", "CCA"],
            feature_extraction=["power_over_time_intervals", "CCA"],
            frequency_bands={
                "delta": [0.5, 3.5],
                "theta": [3.5, 7.5],
                "alpha": [7.5, 12.5],
                "beta": [12.5, 30.0],
            },
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="holdout", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["mobile_BCI"], environment="treadmill"
        ),
        tags=Tags(pathology=["healthy"], modality=["visual"], type=["perception"]),
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
    _ERP_MARKER_MAP = {"Stimulus/S  2": "Target", "Stimulus/S  1": "NonTarget"}

    def __init__(
        self, paradigm, subjects=None, sessions=None, *, return_all_modalities=False
    ):
        if paradigm.lower() == "ssvep":
            code_suffix = "SSVEP"
            events = {"5.45": 11, "8.57": 12, "12.0": 13}
            interval = [0, 5]
            paradigm_type = "ssvep"
            n_sessions = 4  # ses-02 through ses-05 (variable per subject)
            subject_list = list(range(1, 24))  # 23 subjects; sub-24 has no SSVEP data
        elif paradigm.lower() in ("erp", "p300"):
            code_suffix = "ERP"
            events = {"Target": 2, "NonTarget": 1}
            interval = [0, 1.0]
            paradigm_type = "p300"
            n_sessions = 5  # ses-01 through ses-05
            subject_list = list(range(1, 25))  # 24 subjects
        else:
            raise ValueError(f"Unknown paradigm '{paradigm}'. Use 'ssvep' or 'erp'.")

        self._task_name = code_suffix

        super().__init__(
            subjects=subject_list,
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

    # BrainVision files the loader reads. OSF r7s9b stores each of them twice
    # under the same file name: the BIDS copy (``/sub-XX/ses-YY/eeg/``, 73
    # channels at 100 Hz: EEG and EOG resampled and merged with the IMU
    # channels) and the original recording (``/sourcedata/sub-XX/ses-YY/eeg/``,
    # 50 channels at 500 Hz). The loader only reads the original recording.
    _EEG_EXTENSIONS = (".vhdr", ".eeg", ".vmrk")
    _ROOT_FILES = ("dataset_description.json", "participants.tsv", "participants.json")

    def _download_from_osf(self, subject, bids_root, force_update, verbose):
        """Download subject files from OSF per-file API.

        Files are selected by their path on OSF, not by file name: the
        BrainVision files come from ``/sourcedata/sub-XX/`` only (see
        :meth:`_select_osf_files`). They are stored at the same local path
        as before (``sub-XX/ses-YY/eeg/<file name>``).
        """
        bids_root.mkdir(parents=True, exist_ok=True)

        metainfo = self._get_metainfo()
        try:
            to_download = self._select_osf_files(metainfo, subject, self._task_name)
        except FileNotFoundError:
            # The listing may be incomplete (OSF errors are logged and the
            # folder skipped): list again on the next call.
            type(self)._metainfo_cache = None
            raise

        # Download root BIDS files
        for fname in self._ROOT_FILES:
            local_path = bids_root / fname
            if not local_path.exists() or force_update:
                matches = metainfo[metainfo["filename"] == fname]
                if "path" in metainfo.columns:
                    at_root = matches[matches["path"] == f"/{fname}"]
                    if not at_root.empty:
                        matches = at_root
                if not matches.empty:
                    url = matches.iloc[0]["url"]
                    dl.download_if_missing(
                        str(local_path),
                        url,
                        warn_missing=False,
                        force_update=force_update,
                    )

        # Download subject files
        for fname, url in to_download.items():
            local_path = self._filename_to_bids_path(fname, bids_root)
            if not local_path.exists() or force_update:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                dl.download_if_missing(
                    str(local_path), url, warn_missing=False, force_update=force_update
                )

    @classmethod
    def _select_osf_files(cls, metainfo, subject, task):
        """Pick the OSF copy of each of one subject's files.

        Parameters
        ----------
        metainfo : pandas.DataFrame
            OSF listing from :func:`moabb.datasets.download.create_metainfo_osf`,
            with ``filename``, ``url`` and ``path`` columns.
        subject : int
            Subject number.
        task : str
            ``"ERP"`` or ``"SSVEP"``; its recordings must all be available
            in ``/sourcedata/``.

        Returns
        -------
        files : dict
            ``{file name: download url}``, one entry per local file name.
            BrainVision files (``.vhdr``, ``.eeg``, ``.vmrk``) are taken from
            ``/sourcedata/sub-XX/`` only, whatever the listing order; other
            files from ``/sourcedata/sub-XX/`` when present there, else from
            the BIDS copy ``/sub-XX/``.

        Raises
        ------
        FileNotFoundError
            If a BrainVision file of a ``task`` recording listed on OSF
            (in either copy) is missing from ``/sourcedata/sub-XX/``.
        """
        if "path" not in metainfo.columns:
            raise ValueError(
                "The OSF listing has no 'path' column, so the /sourcedata/ "
                "copy of the EEG files cannot be told apart from the BIDS "
                "copy (same file names, 100 Hz instead of 500 Hz)."
            )
        sub = f"sub-{subject:02d}"
        paths = metainfo["path"].astype(str)
        names = metainfo["filename"].astype(str)
        is_eeg = names.str.endswith(cls._EEG_EXTENSIONS)
        in_source = paths.str.startswith(f"/sourcedata/{sub}/")
        in_bids = paths.str.startswith(f"/{sub}/")

        files = {}
        # Other files: BIDS copy first, overridden by the /sourcedata/ copy.
        for mask in (in_bids & ~is_eeg, in_source & ~is_eeg):
            files.update(zip(names[mask], metainfo["url"][mask]))
        # BrainVision files: /sourcedata/ only.
        source_eeg = dict(
            zip(names[in_source & is_eeg], metainfo["url"][in_source & is_eeg])
        )
        files.update(source_eeg)

        suffix = f"_task-{task}_eeg.vhdr"
        recordings = {
            name[: -len(".vhdr")]
            for name in names[in_source | in_bids]
            if name.startswith(f"{sub}_") and name.endswith(suffix)
        }
        missing = sorted(
            stem + ext
            for stem in recordings
            for ext in cls._EEG_EXTENSIONS
            if stem + ext not in source_eeg
        )
        if missing:
            raise FileNotFoundError(
                f"Lee2021Mobile subject {subject}, task {task}: the OSF listing "
                f"of project {OSF_CODE} has no /sourcedata/{sub}/ copy of "
                f"{', '.join(missing)}. MOABB reads only that copy (50 channels "
                "at 500 Hz); the BIDS copy under the same name is 73 channels "
                "at 100 Hz and is not used. The listing may be incomplete if "
                f"OSF returned errors; retry later or check https://osf.io/{OSF_CODE}/."
            )
        return files

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

    See :class:`~moabb.datasets.lee2021_mobile.Lee2021Mobile` for full documentation.
    """

    nemar_id = "nm000125"
    __init__ = partialmethod(Lee2021Mobile.__init__, "SSVEP")


class Lee2021Mobile_ERP(Lee2021Mobile):
    """ERP paradigm of the Mobile BCI dataset.

    See :class:`~moabb.datasets.lee2021_mobile.Lee2021Mobile` for full documentation.
    """

    nemar_id = "nm000201"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            channel_types={"eeg": 48, "eog": 2},
            montage="standard_1005",
            hardware="BrainAmp (Brain Product GmbH)",
            line_freq=60.0,
            sensor_type="Ag/AgCl",
            electrode_material="Ag/AgCl",
            reference="FCz",
            ground="Fpz",
            impedance_threshold_kohm=50,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True, eog_channels=4, eog_type=["vertical", "horizontal"]
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
            events={"Target": 2, "NonTarget": 1},
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
            environment="mobile", online_feedback=False
        ),
        tags=Tags(pathology=["healthy"], modality=["visual"], type=["perception"]),
        sessions_per_subject=5,
        file_format="BrainVision",
    )

    __init__ = partialmethod(Lee2021Mobile.__init__, "ERP")
