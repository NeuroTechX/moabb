"""Song2026 multi-class motor-attempt dataset (chronic stroke)."""

import warnings
from pathlib import Path

import mne
import numpy as np

from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


# The dataset is distributed only through Baidu Netdisk (no direct HTTP mirror).
# The repository https://github.com/sjx66606/MVCMGNet holds the analysis code and
# points to the data at:
#   https://pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg   (extraction code: ma93)
SONG2026_BAIDU_URL = "https://pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg"
SONG2026_BAIDU_CODE = "ma93"

# 32-channel Neuracle cap in 10-10 layout. The exact per-channel order distributed
# on Baidu Netdisk could not be verified (access-gated); this default mirrors the
# common Neuracle 32 montage, where the two mastoids used as the reference
# (EEGLAB channels 17 and 18 in the study's pop_reref) fall at positions 17-18.
# Real channel names are read from the EEGLAB struct when present and only fall
# back to this list otherwise.
SONG2026_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "M1",
    "M2",
    "Fz",
    "Cz",
    "Pz",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "TP9",
    "TP10",
    "Oz",
]

# The tens digit of each EEGLAB marker ("21"/"23", "31"/"33", ...) identifies the
# motor-attempt task; the two units digits (1 and 3) are the two cue markers of the
# same task and collapse to one class. The numeric-to-task assignment below is the
# best inference from the paper text and could not be verified against the raw data.
SONG2026_EVENTS = {"fist_clench": 2, "pinch_grip": 3, "wrist_lift": 4, "elbow_flexion": 5}
_TENS_TO_LABEL = {v: k for k, v in SONG2026_EVENTS.items()}


def _label_from_marker(marker):
    """Map an EEGLAB marker string ("21"/"23", "31"/"33", ...) to a class label.

    The tens digit selects the task via ``_TENS_TO_LABEL``; the units digit (1
    or 3) marks one of two cue repetitions of the same task and is ignored.
    Returns ``None`` if ``marker`` does not parse as a 2+ digit code.
    """
    digits = "".join(ch for ch in str(marker) if ch.isdigit())
    if len(digits) < 2:
        return None
    return _TENS_TO_LABEL.get(int(digits[0]))


class Song2026(BaseDataset):
    """Motor-attempt dataset from chronic stroke patients [1]_.

    **Dataset description**

    EEG (and synchronous 4-channel surface EMG) recorded from 45 chronic stroke
    patients while they attempted four unilateral upper-limb movements of the
    affected side: fist clenching (Fc), pinch grip (Pg), wrist lift up (Wlu) and
    elbow flexion (Ef). The task is framed both as a quad-task (4-class) problem
    and, pairwise, as a set of dual-task (2-class) problems. Signals were acquired
    with a 32-channel Neuracle system referenced to the bilateral mastoids at
    1000 Hz. Each paradigm consists of 10 trials of 10 s (5 s motor attempt
    followed by 5 s rest), preceded by a 3-min baseline.

    .. warning::

        **status = needs-data.** The recordings are only distributed through Baidu
        Netdisk (``pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg``, code ``ma93``), which
        cannot be fetched programmatically, so the
        exact on-disk file naming, per-channel order and the numeric-to-task marker
        mapping could not be verified against the raw files. The ``.mat`` path
        decodes class labels from ``EEG.event.type`` using the tens-digit
        convention below (falling back to ``unknown``, with a warning, if that
        struct is absent or does not have one marker per epoch); this mapping
        itself is unverified. This loader implements the layout documented by
        the authors' processing script
        (``DataProcess/ProdData_EEG.m`` in https://github.com/sjx66606/MVCMGNet):
        one EEGLAB ``EEG`` struct per subject saved as a v7.3 ``.mat`` file with the
        continuous/epoched data, ``EEG.srate == 1000`` and the string markers
        ``21/31/41/51`` and ``23/33/43/53``. Download the archive manually and place
        one ``sub-XX.mat`` (or ``.set``) file per subject under the dataset folder;
        :meth:`data_path` explains where.

    References
    ----------

    .. [1] Song, J., Wang, N., Li, Z., Zhang, X., Lv, Z., Shan, X., Yang, Y.,
       Liu, J., & Chai, X. (2026). Decoding multi-class motor attempt from the
       affected unilateral limbs in chronic stroke patients. Journal of
       NeuroEngineering and Rehabilitation, 23(1), 109.
       DOI: https://doi.org/10.1186/s12984-026-01920-z

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32, "emg": 4},
            montage="10-10",
            hardware="Neuracle 32-channel EEG amplifier (with synchronous 4-channel EMG)",
            cap_manufacturer="Neuracle",
            reference="bilateral mastoids (M1, M2)",
            ground=None,
            sensors=SONG2026_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True,
                emg_channels=4,
                other_physiological=[
                    "EMG brachioradialis",
                    "EMG wrist flexors",
                    "EMG wrist extensors",
                    "EMG biceps brachii",
                ],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=45,
            health_status="chronic stroke patients",
            clinical_population="chronic stroke patients with affected unilateral upper limb",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["fist_clench", "pinch_grip", "wrist_lift", "elbow_flexion"],
            trials_per_class={
                "fist_clench": 10,
                "pinch_grip": 10,
                "wrist_lift": 10,
                "elbow_flexion": 10,
            },
            trial_duration=10.0,
            study_design="Motor attempt (attempted movement) of the affected unilateral upper limb: fist clenching, pinch grip, wrist lift up, elbow flexion. Each paradigm: 10 trials of 10 s (5 s attempt + 5 s rest), preceded by 3-min baseline.",
            feedback_type="none",
            stimulus_type="cue",
            synchronicity="cue-based",
            mode="offline",
            events=SONG2026_EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.1186/s12984-026-01920-z",
            description="EEG + EMG motor-attempt dataset from 45 chronic stroke patients performing four cued unilateral upper-limb tasks with the affected side.",
            investigators=[
                "Jiuxiang Song",
                "Nan Wang",
                "Zhaolin Li",
                "Xuemin Zhang",
                "Zeping Lv",
                "Xinying Shan",
                "Yi Yang",
                "Jizhong Liu",
                "Xiaoke Chai",
            ],
            senior_author="Jizhong Liu",
            institution="Beijing Tiantan Hospital, Capital Medical University",
            country="CN",
            data_url=SONG2026_BAIDU_URL,
            repository="Baidu Netdisk (code ma93); code at https://github.com/sjx66606/MVCMGNet",
            license="CC-BY-NC-ND-4.0",
            publication_year=2026,
            keywords=[
                "motor attempt",
                "attempted movement",
                "BCI",
                "brain-computer interface",
                "chronic stroke",
                "EEG",
                "EMG",
                "upper limb",
                "rehabilitation",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        file_format="MAT (EEGLAB struct)",
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Motor Attempt"]),
        abstract="Multi-class motor-attempt decoding from the affected unilateral limbs of 45 chronic stroke patients, using synchronous EEG and EMG. The authors report 78.52% accuracy for dual-task (2-class) and 52.79% for quad-task (4-class) scenarios with the proposed MVCMGNet.",
    )

    def __init__(self, subjects=None, sessions=None, **kwargs):
        super().__init__(
            subjects=list(range(1, 45 + 1)),
            sessions_per_subject=1,
            events=SONG2026_EVENTS,
            code="Song2026",
            interval=(0, 5),
            paradigm="imagery",
            doi="10.1186/s12984-026-01920-z",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path to a single subject's data file.

        The recordings are distributed only through Baidu Netdisk
        (``pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg``, extraction code ``ma93``),
        which cannot be downloaded automatically.
        Download the archive manually and place one file per subject named
        ``sub-XX.mat`` (EEGLAB ``EEG`` struct, ``-v7.3``) or ``sub-XX.set`` under::

            <mne_data>/MNE-song2026-data/

        where ``<mne_data>`` is the MOABB/MNE data directory (``path`` argument,
        or the ``MNE_DATA`` config, defaulting to ``~/mne_data``).

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of the dataset. If None, the MNE_DATA config (or
            ``~/mne_data``) is used.
        force_update : bool
            Unused; kept for API compatibility (no remote copy to refresh).
        update_path : bool | None
            Unused; kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list
            A one-element list with the path to the subject's data file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        if path is None:
            path = mne.get_config("MNE_DATA", str(Path.home() / "mne_data"))
        base = Path(path) / "MNE-song2026-data"

        sub = f"sub-{subject:02d}"
        for ext in (".mat", ".set"):
            candidate = base / f"{sub}{ext}"
            if candidate.exists():
                return [str(candidate)]

        raise RuntimeError(
            f"Song2026 data for {sub} not found under {base}. The dataset is only "
            f"available on Baidu Netdisk ({SONG2026_BAIDU_URL}, code "
            f"{SONG2026_BAIDU_CODE}); download it manually and place one file per "
            f"subject as {sub}.mat or {sub}.set there. See data_path docstring."
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{session: {run: mne.io.Raw}}`` with a single session/run.
        """
        file_path = self.data_path(subject)[0]

        if file_path.endswith(".set"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        else:
            raw = self._read_eeglab_mat(file_path)

        return {"0": {"0": raw}}

    def _read_eeglab_mat(self, file_path):
        """Read an EEGLAB ``EEG`` struct saved as a v7.3 (HDF5) ``.mat`` file.

        The study's ``ProdData_EEG.m`` saves each subject as ``save(..., 'EEG',
        '-v7.3')`` after 1-40 Hz filtering, 48-52 Hz notch and bilateral-mastoid
        re-referencing, epoched from -2 to 5 s around the task markers. This reader
        reconstructs a continuous :class:`mne.io.RawArray` by concatenating the
        epochs and annotating each epoch onset (at the marker, 2 s into the
        block) with the class label decoded from ``EEG.event.type``.
        """
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Reading the Song2026 v7.3 .mat files requires h5py (`pip install h5py`)."
            ) from exc

        with h5py.File(file_path, "r") as f:
            eeg = f["EEG"]
            sfreq = float(np.array(eeg["srate"]).ravel()[0])
            data = np.array(eeg["data"])  # HDF5 stores as (epochs, times, channels)

            # h5py returns the MATLAB array transposed; normalise to
            # (n_channels, n_times, n_epochs).
            if data.ndim == 3:
                data = np.transpose(data, (2, 1, 0))
            elif data.ndim == 2:
                data = data.T[:, :, np.newaxis]

            n_channels = data.shape[0]
            ch_names = self._resolve_channel_names(f, eeg, n_channels)
            event_markers = self._resolve_event_markers(f, eeg)

        # EEGLAB data are in microvolts; MNE expects volts.
        data = data * 1e-6
        n_channels, n_times, n_epochs = data.shape
        continuous = data.transpose(0, 2, 1).reshape(n_channels, n_epochs * n_times)

        info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(continuous, info, verbose=False)

        # Each epoch spans -2..+5 s around its task marker (2 s pre-marker
        # baseline), so the block start (previously used directly as the
        # onset) is 2 s *before* the marker. Offset every onset by the
        # baseline so it lands on the marker, matching interval=(0, 5) in
        # __init__ (instead of windowing -2..+3 s relative to the marker).
        baseline_s = 2.0
        onsets = np.arange(n_epochs) * (n_times / sfreq) + baseline_s
        durations = np.full(n_epochs, n_times / sfreq - baseline_s)

        if event_markers is not None and len(event_markers) == n_epochs:
            descriptions = [_label_from_marker(m) for m in event_markers]
            n_unresolved = sum(d is None for d in descriptions)
            descriptions = [d or "unknown" for d in descriptions]
            if n_unresolved:
                warnings.warn(
                    f"Song2026: could not map {n_unresolved}/{n_epochs} "
                    f"EEG.event.type markers in {Path(file_path).name} to a "
                    "class label; those epochs are annotated 'unknown' and "
                    "will be dropped by the imagery paradigm.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"Song2026: EEG.event.type markers were not found or do not "
                f"match the {n_epochs} epochs in {Path(file_path).name}; all "
                "epochs annotated 'unknown' (needs-data).",
                stacklevel=2,
            )
            descriptions = ["unknown"] * n_epochs

        raw.set_annotations(
            mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        )
        return raw

    @staticmethod
    def _resolve_channel_names(f, eeg, n_channels):
        """Read channel labels from ``EEG.chanlocs.labels`` if available."""
        try:
            labels_ref = eeg["chanlocs"]["labels"]
            names = []
            for ref in np.array(labels_ref).ravel():
                chars = np.array(f[ref]).ravel()
                names.append("".join(chr(int(c)) for c in chars))
            if len(names) == n_channels:
                return names
        except Exception:
            pass
        if n_channels == len(SONG2026_CHANNELS):
            return list(SONG2026_CHANNELS)
        return [f"EEG{i + 1:03d}" for i in range(n_channels)]

    @staticmethod
    def _resolve_event_markers(f, eeg):
        """Read the ``EEG.event.type`` marker strings, in event order.

        Mirrors :meth:`_resolve_channel_names`: each ``event.type`` entry is an
        HDF5 reference to a char-code array (``ProdData_EEG.m`` stores the
        markers "21"/"23"/"31"/... as strings). Returns ``None`` if the
        ``event`` struct is absent or not in the expected layout.
        """
        try:
            type_ref = eeg["event"]["type"]
            markers = []
            for ref in np.array(type_ref).ravel():
                chars = np.array(f[ref]).ravel()
                if chars.size and np.issubdtype(chars.dtype, np.number):
                    markers.append("".join(chr(int(c)) for c in chars))
                else:
                    markers.append("".join(np.asarray(chars).astype(str)))
            return markers
        except Exception:
            return None
