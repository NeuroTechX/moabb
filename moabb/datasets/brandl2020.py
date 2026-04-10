"""Motor Imagery Under Distraction Dataset.

Brandl and Blankertz (2020), Frontiers in Neuroscience.
DOI: 10.3389/fnins.2020.566147
"""

import logging

import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.io import RawArray
from pymatreader import read_mat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# DepositOnce TU Berlin bitstream base URL
_BITSTREAM_URL = (
    "https://depositonce.tu-berlin.de/bitstreams/handle/11303/10934.2/{filename}"
)

# Bitstream UUIDs for direct download (resolved from DepositOnce DSpace)
_BITSTREAM_UUIDS = {
    "pp1.mat": "15de7e5f-8e0c-4804-aac4-7452c38baee5",
    "pp2.mat": "95b19d45-dfb0-4cff-890f-454d940667ca",
    "pp3.mat": "33a17a5a-b7f6-4029-a8bb-e755b87d9814",
    "pp4.mat": "6c062ca7-9ea1-4c90-b099-4e02e6a4f2aa",
    "pp5.mat": "05ea8c98-ef3f-4480-8698-4c3717778d8e",
    "pp6.mat": "eda3529d-ed90-44a3-a00a-51a9e2e9ebc7",
    "pp7.mat": "c22c97e4-a64d-4507-b4a4-d4ae5fced559",
    "pp8.mat": "dab37412-2631-43b8-9e1c-98d0ace499bb",
    "pp9.mat": "762a8d35-dcb2-451e-ac69-4a88e583ad62",
    "pp10.mat": "0c2beb44-c5c3-45d8-b3c3-1ba4652b26d3",
    "pp11.mat": "dc4323d1-5192-4b21-b6c8-9332fafbebfa",
    "pp12.mat": "e634f7ff-9531-42c5-924d-064eaeefc6bf",
    "pp13.mat": "33c14c21-2434-40af-ac78-6df7e4cad2f9",
    "pp14.mat": "71edb5a4-4166-426d-b4f3-16de7909ac78",
    "pp15.mat": "ae369c8d-87a5-405e-9636-5038df03d5b8",
    "pp16.mat": "358eef6d-bffe-4562-91fb-0173deafa0ed",
    "mnt.mat": "6bc4de0d-4c65-4bd8-96b2-1f921a800ace",
}

# Distraction condition names (runs 2-7 map to conditions 1-6)
_CONDITION_NAMES = {
    1: "clean",
    2: "eyesclosed",
    3: "news",
    4: "numbers",
    5: "flicker",
    6: "stimulation",
}


class Brandl2020(BaseDataset):
    """Motor Imagery under distraction dataset from Brandl and Blankertz 2020.

    Dataset from [1]_.

    This dataset contains 63-channel EEG recordings from 16 healthy subjects
    (6 female, 10 male, age 22-30, mean 26.3) performing left/right hand
    motor imagery under various distraction conditions.

    Each subject completed 1 session with 7 runs:

    - **Run 0 (calibration):** 72 trials, no feedback, no distraction
    - **Runs 1-6 (feedback):** 72 trials each with auditory feedback,
      under 6 distraction conditions (clean, eyes closed, news, numbers,
      flicker, vibro-tactile stimulation)

    Total: 504 trials per subject (252 left, 252 right).

    Auditory cues ("links"/"rechts") indicated left/right hand imagery.
    Trial duration was 4.5 s with 2.5 s inter-trial interval. EEG was
    recorded at 1000 Hz using 63 wet Ag/AgCl electrodes (Fast'n Easy Cap)
    with nose reference and two BrainAmp amplifiers (Brain Products).

    Event codes encode both the distraction condition and the motor imagery
    class: condition * 10 + class, where class 1 = left_hand and
    class 2 = right_hand. For the calibration run, codes are 11 (left)
    and 12 (right). For feedback runs, codes are condition * 10 + class
    (e.g., 21/22 for eyes closed, 31/32 for news, etc.).

    For MOABB, all codes ending in 1 are mapped to ``left_hand`` and
    all codes ending in 2 are mapped to ``right_hand``.

    .. note::

       The data files are MATLAB v7.3 (HDF5) format, approximately
       600-770 MB each (total ~10.7 GB for all 16 subjects). The first
       download may take considerable time.

    References
    ----------
    .. [1] Brandl, S. and Blankertz, B. (2020). Motor Imagery Under
           Distraction -- An Open Access BCI Dataset. Frontiers in
           Neuroscience, 14, 566147.
           https://doi.org/10.3389/fnins.2020.566147

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=63,
            channel_types={"eeg": 63},
            hardware="2x BrainAmp (Brain Products)",
            reference="nose",
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "AFz",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT7",
                "FT8",
                "Fp1",
                "Fp2",
                "Fpz",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
            ],
            line_freq=50.0,
            montage="standard_1005",
            sensor_type="Ag/AgCl wet",
            cap_manufacturer="EasyCap",
            cap_model="Fast'n Easy Cap",
            software="BBCI Toolbox (MATLAB)",
        ),
        participants=ParticipantMetadata(
            n_subjects=16,
            gender={"female": 6, "male": 10},
            age_mean=26.3,
            health_status="healthy",
            species="homo sapiens",
            bci_experience="mostly naive (3/16 had prior BCI experience)",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            events={"left_hand": 1, "right_hand": 2},
            trial_duration=4.5,
            study_design=(
                "Motor imagery under distraction: 1 calibration run (no feedback, "
                "no distraction) + 6 feedback runs with different distraction "
                "conditions (clean, eyes closed, news, number search, flicker, "
                "vibro-tactile stimulation)"
            ),
            stimulus_type="auditory",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            mode="online",
            synchronicity="cue-based",
            feedback_type="auditory",
            has_training_test_split=False,
            instructions=(
                "Subjects received auditory cues ('links' for left, 'rechts' for "
                "right) and performed motor imagery of left or right hand movement"
            ),
            tasks=[
                "calibration",
                "clean",
                "eyesclosed",
                "news",
                "numbers",
                "flicker",
                "stimulation",
            ],
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnins.2020.566147",
            publication_year=2020,
            investigators=["Stephanie Brandl", "Benjamin Blankertz", "Tobias Dahne"],
            senior_author="Benjamin Blankertz",
            institution="Technische Universitaet Berlin",
            institution_department="Department of Neurotechnology",
            country="DE",
            data_url="https://depositonce.tu-berlin.de/handle/11303/10934.2",
            license="CC-BY-NC-ND-4.0",
            repository="DepositOnce TU Berlin",
            ethics_approval=[
                "Approved by the ethics committee of the Charite University "
                "Medicine Berlin"
            ],
            keywords=[
                "brain-computer interface",
                "motor imagery",
                "EEG",
                "distraction",
                "open access",
                "BCI",
            ],
            funding=["BMBF/BIFOLD (01IS18025A, 01IS18037A)"],
            how_to_acknowledge=(
                "Please cite: Brandl, S. and Blankertz, B. (2020). Motor Imagery "
                "Under Distraction -- An Open Access BCI Dataset. Frontiers in "
                "Neuroscience, 14, 566147. https://doi.org/10.3389/fnins.2020.566147"
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=7,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+LDA"],
            feature_extraction=["CSP", "bandpower"],
            frequency_bands={"mu": [8.0, 13.0], "beta": [13.0, 30.0]},
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="holdout", evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"], environment="laboratory", online_feedback=True
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="motor_imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=None,
            imagery_duration_s=4.5,
        ),
        data_structure=DataStructureMetadata(
            n_trials=504,
            n_trials_per_class={"left_hand": 252, "right_hand": 252},
            n_blocks=7,
            trials_context=(
                "7 runs per subject: 1 calibration (72 trials) + 6 feedback "
                "runs (72 trials each, 6 distraction conditions)"
            ),
        ),
        data_processed=False,
        file_format="MAT (HDF5 v7.3)",
        abstract=(
            "We present an open-access dataset of a motor imagery "
            "brain-computer interface (BCI) experiment conducted under "
            "six different distraction conditions. Sixteen healthy participants "
            "performed left vs. right hand motor imagery while being "
            "distracted by flickering video, number search tasks, news "
            "listening, eyes closed, vibro-tactile stimulation, or no "
            "distraction. Each participant completed one calibration run "
            "without feedback and six feedback runs under the different "
            "distraction conditions, resulting in 504 trials per subject."
        ),
        methodology=(
            "Participants completed one session with 7 runs of 72 trials each. "
            "Run 1 was calibration (no feedback, no distraction). Runs 2-7 "
            "included auditory feedback and one of six distraction conditions. "
            "Auditory cues indicated left or right hand imagery. Trial "
            "duration was 4.5 s with 2.5 s ITI. Online classification used "
            "CSP with LDA. EEG recorded at 1000 Hz with 63 channels, nose "
            "reference, using two BrainAmp amplifiers."
        ),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 17)),
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="Brandl2020",
            interval=[0, 4.5],
            paradigm="imagery",
            doi="10.3389/fnins.2020.566147",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject number {subject}, must be in {self.subject_list}"
            )

        paths = []

        # Download subject data file
        subj_fname = f"pp{subject}.mat"
        url = _BITSTREAM_URL.format(filename=subj_fname)
        paths.append(dl.data_dl(url, "Brandl2020", path, force_update, verbose))

        # Download montage file (shared across subjects)
        url_mnt = _BITSTREAM_URL.format(filename="mnt.mat")
        paths.append(dl.data_dl(url_mnt, "Brandl2020", path, force_update, verbose))

        return paths

    def _get_single_subject_data(self, subject):
        """Return data for a single subject.

        Returns
        -------
        dict
            ``{"0": {"0calibration": raw, "1clean": raw, "2eyesclosed": raw,
            "3news": raw, "4numbers": raw, "5flicker": raw,
            "6stimulation": raw}}``
        """
        file_paths = self.data_path(subject)
        subj_path = file_paths[0]

        mat = read_mat(subj_path)

        # cnt_orig is a 1x2 cell array (calibration, feedback).
        # pymatreader resolves it to a list of dicts.
        cnt_orig = mat["cnt_orig"]
        mrk_orig = mat["mrk_orig"]

        # Extract channel names and sfreq from first entry
        cnt0 = cnt_orig[0]
        ch_names = cnt0["clab"]
        if isinstance(ch_names, str):
            ch_names = [ch_names]
        sfreq = float(np.asarray(cnt0["fs"]).flat[0])

        # Standardize channel names for MNE compatibility
        ch_names = [ch.replace("Z", "z").replace("FP", "Fp") for ch in ch_names]

        runs = {}

        # --- Process calibration data (cnt_orig[0]) ---
        calib_raw = self._process_segment(cnt_orig[0], mrk_orig[0], ch_names, sfreq)
        runs["0calibration"] = calib_raw

        # --- Process feedback data (cnt_orig[1]) ---
        # The feedback data is one concatenated recording of runs 2-7.
        # We need to split by distraction condition using marker codes.
        feedback_raw = self._process_segment(
            cnt_orig[1], mrk_orig[1], ch_names, sfreq, split_by_condition=True
        )
        if isinstance(feedback_raw, dict):
            runs.update(feedback_raw)
        else:
            runs["1feedback"] = feedback_raw

        return {"0": runs}

    def _process_segment(self, cnt, mrk, ch_names, sfreq, split_by_condition=False):
        """Process one segment (calibration or feedback) of the data.

        Parameters
        ----------
        cnt : dict
            The cnt_orig entry (pymatreader dict with keys 'x', 'clab', 'fs').
        mrk : dict
            The mrk_orig entry (pymatreader dict with keys 'time', 'event').
        ch_names : list of str
            Channel names.
        sfreq : float
            Sampling frequency.
        split_by_condition : bool
            If True, split feedback data into separate runs by condition.

        Returns
        -------
        raw or dict of raw
            If split_by_condition is False, returns a single Raw object.
            If True, returns a dict of {run_name: Raw} for each condition.
        """
        # Data: ensure shape is [n_channels, n_samples].
        # pymatreader returns MATLAB's [n_samples, n_channels] convention.
        raw_data = np.asarray(cnt["x"])
        n_ch = len(ch_names)
        if raw_data.shape[0] == n_ch:
            data = raw_data  # already [n_channels, n_samples]
        else:
            data = raw_data.T  # transpose to [n_channels, n_samples]

        # Marker times (in ms) and event descriptions
        mrk_time = np.asarray(mrk["time"]).flatten()
        desc = np.asarray(mrk["event"]["desc"]).flatten().astype(int)

        # Convert marker times from ms to samples
        mrk_samples = np.round(mrk_time / 1000.0 * sfreq).astype(int)

        if not split_by_condition:
            # Simple case: create one Raw with all data
            return self._make_raw(data, ch_names, sfreq, mrk_samples, desc)
        else:
            # Split feedback data into runs by distraction condition
            # Condition codes: 1x=clean, 2x=eyes_closed, 3x=news,
            # 4x=numbers, 5x=flicker, 6x=stimulation
            # where x=1 is left, x=2 is right

            conditions = desc // 10  # Extract condition number
            unique_conditions = np.unique(conditions)

            runs = {}
            for cond_num in sorted(unique_conditions):
                if cond_num < 1 or cond_num > 6:
                    continue

                cond_mask = conditions == cond_num
                cond_samples = mrk_samples[cond_mask]
                cond_desc = desc[cond_mask]

                # Find data boundaries for this condition
                # Include some buffer before first marker and after last marker
                buffer_before = int(1.0 * sfreq)  # 1 s before first trial
                buffer_after = int(6.0 * sfreq)  # 6 s after last trial

                start_sample = max(0, cond_samples[0] - buffer_before)
                end_sample = min(data.shape[1], cond_samples[-1] + buffer_after)

                # Extract data segment for this condition
                seg_data = data[:, start_sample:end_sample].copy()

                # Adjust marker samples to be relative to segment start
                seg_samples = cond_samples - start_sample

                cond_name = _CONDITION_NAMES.get(cond_num, f"cond{cond_num}")
                run_name = f"{cond_num}{cond_name}"

                runs[run_name] = self._make_raw(
                    seg_data, ch_names, sfreq, seg_samples, cond_desc
                )

            return runs

    def _make_raw(self, data, ch_names, sfreq, marker_samples, marker_desc):
        """Create an MNE Raw object from continuous data and markers.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            EEG data in microvolts.
        ch_names : list of str
            Channel names.
        sfreq : float
            Sampling frequency.
        marker_samples : ndarray
            Sample indices of event markers.
        marker_desc : ndarray
            Event descriptions/trigger codes.

        Returns
        -------
        raw : mne.io.RawArray
            MNE Raw object with events as annotations.
        """
        n_channels = data.shape[0]

        # Scale from microvolts to volts
        data_volts = data.astype(np.float64) * 1e-6

        # Build stim channel
        stim = np.zeros((1, data_volts.shape[1]))
        for samp, code in zip(marker_samples, marker_desc):
            if 0 <= samp < stim.shape[1]:
                # Map event codes: codes ending in 1 -> left_hand (1),
                # codes ending in 2 -> right_hand (2)
                class_code = code % 10
                if class_code in (1, 2):
                    stim[0, samp] = class_code

        # Combine EEG and stim channels
        full_data = np.vstack([data_volts, stim])

        ch_names_full = list(ch_names) + ["STI"]
        ch_types = ["eeg"] * n_channels + ["stim"]
        info = mne.create_info(ch_names_full, sfreq, ch_types)
        raw = RawArray(data=full_data, info=info, verbose=False)

        # Set montage
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

        return raw
