"""Physionet Motor imagery dataset."""

import mne
import numpy as np
from mne.io import read_raw_edf

from moabb.datasets.base import BaseDataset
from moabb.datasets.download import data_dl, get_dataset_path
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.utils import _handle_deprecated_kwargs


BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset.

    Physionet MI dataset: https://physionet.org/content/eegmmidb/1.0.0/
    (dataset DOI 10.13026/C28G6P, released on PhysioNet in 2009).

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers [2]_. The only associated publication is the
    BCI2000 platform paper [1]_, which describes the recording software, not
    this experiment; the protocol below is taken from the PhysioNet record.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org) [1]_.
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    .. note::
        Subject 88 was recorded at 128 Hz instead of 160 Hz like all other
        subjects. Loading subject 88 together with other subjects will cause
        errors in any paradigm due to incompatible sampling rates. To avoid
        this, exclude subject 88 when loading the full dataset, e.g.
        ``PhysionetMI(subjects=[s for s in range(1, 110) if s != 88])``.

    Parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=160.0,
            channel_types={"eeg": 64},
            software="BCI2000",
            sensors=[
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FT8",
                "T7",
                "T8",
                "T9",
                "T10",
                "TP7",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=60.0,
            sensor_type="EEG",
            montage="standard_1020",
        ),
        participants=ParticipantMetadata(
            n_subjects=109, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events={"left_hand": 2, "right_hand": 3, "feet": 5, "hands": 4, "rest": 1},
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "rest"],
            study_design="14 runs per subject: two one-minute baseline recordings (eyes open, eyes closed) followed by three repetitions of four two-minute task conditions (executed/imagined unilateral fist movements to left/right targets; executed/imagined bilateral fists/feet movements to top/bottom targets)",
            stimulus_type="cue-based",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="cued",
        ),
        documentation=DocumentationMetadata(
            doi="10.13026/C28G6P",
            associated_paper_doi="10.1109/TBME.2004.827072",
            investigators=[
                "Gerwin Schalk",
                "W. A. Sarnacki",
                "Aditya Joshi",
                "Dennis J. McFarland",
                "Jonathan R. Wolpaw",
            ],
            institution="Wadsworth Center, New York State Department of Health",
            country="US",
            publication_year=2009,
            senior_author="Jonathan R. Wolpaw",
            institution_address="Albany, New York, USA",
            institution_department="BCI R&D Program",
            funding=["NIH/NIBIB grants EB006356 and EB00856"],
            contact_info=["schalk@wadsworth.org"],
            keywords=[
                "brain-computer interface (BCI)",
                "electroencephalography (EEG)",
                "motor imagery",
                "motor execution",
            ],
            license="ODC-By-1.0",
            repository="Physionet",
            data_url="https://physionet.org/content/eegmmidb/1.0.0/",
            how_to_acknowledge="Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. https://doi.org/10.13026/C28G6P; additionally cite Schalk et al. (2004) BCI2000, IEEE TBME 51(6):1034-1043.",
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet", "rest"],
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        data_processed=False,
        file_format="edf",
    )
    nemar_id = "on004362"
    nemar_subject_template = "{subject:03d}"

    def __init__(
        self,
        imagined=True,
        executed=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {"Imagined": "imagined", "Executed": "executed"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, "PhysionetMotorImagery"
        )
        imagined = resolved.get("imagined", imagined)
        executed = resolved.get("executed", executed)

        super().__init__(
            subjects=list(range(1, 110)),
            sessions_per_subject=1,
            events={"left_hand": 2, "right_hand": 3, "feet": 5, "hands": 4, "rest": 1},
            code="PhysionetMotorImagery",
            # website does not specify how long the trials are, but the
            # interval between 2 trial is 4 second.
            interval=[0, 3],
            paradigm="imagery",
            doi="10.13026/C28G6P",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )
        self.events = {"left_hand": 2, "right_hand": 3, "feet": 5, "hands": 4, "rest": 1}
        self.imagined = imagined
        self.executed = executed
        self.feet_runs = []
        self.hand_runs = []

        if imagined:
            self.feet_runs += [6, 10, 14]
            self.hand_runs += [4, 8, 12]

        if executed:
            self.feet_runs += [5, 9, 13]
            self.hand_runs += [3, 7, 11]

    def _load_one_run(self, subject, run, preload=True):
        raw_fname = self._load_data(subject, runs=[run], verbose="ERROR")[0]
        raw = read_raw_edf(raw_fname, preload=preload, verbose="ERROR")
        raw.rename_channels(lambda x: x.strip("."))
        raw.rename_channels(lambda x: x.upper())
        # fmt: off
        renames = {
            "AFZ": "AFz", "PZ": "Pz", "FPZ": "Fpz", "FCZ": "FCz", "FP1": "Fp1", "CZ": "Cz",
            "OZ": "Oz", "POZ": "POz", "IZ": "Iz", "CPZ": "CPz", "FP2": "Fp2", "FZ": "Fz",
        }
        # fmt: on
        raw.rename_channels(renames)
        raw.set_montage(mne.channels.make_standard_montage("colin27_1005"))
        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        data = {}
        sign = "EEGBCI"
        get_dataset_path(sign, None)

        # hand runs
        idx = 0
        for run in self.hand_runs:
            raw = self._load_one_run(subject, run)
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "left_hand"
            stim[stim == "T2"] = "right_hand"
            raw.annotations.description = stim
            data[str(idx)] = stim_channels_with_selected_ids(
                raw, desired_event_id=self.events
            )
            idx += 1

        # feet runs
        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)
            # modify stim channels to match new event ids. for feet runs,
            # hand = 2 modified to 4, and feet = 3, modified to 5
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "hands"
            stim[stim == "T2"] = "feet"
            raw.annotations.description = stim
            data[str(idx)] = stim_channels_with_selected_ids(
                raw, desired_event_id=self.events
            )
            idx += 1

        return {"0": data}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        runs = [1, 2] + self.hand_runs + self.feet_runs

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sign = "EEGBCI"
        get_dataset_path(sign, None)
        paths = self._load_data(
            subject, runs=runs, path=path, force_update=force_update, verbose=verbose
        )
        return paths

    def _load_data(self, subject, runs, path=None, force_update=False, verbose=None):
        # Function to load the data run by run
        if not hasattr(runs, "__iter__"):
            runs = [runs]

        # get local storage path
        sign = "EEGBCI"
        path = get_dataset_path(sign, path)

        # fetch the file(s)
        data_paths = []
        for run in runs:
            file_part = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            url = BASE_URL + file_part
            p = data_dl(url, sign, path, force_update, verbose)
            data_paths.append(p)
        return data_paths
