"""Comprehensive metadata catalog for all MOABB datasets.

This module provides pre-defined DatasetMetadata instances for all MOABB datasets,
extracted from the dataset source code, CSV summaries, and associated publications.

Sources:
- Dataset source files in moabb/datasets/
- CSV summary files (summary_imagery.csv, summary_p300.csv, etc.)
- Original publications and data repositories
- Web searches for verification and enrichment (January 2026)
"""

from .schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)


# =============================================================================
# Motor Imagery Datasets
# =============================================================================

ALEXMI_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=17,
        channel_types={"eeg": 16, "stim": 1},
        sensors=[
            "Fpz",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
        ],
        hardware="g.tec g.USBamp",
        sensor_type="wet",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=8,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="right_hand_feet_rest",
        n_classes=3,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.806023",
        description="Motor imagery dataset from A. Barachant PhD dissertation",
        investigators=["Barachant, A."],
        institution="University of Grenoble",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/806023",
        license="CC BY 4.0",
        publication_year=2012,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2014_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=22,
        channel_types={"eeg": 22, "eog": 3, "stim": 1},
        sensors=[
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ],
        sensor_type="Ag/AgCl wet",
        reference="left mastoid",
        ground="right mastoid",
        filters="0.5-100 Hz bandpass, 50 Hz notch",
        line_freq=50.0,
        montage="standard_1005",
    ),
    participants=ParticipantMetadata(
        n_subjects=9,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="4_class_limbs_tongue",
        n_classes=4,
        trials_per_class={"left_hand": 72, "right_hand": 72, "feet": 72, "tongue": 72},
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2012.00055",
        description="BCI Competition IV Dataset 2a - 4-class motor imagery",
        investigators=[
            "M. Tangermann",
            "K.R. Müller",
            "C. Brunner",
            "R. Leeb",
            "G.R. Müller-Putz",
            "G. Pfurtscheller",
            "A. Schlögl",
        ],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2014/",
        license="CC BY 4.0",
        publication_year=2012,
    ),
    sessions_per_subject=2,
    runs_per_session=6,
)

BNCI2014_002_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=16,
        channel_types={"eeg": 15, "stim": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=14,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="right_hand_feet",
        n_classes=2,
        trials_per_class={"right_hand": 80, "feet": 80},
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2012.00055",
        description="Motor imagery dataset - right hand and feet kinesthetic MI",
        investigators=["Steyrl, D.", "Scherer, R.", "Faller, J.", "Müller-Putz, G.R."],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/002-2014/",
        license="CC BY 4.0",
        publication_year=2012,
    ),
    sessions_per_subject=1,
    runs_per_session=8,
)

BNCI2014_004_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=7,
        channel_types={"eeg": 3, "eog": 3, "stim": 1},
        sensors=["C3", "Cz", "C4"],
        filters="0.5-100 Hz bandpass, 50 Hz notch",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=9,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=4.5,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2007.906956",
        description="BCI Competition IV Dataset 2b - 2-class motor imagery with EOG artifacts",
        investigators=["R. Leeb", "C. Brunner", "G.R. Müller-Putz", "G. Pfurtscheller"],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/004-2014/",
        license="CC BY 4.0",
        publication_year=2007,
    ),
    sessions_per_subject=5,
    runs_per_session=1,
)

BNCI2015_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=14,
        channel_types={"eeg": 13, "stim": 1},
        sensors=[
            "FC3",
            "FCz",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CPz",
            "CP4",
        ],
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="right_hand_feet",
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/tnsre.2012.2189584",
        description="BNCI 2015-001 Motor imagery dataset",
        investigators=[
            "J. Faller",
            "R. Scherer",
            "U. Costa",
            "E. Opisso",
            "J. Medina",
            "G.R. Müller-Putz",
        ],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2015/",
        license="CC BY 4.0",
        publication_year=2014,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BNCI2015_004_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=31,
        channel_types={"eeg": 30, "stim": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=9,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="5_class_mental_tasks",
        n_classes=5,
        trial_duration=7.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TCDS.2017.2688350",
        description="5-class mental tasks for users with disability",
        investigators=[
            "X. Zhang",
            "L. Yao",
            "Q. Zhang",
            "S. Kanhere",
            "M. Sheng",
            "Y. Liu",
        ],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/004-2015/",
        license="CC BY 4.0",
        publication_year=2017,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

BNCI2003_004_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=100.0,
        n_channels=118,
        channel_types={"eeg": 118},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=5,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="right_hand_feet",
        n_classes=2,
        trial_duration=3.5,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TBME.2004.827088",
        description="BCI Competition III Dataset IVa - 2-class motor imagery",
        investigators=[
            "G. Dornhege",
            "B. Blankertz",
            "G. Curio",
            "K.R. Müller",
        ],
        institution="Berlin Institute of Technology",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/004-2003/",
        license="CC BY 4.0",
        publication_year=2004,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CHO2017_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=69,
        channel_types={"eeg": 64, "emg": 4, "stim": 1},
        hardware="BrainAmp",
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=52,
        health_status="healthy",
        gender={"female": 19, "male": 33},
        age_mean=24.8,
        age_std=3.86,
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trials_per_class={"left_hand": 100, "right_hand": 100},
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/gix034",
        description="Motor imagery BCI EEG dataset with EMG from GigaScience",
        investigators=["H. Cho", "M. Ahn", "S. Ahn", "M. Kwon", "S.C. Jun"],
        institution="Gwangju Institute of Science and Technology",
        country="KR",
        repository="GigaDB",
        data_url="http://dx.doi.org/10.5524/100295",
        license="CC0",
        publication_year=2017,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

LEE2019_MI_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=62,
        channel_types={"eeg": 62},
        sensor_type="Ag/AgCl wet",
        hardware="BrainAmp (Brain Products)",
        reference="nasion",
        ground="AFz",
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=54,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/giz002",
        description="OpenBMI Motor Imagery dataset - investigation into BCI illiteracy",
        investigators=[
            "M.H. Lee",
            "O.Y. Kwon",
            "Y.J. Kim",
            "H.K. Kim",
            "Y.E. Lee",
            "J. Williamson",
            "S. Fazli",
            "S.W. Lee",
        ],
        institution="Korea University",
        country="KR",
        repository="GigaDB",
        data_url="http://deepbci.korea.ac.kr/opensource/opendb/",
        license="CC0",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

PHYSIONETMI_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=160.0,
        n_channels=65,
        channel_types={"eeg": 64, "stim": 1},
        hardware="BCI2000",
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=109,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="4_class_hands_feet",
        n_classes=4,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.13026/C28G6P",
        description="PhysioNet EEG Motor Movement/Imagery Dataset",
        investigators=[
            "G. Schalk",
            "D.J. McFarland",
            "T. Hinterberger",
            "N. Birbaumer",
            "J.R. Wolpaw",
        ],
        institution="Wadsworth Center, New York State Department of Health",
        country="US",
        repository="PhysioNet",
        data_url="https://physionet.org/content/eegmmidb/1.0.0/",
        license="Open Data Commons Attribution License v1.0",
        publication_year=2009,
    ),
    sessions_per_subject=1,
    runs_per_session=14,
)

SCHIRRMEISTER2017_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=128,
        channel_types={"eeg": 128},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=14,
        health_status="healthy",
        gender={"female": 6, "male": 8},
        age_mean=27.2,
        age_std=3.6,
        handedness={"right": 12, "left": 2},
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="4_class_motor_execution",
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1002/hbm.23730",
        description="High-Gamma Dataset for deep learning motor imagery/execution",
        investigators=[
            "R.T. Schirrmeister",
            "J.T. Springenberg",
            "L.D.J. Fiederer",
            "M. Glasstetter",
            "K. Eggensperger",
            "M. Tangermann",
            "F. Hutter",
            "W. Burgard",
            "T. Ball",
        ],
        institution="University of Freiburg",
        country="DE",
        repository="GIN",
        data_url="https://gin.g-node.org/robintibor/high-gamma-dataset",
        publication_year=2017,
        license="CC BY-SA 4.0",
    ),
    sessions_per_subject=1,
    runs_per_session=2,
)

GROSSEWENTRUP2009_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=128,
        channel_types={"eeg": 128},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=10,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=7.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TBME.2008.2009768",
        description="Beamforming in Noninvasive Brain-Computer Interfaces",
        investigators=["M. Grosse-Wentrup", "C. Liefhold", "K. Gramann", "M. Buss"],
        institution="Max Planck Institute for Biological Cybernetics",
        country="DE",
        repository="Zenodo",
        data_url="https://zenodo.org/records/1217449",
        license="CC BY 4.0",
        publication_year=2009,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

SHIN2017A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=30,
        channel_types={"eeg": 30},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=29,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=10.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2016.2628057",
        description="EEG/fNIRS hybrid motor imagery dataset - Modality A (EEG only)",
        investigators=[
            "Shin, Jaeyoung",
            "von Lühmann, Alexander",
            "Blankertz, Benjamin",
            "Kim, Do-Won",
            "Jeong, Jichai",
            "Hwang, Han-Jeong",
            "Müller, Klaus-Robert",
        ],
        institution="Korea University / TU Berlin",
        country="KR",
        repository="TU Berlin",
        data_url="http://doc.ml.tu-berlin.de/hBCI",
        license="GNU GPLv3",
        publication_year=2017,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

WEIBO2014_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=65,
        channel_types={"eeg": 60, "misc": 2, "eog": 2, "stim": 1},
        hardware="Neuroscan SynAmps2",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=10,
        health_status="healthy",
        gender={"female": 3, "male": 7},
        handedness={"right": 10},
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="7_class_simple_compound_limb_MI",
        n_classes=7,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0114853",
        description="Simple and compound limb motor imagery EEG patterns",
        investigators=["W. Yi", "S. Qiu", "H. Qi", "L. Zhang", "B. Wan", "D. Ming"],
        institution="Tianjin University",
        country="CN",
        repository="Harvard Dataverse",
        data_url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27306",
        license="CC BY 4.0",
        publication_year=2014,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ZHOU2016_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=14,
        channel_types={"eeg": 14},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=4,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="3_class",
        n_classes=3,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0162657",
        description="Motor imagery dataset - 3 class",
        investigators=["B. Zhou", "X. Wu", "Z. Lv", "L. Zhang", "X. Guo"],
        institution="Anhui University",
        country="CN",
        repository="Zenodo",
        data_url="https://zenodo.org/record/16534752",
        license="Public Domain",
        publication_year=2016,
    ),
    sessions_per_subject=3,
    runs_per_session=2,
)

STIEGER2021_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=64,
        channel_types={"eeg": 64},
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=62,
        health_status="healthy",
        handedness={"right": 62},
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="continuous_cursor_control",
        n_classes=4,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-021-00883-1",
        description="Large-scale continuous sensorimotor rhythm BCI learning dataset",
        investigators=["J.R. Stieger", "S.A. Engel", "B. He"],
        institution="Carnegie Mellon University",
        country="US",
        repository="Figshare",
        data_url="https://doi.org/10.6084/m9.figshare.13123148",
        publication_year=2021,
        license="CC BY 4.0",
    ),
    sessions_per_subject=11,
    runs_per_session=1,
)

LIU2024_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=31,
        channel_types={"eeg": 29, "eog": 2},
        hardware="ZhenTec NT1 wireless multichannel EEG acquisition system",
        sensor_type="Ag/AgCl semi-dry electrodes",
        reference="CPz",
        ground="FPz",
        montage="standard_1010",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=50,
        health_status="patients",
        clinical_population="acute stroke",
        gender={"male": 39, "female": 11},
        age_mean=56.70,
        age_std=10.57,
        age_min=31.0,
        age_max=77.0,
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02787-8",
        description="Motor imagery EEG dataset from acute stroke patients",
        investigators=["H. Liu", "X. Lv", "P. Wei", "H. Wang"],
        institution="Xuanwu Hospital of Capital Medical University",
        country="CN",
        repository="Figshare",
        data_url="https://figshare.com/articles/dataset/MI-BCI_EEG_data/21679035",
        license="CC BY 4.0",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

DREYER2023_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=27,
        channel_types={"eeg": 27, "eog": 3, "emg": 2},
        sensors=[
            "Fz",
            "FCz",
            "Cz",
            "CPz",
            "Pz",
            "C1",
            "C3",
            "C5",
            "C2",
            "C4",
            "C6",
            "EOG1",
            "EOG2",
            "EOG3",
            "EMGg",
            "EMGd",
            "F4",
            "FC2",
            "FC4",
            "FC6",
            "CP2",
            "CP4",
            "CP6",
            "P4",
            "F3",
            "FC1",
            "FC3",
            "FC5",
            "CP1",
            "CP3",
            "CP5",
            "P3",
        ],
        hardware="g.USBAmp (g.tec)",
        reference="left earlobe",
        software="OpenViBE 2.1.0/2.2.0",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=87,
        health_status="healthy",
        age_mean=29.0,
        age_std=9.3,
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trials_per_class={"left_hand": 120, "right_hand": 120},
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Large EEG database with user profiles for MI BCI research",
        investigators=[
            "P. Dreyer",
            "A. Roc",
            "L. Pillette",
            "S. Rimbert",
            "F. Lotte",
        ],
        institution="Inria Bordeaux",
        country="FR",
        repository="Zenodo",
        data_url="https://doi.org/10.5281/zenodo.8089820",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

OFNER2017_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=61,
        channel_types={"eeg": 61},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=45,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="upper_limb_movements",
        n_classes=7,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0182578",
        description="Upper limb movements decoded from low-frequency EEG time-domain",
        investigators=["P. Ofner", "A. Schwarz", "J. Pereira", "G.R. Müller-Putz"],
        institution="Graz University of Technology, BCI-Lab",
        country="AT",
        repository="Zenodo",
        data_url="https://zenodo.org/record/834976",
        license="CC BY 4.0",
        publication_year=2017,
    ),
    sessions_per_subject=1,
    runs_per_session=10,
)

BNCI2019_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=64,
        channel_types={"eeg": 61, "eog": 3},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=10,
        health_status="patients",
        clinical_population="spinal cord injury",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="upper_limb_grasp",
        n_classes=5,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41598-019-43594-9",
        description="Motor imagery of grasp movements in spinal cord injury patients",
        investigators=["A. Schwarz", "P. Ofner", "J. Pereira", "G.R. Müller-Putz"],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2019/",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=10,
)

BNCI2020_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=64,
        channel_types={"eeg": 58, "eog": 6},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=45,  # Fixed: was incorrectly 15
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="reach_and_grasp",
        n_classes=3,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2020.00849",
        description="Reach-and-grasp electrode comparison (gel, water, dry)",
        investigators=["A. Schwarz", "J. Pereira", "G.R. Müller-Putz"],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2020/",
        license="CC BY 4.0",
        publication_year=2020,
    ),
    sessions_per_subject=1,
    runs_per_session=4,
)

BNCI2024_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,  # Fixed: was incorrectly 512.0
        n_channels=65,
        channel_types={"eeg": 60, "eog": 4, "stim": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=20,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="handwritten_character_imagery",
        n_classes=10,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.compbiomed.2024.109132",
        description="Handwritten character classification from EEG",
        investigators=["Crell, M.R.", "Müller-Putz, G.R."],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2024/",
        license="CC BY 4.0",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2022_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=67,
        channel_types={"eeg": 64, "eog": 3},
        hardware="Biosemi ActiveTwo",
        montage="biosemi64",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=13,
        health_status="healthy",
        gender={"female": 8, "male": 5},
        age_mean=22.6,
        age_std=1.04,
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="cognitive_workload",
        n_classes=4,
        trial_duration=90.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/THMS.2020.3038339",
        description="EEG correlates of difficulty level in simulated drone piloting",
        investigators=["P.-K. Jao", "R. Chavarriaga", "J. d. R. Millán"],
        institution="EPFL",
        country="CH",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2022/",
        license="CC BY 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2025_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=71,
        channel_types={"eeg": 67, "eog": 4},
        hardware="BrainAmp (Brain Products GmbH)",
        reference="common average",
        montage="standard_1005",
        filters="0.3-100 Hz bandpass, 50 Hz notch",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=20,
        health_status="healthy",
        gender={"male": 12, "female": 8},
        age_mean=26.1,
        age_std=4.1,
        handedness={"right": 17, "left": 3},
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="motor_kinematics_reaching",
        n_classes=16,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2552/ada0ea",
        description="Simultaneous encoding of speed, distance, and direction in discrete reaching",
        investigators=["N. Srisrisawang", "G. R. Müller-Putz"],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2025/",
        license="CC BY 4.0",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2025_002_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=64,
        channel_types={"eeg": 60, "eog": 4},
        hardware="actiCAP system (Brain Products GmbH)",
        reference="common average",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=2,
        health_status="healthy",
        gender={"male": 10, "female": 10},
        age_mean=24.0,
        age_std=5.0,
        handedness={"right": 20},
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="continuous_2d_trajectory_decoding",
        n_classes=3,
        trial_duration=8.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2552/ac689f",
        description="Continuous 2D trajectory decoding from attempted movement",
        investigators=[
            "R. J. Kobler",
            "I. Almeida",
            "A. I. Sburlea",
            "G. R. Müller-Putz",
        ],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/002-2025/",
        license="CC BY 4.0",
        publication_year=2022,
    ),
    sessions_per_subject=3,
    runs_per_session=3,
)

SHIN2017B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=32,
        channel_types={"eeg": 30, "eog": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=29,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="mental_arithmetic",
        n_classes=2,
        trial_duration=10.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2016.2628057",
        description="EEG/fNIRS hybrid mental arithmetic dataset - Modality B (EEG only)",
        investigators=[
            "Shin, Jaeyoung",
            "von Lühmann, Alexander",
            "Blankertz, Benjamin",
            "Kim, Do-Won",
            "Jeong, Jichai",
            "Hwang, Han-Jeong",
            "Müller, Klaus-Robert",
        ],
        institution="Korea University / TU Berlin",
        country="KR",
        repository="TU Berlin",
        data_url="http://doc.ml.tu-berlin.de/hBCI",
        license="GNU GPLv3",
        publication_year=2017,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BEETL2021_A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=63,
        channel_types={"eeg": 63},
        filters="1-100 Hz bandpass, 50 Hz notch",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=3,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="4_class_motor_imagery",
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.48550/arXiv.2202.12950",
        description="BEETL 2021 Challenge Dataset A - Motor Imagery",
        investigators=[
            "X. Wei",
            "A.A. Faisal",
            "M. Grosse-Wentrup",
            "A. Gramfort",
            "S. Chevallier",
            "V. Jayaram",
            "P. Tempczyk",
        ],
        institution="Imperial College London",
        country="GB",
        repository="BEETL Competition",
        data_url="https://beetl.ai/introduction",
        license="CC BY 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BEETL2021_B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=32,
        channel_types={"eeg": 32},
        filters="1-100 Hz bandpass",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=2,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="4_class_motor_imagery",
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.48550/arXiv.2202.12950",
        description="BEETL 2021 Challenge Dataset B - Motor Imagery",
        investigators=[
            "X. Wei",
            "A.A. Faisal",
            "M. Grosse-Wentrup",
            "A. Gramfort",
            "S. Chevallier",
            "V. Jayaram",
            "P. Tempczyk",
        ],
        institution="Imperial College London",
        country="GB",
        repository="BEETL Competition",
        data_url="https://beetl.ai/introduction",
        license="CC BY 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

DREYER2023A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=32,
        channel_types={"eeg": 27, "eog": 3, "emg": 2},
        hardware="g.USBAmp (g.tec)",
        reference="left earlobe",
        software="OpenViBE",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=60,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Dreyer2023 Dataset A - Large MI dataset user profile study",
        investigators=["P. Dreyer", "A. Roc", "L. Pillette", "S. Rimbert", "F. Lotte"],
        institution="Inria Bordeaux",
        country="FR",
        repository="Zenodo",
        data_url="https://osf.io/8tdk5/",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

DREYER2023B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=32,
        channel_types={"eeg": 27, "eog": 3, "emg": 2},
        hardware="g.USBAmp (g.tec)",
        reference="left earlobe",
        software="OpenViBE",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=21,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Dreyer2023 Dataset B - Large MI dataset user profile study",
        investigators=["P. Dreyer", "A. Roc", "L. Pillette", "S. Rimbert", "F. Lotte"],
        institution="Inria Bordeaux",
        country="FR",
        repository="Zenodo",
        data_url="https://osf.io/8tdk5/",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

DREYER2023C_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=32,
        channel_types={"eeg": 27, "eog": 3, "emg": 2},
        hardware="g.USBAmp (g.tec)",
        reference="left earlobe",
        software="OpenViBE",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=6,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Dreyer2023 Dataset C - Large MI dataset user profile study",
        investigators=["P. Dreyer", "A. Roc", "L. Pillette", "S. Rimbert", "F. Lotte"],
        institution="Inria Bordeaux",
        country="FR",
        repository="Zenodo",
        data_url="https://osf.io/8tdk5/",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

# =============================================================================
# P300/ERP Datasets
# =============================================================================

BNCI2014_008_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=10,
        channel_types={"eeg": 8, "stim": 2},
        reference="linked mastoids",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=8,
        health_status="patients",
        clinical_population="ALS",
        gender={"male": 6, "female": 2},
        age_min=25.0,
        age_max=60.0,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnhum.2013.00732",
        description="P300 speller for ALS patients",
        investigators=[
            "A. Riccio",
            "L. Simione",
            "F. Schettini",
            "A. Pizzimenti",
            "M. Inghilleri",
            "M.O. Belardinelli",
            "D. Mattia",
        ],
        institution="Fondazione Santa Lucia",
        country="IT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/008-2014/",
        license="CC BY 4.0",
        publication_year=2013,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2014_009_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=18,
        channel_types={"eeg": 16, "stim": 2},
        reference="linked mastoids",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=10,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="speller",
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/11/3/035008",
        description="P300 grid speller dataset",
        investigators=[
            "A. Riccio",
            "L. Simione",
            "F. Schettini",
            "A. Pizzimenti",
            "M. Inghilleri",
            "M.O. Belardinelli",
            "D. Mattia",
        ],
        institution="Fondazione Santa Lucia",
        country="IT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/009-2014/",
        license="CC BY 4.0",
        publication_year=2013,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BNCI2015_003_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=10,
        channel_types={"eeg": 8, "stim": 2},
        sensors=["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8"],
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=10,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="speller",
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neulet.2009.06.045",
        description="BNCI 2015-003 P300 Speller Dataset",
        investigators=["Schreuder, M.", "Rost, T.", "Tangermann, M."],
        institution="Graz University of Technology",
        country="AT",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/003-2015/",
        license="CC BY 4.0",
        publication_year=2009,
    ),
    sessions_per_subject=1,
    runs_per_session=2,
)

BNCI2015_006_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,  # Fixed: was incorrectly 100.0
        n_channels=65,
        channel_types={"eeg": 64, "stim": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=11,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="music_auditory_attention",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/11/2/026009",
        description="Music BCI - Auditory attention detection with ERP",
        investigators=[
            "Treder, M.S.",
            "Purwins, H.",
            "Miklody, D.",
            "Sturm, I.",
            "Blankertz, B.",
        ],
        institution="Technische Universität Berlin",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/006-2015/",
        license="CC BY 4.0",
        publication_year=2014,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2015_007_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=100.0,
        n_channels=65,
        channel_types={"eeg": 63, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
        reference="nose",
        hardware="BrainProducts actiCap",
        sensor_type="active",
    ),
    participants=ParticipantMetadata(
        n_subjects=16,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="motion_vep_speller",
        n_classes=2,
        trial_duration=0.7,  # Fixed: was incorrectly 1.0
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/9/4/045006",
        description="Motion VEP Speller - mVEP based BCI speller",
        investigators=["Treder, M.S.", "Blankertz, B."],
        institution="Technische Universität Berlin",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/007-2015/",
        license="CC BY 4.0",
        publication_year=2012,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2015_008_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=65,
        channel_types={"eeg": 63, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=13,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="center_speller",
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/8/6/066003",
        description="Center Speller P300 - gaze-independent BCI speller",
        investigators=[
            "Treder, M.S.",
            "Schmidt, N.M.",
            "Blankertz, B.",
        ],
        institution="Berlin Institute of Technology",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/008-2015/",
        license="CC BY 4.0",
        publication_year=2011,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

BNCI2015_009_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,  # Fixed: was incorrectly 1000.0
        n_channels=64,
        channel_types={"eeg": 60, "eog": 2, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=21,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="amuse_auditory_spatial",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0009813",
        description="AMUSE - Auditory Multi-class Spatial ERP BCI",
        investigators=["Höhne, J.", "Schreuder, M.", "Blankertz, B.", "Tangermann, M."],
        institution="Berlin Institute of Technology",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/009-2015/",
        license="CC BY 4.0",
        publication_year=2010,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2015_010_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,  # Fixed: was incorrectly 1000.0
        n_channels=65,
        channel_types={"eeg": 63, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="rsvp",
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.clinph.2012.12.050",
        description="RSVP - Rapid Serial Visual Presentation BCI",
        investigators=["Acqualagna, L.", "Blankertz, B."],
        institution="Technische Universität Berlin",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/010-2015/",
        license="CC BY 4.0",
        publication_year=2013,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2015_012_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,  # Fixed: was incorrectly 1000.0
        n_channels=65,
        channel_types={"eeg": 63, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=10,  # Fixed: was incorrectly 12
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="pass2d_auditory_speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2011.00099",
        description="PASS2D - Auditory Spatial Speller BCI",
        investigators=["Höhne, J.", "Schreuder, M.", "Blankertz, B.", "Tangermann, M."],
        institution="Technische Universität Berlin",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/012-2015/",
        license="CC BY 4.0",
        publication_year=2011,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2015_013_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=65,
        channel_types={"eeg": 64, "stim": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=6,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="error_related_potentials",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2010.2053387",
        description="Error-Related Potentials (ErrP) for BCI error detection",
        investigators=[
            "Chavarriaga, R.",
            "Millán, J.d.R.",
        ],
        institution="EPFL",
        country="CH",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/013-2015/",
        license="CC BY 4.0",
        publication_year=2010,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

BNCI2016_002_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=69,
        channel_types={"eeg": 59, "eog": 2, "emg": 1, "misc": 7},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=15,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="emergency_braking",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/8/5/056001",
        description="Emergency braking during simulated driving - ERP detection",
        investigators=[
            "Haufe, S.",
            "Treder, M.S.",
            "Gugler, M.F.",
            "Sagebaum, M.",
            "Curio, G.",
            "Blankertz, B.",
        ],
        institution="TU Berlin / Charité University Medicine Berlin",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/002-2016/",
        license="CC BY 4.0",
        publication_year=2011,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2020_002_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=33,
        channel_types={"eeg": 29, "eog": 2, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=18,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="covert_spatial_attention",
        n_classes=2,
        trial_duration=16.0,  # Fixed: was incorrectly 1.0
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2020.591777",
        description="Attention Shift - Covert Spatial Attention BCI",
        investigators=[
            "Reichert, C.",
            "Tellez Ceja, I.F.",
            "Sweeney-Reed, C.M.",
            "Heinze, H.J.",
            "Hinrichs, H.",
            "Dürschmid, S.",
        ],
        institution="Otto von Guericke University Magdeburg",
        country="DE",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/002-2020/",
        license="CC BY 4.0",
        publication_year=2020,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2012_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=128.0,
        n_channels=17,
        channel_types={"eeg": 16, "stim": 1},
        sensors=[
            "Fp1",
            "Fp2",
            "F5",
            "AFz",
            "F6",
            "T7",
            "Cz",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
        ],
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=25,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brain_invaders",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.2649006",
        description="Brain Invaders P300 dataset 2012 - experimental validation",
        investigators=["Van Veen, G.", "Barachant, A.", "Congedo, M."],
        institution="GIPSA-lab, University of Grenoble Alpes",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2649069",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=2,
)

BI2013A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=16,
        channel_types={"eeg": 16},
        sensors=[
            "Fp1",
            "Fp2",
            "F5",
            "AFz",
            "F6",
            "T7",
            "Cz",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
        ],
        hardware="Nexus (TMSi)",
        sensor_type="Ag/AgCl wet",
        reference="left ear-lobe",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=24,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brain_invaders_adaptive",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.1494163",
        description="Brain Invaders adaptive vs non-adaptive P300 BCI",
        investigators=["Vaineau, E.", "Barachant, A.", "Congedo, M."],
        institution="GIPSA-lab, CNRS, Grenoble-INP",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2669187",
        license="CC BY 4.0",
        publication_year=2018,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2014A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=17,
        channel_types={"eeg": 16, "stim": 1},
        sensors=[
            "Fp1",
            "Fp2",
            "F3",
            "AFz",
            "F4",
            "T7",
            "Cz",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
        ],
        sensor_type="active dry",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=64,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brain_invaders_calibration_less",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3266222",
        description="Brain Invaders calibration-less P300 BCI with dry electrodes",
        investigators=["Korczowski, L.", "Ostaschenko, E.", "Congedo, M."],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/3266222",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2014B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=33,
        channel_types={"eeg": 32, "stim": 1},
        sensor_type="active wet",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=38,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brain_invaders_multi_user",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3267301",
        description="Brain Invaders Solo vs Collaboration multi-user P300 BCI",
        investigators=["Korczowski, L.", "Ostaschenko, E.", "Congedo, M."],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/3267301",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2015A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=33,
        channel_types={"eeg": 32, "stim": 1},
        sensor_type="active wet",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=43,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brain_invaders_flash_duration",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3266929",
        description="Brain Invaders with modulation of flash duration",
        investigators=["Korczowski, L.", "Cederhout, M.", "Congedo, M."],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/3266929",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BI2015B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=33,
        channel_types={"eeg": 32, "stim": 1},
        sensor_type="active wet",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=44,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brain_invaders_cooperation_competition",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3267307",
        description="Brain Invaders Cooperative vs Competitive multi-user P300",
        investigators=["Korczowski, L.", "Cederhout, M.", "Congedo, M."],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/3267307",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=4,
)

CATTAN2019_VR_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=17,
        channel_types={"eeg": 16, "stim": 1},
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=21,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="virtual_reality_p300",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.2605204",
        description="EEG-based BCI in Virtual Reality using P300",
        investigators=["G. Cattan", "A. Andreev", "P.L.C. Rodrigues", "M. Congedo"],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        license="CC BY 4.0",
        publication_year=2019,
        data_url="https://zenodo.org/record/2605204",
    ),
    sessions_per_subject=2,
    runs_per_session=5,
)

EPFLP300_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=2048.0,
        n_channels=35,
        channel_types={"eeg": 32, "misc": 2, "stim": 1},
        hardware="Biosemi ActiveTwo",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=8,
        health_status="patients",
        clinical_population="disabled",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.jneumeth.2007.03.005",
        description="EPFL P300 BCI dataset for disabled users",
        investigators=["Hoffmann, U.", "Vesin, J.M.", "Ebrahimi, T.", "Diserens, K."],
        institution="EPFL",
        country="CH",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets",
        license="CC BY 4.0",
        publication_year=2008,
    ),
    sessions_per_subject=4,
    runs_per_session=1,
)

LEE2019_ERP_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=62,
        channel_types={"eeg": 62},
        sensor_type="Ag/AgCl wet",
        hardware="BrainAmp (Brain Products)",
        reference="nasion",
        ground="AFz",
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=54,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="row_col_speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/giz002",
        description="OpenBMI ERP dataset - investigation into BCI illiteracy",
        investigators=["M.H. Lee", "S.W. Lee"],
        institution="Korea University",
        country="KR",
        repository="GigaDB",
        data_url="https://gigadb.org/dataset/100542",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=2,
)

DEMONSP300_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=10,
        channel_types={"eeg": 8, "misc": 1, "stim": 1},
        sensors=["Cz", "P3", "Pz", "P4", "PO3", "PO4", "O1", "O2"],
        hardware="NVX-52 encephalograph (MCS, Zelenograd, Russia)",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=60,
        health_status="healthy",
        gender={"male": 23, "female": 37},
        age_mean=28.0,
        age_min=19.0,
        age_max=45.0,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="vr_speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        description="Visual P300 BCI dataset recorded in VR game Raccoons versus Demons",
        doi="10.48550/arXiv.2005.02251",
        investigators=["V. Goncharenko", "R. Grigoryan", "A. Samokhina"],
        institution="Neiry",
        country="RU",
        repository="GIN",
        data_url="https://gin.g-node.org/v-goncharenko/neiry-demons",
        license="CC BY 4.0",
        publication_year=2020,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_N170_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
        handedness={"right": 38, "left": 2},
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="n170_face_car",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE N170 - Face vs object perception paradigm",
        investigators=[
            "E.S. Kappenman",
            "J.L. Farrens",
            "W. Zhang",
            "A.X. Stewart",
            "S.J. Luck",
        ],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_MMN_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="mmn_auditory",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE MMN - Mismatch Negativity (passive auditory oddball)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_P3_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="p3_active_visual_oddball",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE P3 - Active visual oddball paradigm",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_N400_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="n400_word_pair_judgement",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE N400 - Semantic processing (word pair judgement)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_ERN_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="ern_error_related_negativity",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE ERN - Error-related negativity (Flankers task)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_LRP_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="lrp_lateralized_readiness",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE LRP - Lateralized Readiness Potential",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_N2PC_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=33,
        channel_types={"eeg": 30, "eog": 3},
        hardware="Biosemi ActiveTwo",
        reference="CMS/DRL",
        montage="standard_1020",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=40,
        health_status="healthy",
        gender={"female": 25, "male": 15},
        age_mean=21.5,
        age_std=2.87,
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="n2pc_attention_contralateral",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE N2pc - Attention-related N2pc (visual search)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="US",
        repository="OSF",
        data_url="https://osf.io/thsqg/",
        license="CC BY-SA 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

HUEBNER2017_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=31,
        channel_types={"eeg": 31},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=13,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="visual_matrix_speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0175856",
        description="Visual Matrix Speller LLP dataset 2017",
        investigators=["D. Huebner", "T. Verhoeven", "K.R. Müller", "P.J. Kindermans"],
        institution="Albert-Ludwigs-University Freiburg",
        country="DE",
        repository="Zenodo",
        data_url="https://zenodo.org/records/5831826",
        publication_year=2017,
        license="CC BY 4.0",
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

HUEBNER2018_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=31,
        channel_types={"eeg": 31},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="visual_matrix_speller",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/MCI.2018.2807039",
        description="Visual Matrix Speller LLP dataset 2018",
        investigators=["D. Huebner", "T. Verhoeven", "K.R. Müller", "P.J. Kindermans"],
        institution="Albert-Ludwigs-University Freiburg",
        country="DE",
        repository="Zenodo",
        data_url="https://zenodo.org/records/5831879",
        publication_year=2018,
        license="CC BY 4.0",
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

KOJIMA2024A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=66,
        channel_types={"eeg": 64, "eog": 2},
        hardware="BrainAmp (Brain Products, Germany)",
        sensor_type="Ag/AgCl",
        reference="right mastoid",
        ground="left mastoid",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=11,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="auditory_3_class",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.7910/DVN/MQOVEY",
        description="Auditory stream segregation BCI dataset - 3-class ASME-BCI",
        investigators=["S. Kojima", "S. Kanoh"],
        institution="Shibaura Institute of Technology",
        country="JP",
        repository="Harvard Dataverse",
        data_url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MQOVEY",
        license="CC0",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

KOJIMA2024B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=66,
        channel_types={"eeg": 64, "eog": 2},
        hardware="BrainAmp (Brain Products, Germany)",
        sensor_type="Ag/AgCl",
        reference="right mastoid",
        ground="left mastoid",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=15,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="auditory_4_class",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.7910/DVN/1UJDV6",
        description="Auditory stream segregation BCI dataset - 4-class & 2-class ASME-BCI",
        investigators=["S. Kojima", "S. Kanoh"],
        institution="Shibaura Institute of Technology",
        country="JP",
        repository="Harvard Dataverse",
        data_url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1UJDV6",
        license="CC0",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

ROMANIBF2025ERP_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=8,
        channel_types={"eeg": 8},
        sensors=["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
        hardware="g.tec Unicorn with conductive gel",
        reference="right mastoid",
        ground="left mastoid",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=22,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="brainform_erp",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.48550/arXiv.2510.10169",
        description="BrainForm ERP dataset - serious game-based BCI training",
        investigators=["M. Romani", "D. Zanoni", "E. Farella", "L. Turchet"],
        institution="University of Trento",
        country="IT",
        repository="Zenodo",
        data_url="https://zenodo.org/records/17225966",
        license="CC BY 4.0",
        publication_year=2025,
    ),
    sessions_per_subject=18,
    runs_per_session=1,
)

SOSULSKI2019_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=32,
        channel_types={"eeg": 31, "eog": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=13,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="auditory_oddball_soa",
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6094/UNIFR/154576",
        description="Auditory Oddball with SOA variations",
        investigators=["Sosulski, J.", "Tangermann, M."],
        institution="University of Freiburg",
        country="DE",
        repository="FreiDok",
        data_url="https://freidok.uni-freiburg.de/data/154576",
        publication_year=2019,
        license="CC BY 4.0",
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

# =============================================================================
# SSVEP Datasets
# =============================================================================

LEE2019_SSVEP_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=62,
        channel_types={"eeg": 62},
        sensor_type="Ag/AgCl wet",
        hardware="BrainAmp (Brain Products)",
        reference="nasion",
        ground="AFz",
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=54,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="4_frequency",
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/giz002",
        description="OpenBMI SSVEP dataset - investigation into BCI illiteracy",
        investigators=["M.H. Lee", "S.W. Lee"],
        institution="Korea University",
        country="KR",
        repository="GigaDB",
        data_url="https://gigadb.org/dataset/100542",
        license="CC0",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

KALUNGA2016_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=9,
        channel_types={"eeg": 8, "stim": 1},
        sensors=["Oz", "O1", "O2", "POz", "PO3", "PO4", "PO7", "PO8"],
        hardware="g.Mobilab+",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
        age_mean=24.0,
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="4_frequency_exoskeleton",
        n_classes=4,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neucom.2016.01.007",
        description="SSVEP Exoskeleton dataset using Riemannian geometry",
        investigators=["E.K. Kalunga", "S. Chevallier", "Q. Barthelemy"],
        institution="University of Versailles",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2392979",
        license="CC BY 4.0",
        publication_year=2016,
    ),
    sessions_per_subject=1,
    runs_per_session=2,
)

NAKANISHI2015_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=8,
        channel_types={"eeg": 8},
        sensors=["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
        montage="standard_1005",
        line_freq=60.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=9,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="12_frequency_jfpm",
        n_classes=12,
        trial_duration=4.15,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0140703",
        description="12-class JFPM SSVEP dataset comparing CCA-based methods",
        investigators=["M. Nakanishi", "Y. Wang", "Y.T. Wang", "T.P. Jung"],
        institution="UCSD / Swartz Center for Computational Neuroscience",
        country="US",
        repository="GitHub",
        data_url="https://github.com/mnakanishi/12JFPM_SSVEP",
        publication_year=2015,
        license="Public Domain",
    ),
    sessions_per_subject=1,
    runs_per_session=15,
)

WANG2016_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=64,
        channel_types={"eeg": 64},
        hardware="Synamps2 (Neuroscan)",
        reference="vertex",
        ground="between Fz and FPz",
        filters="0.15-200 Hz bandpass, 50 Hz notch",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=34,
        health_status="healthy",
        age_mean=22.0,
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="40_frequency_benchmark",
        n_classes=40,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2016.2627556",
        description="SSVEP Benchmark Dataset - 40 frequencies JFPM coding",
        investigators=["Y. Wang", "X. Chen", "X. Gao", "S. Gao"],
        institution="Tsinghua University",
        country="CN",
        repository="Tsinghua BCI Lab",
        data_url="http://bci.med.tsinghua.edu.cn/download.html",
        publication_year=2016,
        license="Public Domain",
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

MAMEM1_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=257,
        channel_types={"eeg": 256, "stim": 1},
        hardware="EGI 300 Geodesic EEG System (GES 300)",
        sensor_type="HydroCel Geodesic Sensor Net",
        montage="GSN-HydroCel-256",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=11,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="5_frequency_isolated",
        n_classes=5,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6084/m9.figshare.2068677.v5",
        description="MAMEM SSVEP Dataset I - 5 frequencies presented in isolation",
        investigators=["Nikolopoulos, S.", "Lazarou, I.", "Kompatsiaris, I."],
        institution="CERTH-ITI",
        country="GR",
        repository="Figshare/Zenodo",
        data_url="https://zenodo.org/records/1295936",
        license="CC BY 4.0",
        publication_year=2016,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

MAMEM2_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=257,
        channel_types={"eeg": 256, "stim": 1},
        hardware="EGI 300 Geodesic EEG System (GES 300)",
        sensor_type="HydroCel Geodesic Sensor Net",
        montage="GSN-HydroCel-256",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=11,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="5_frequency_simultaneous",
        n_classes=5,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6084/m9.figshare.3153409",
        description="MAMEM SSVEP Dataset II - 5 frequencies presented simultaneously",
        investigators=["Nikolopoulos, S.", "Lazarou, I.", "Kompatsiaris, I."],
        institution="CERTH-ITI",
        country="GR",
        repository="Figshare",
        data_url="https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_II/3153409",
        license="CC BY 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

MAMEM3_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=128.0,
        n_channels=15,
        channel_types={"eeg": 14, "stim": 1},
        hardware="Emotiv EPOC",
        montage="standard_1020",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=11,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="5_frequency_consumer_grade",
        n_classes=5,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6084/m9.figshare.3413851",
        description="MAMEM SSVEP Dataset III - consumer-grade EEG (Emotiv EPOC)",
        investigators=["Nikolopoulos, S.", "Lazarou, I.", "Kompatsiaris, I."],
        institution="CERTH-ITI",
        country="GR",
        repository="Figshare",
        data_url="https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_III/3413851",
        license="CC BY 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

# =============================================================================
# cVEP Datasets
# =============================================================================

THIELEN2015_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=2048.0,
        n_channels=67,
        channel_types={"eeg": 64, "stim": 3},
        hardware="Biosemi ActiveTwo",
        montage="standard_1010",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="36_class_gold_codes",
        n_classes=36,
        trial_duration=4.2,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41598-017-15373-x",
        description="cVEP dataset with Gold codes and spatiotemporal beamformer",
        investigators=["Wittevrongel, B.", "Van Wolputte, E.", "Van Hulle, M.M."],
        institution="KU Leuven",
        country="BE",
        repository="Radboud Data Repository",
        data_url="https://public.data.ru.nl/dcc/DSC_2018.00047_553_v3",
        license="CC BY 4.0",
        publication_year=2017,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

THIELEN2021_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=11,
        channel_types={"eeg": 8, "stim": 3},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=30,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="20_class_noise_tagging",
        n_classes=20,
        trial_duration=31.5,
    ),
    documentation=DocumentationMetadata(
        doi="10.34973/1ecz-1232",
        description="cVEP noise-tagging BCI dataset from zero training study",
        investigators=["J. Thielen", "P. Desain"],
        institution="Radboud University, Donders Institute",
        country="NL",
        repository="Donders Repository",
        data_url="https://public.data.ru.nl/dcc/DSC_2018.00122_448_v3",
        license="CC BY 4.0",
        publication_year=2018,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_BURSTVEP40_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=34,
        channel_types={"eeg": 32, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="burst_vep_40hz",
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="Burst VEP at 40 Hz refresh rate",
        investigators=[
            "Cabrera Castillos, K.",
            "Ladouce, S.",
            "Darmet, L.",
            "Dehais, F.",
        ],
        institution="ISAE-SUPAERO",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/records/8255618",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_BURSTVEP100_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=34,
        channel_types={"eeg": 32, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="burst_vep_100hz",
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="Burst VEP at 100 Hz refresh rate",
        investigators=[
            "Cabrera Castillos, K.",
            "Ladouce, S.",
            "Darmet, L.",
            "Dehais, F.",
        ],
        institution="ISAE-SUPAERO",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/records/8255618",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_CVEP40_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=34,
        channel_types={"eeg": 32, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="cvep_40hz",
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="cVEP at 40 Hz refresh rate",
        investigators=[
            "Cabrera Castillos, K.",
            "Ladouce, S.",
            "Darmet, L.",
            "Dehais, F.",
        ],
        institution="ISAE-SUPAERO",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/records/8255618",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_CVEP100_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=34,
        channel_types={"eeg": 32, "stim": 2},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="cvep_100hz",
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="cVEP at 100 Hz refresh rate",
        investigators=[
            "Cabrera Castillos, K.",
            "Ladouce, S.",
            "Darmet, L.",
            "Dehais, F.",
        ],
        institution="ISAE-SUPAERO",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/records/8255618",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

MARTINEZCAGIGAL2023_CHECKER_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=16,
        channel_types={"eeg": 16},
        sensors=[
            "Oz",
            "F3",
            "Fz",
            "F4",
            "I1",
            "I2",
            "C3",
            "Cz",
            "C4",
            "CPz",
            "P3",
            "Pz",
            "P4",
            "PO7",
            "POz",
            "PO8",
        ],
        hardware="g.USBamp (g.Tec, Guger Technologies, Austria)",
        reference="earlobe",
        ground="AFz",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=16,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="checkerboard_cvep",
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.71569/7c67-v596",
        description="Checkerboard c-VEP BCI dataset with varying spatial frequencies",
        investigators=[
            "V. Martínez-Cagigal",
            "Á. Fernández-Rodríguez",
            "E. Santamaría-Vázquez",
            "R. Ron-Angevin",
            "R. Hornero",
        ],
        institution="University of Valladolid",
        country="ES",
        repository="UVaDoc",
        data_url="https://uvadoc.uva.es/handle/10324/70973",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=8,
    runs_per_session=1,
)

MARTINEZCAGIGAL2023_PARY_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=15,
        channel_types={"eeg": 15},
        sensors=[
            "F3",
            "Fz",
            "F4",
            "C3",
            "Cz",
            "C4",
            "CPz",
            "P3",
            "Pz",
            "P4",
            "PO7",
            "PO8",
            "Oz",
            "I1",
            "I2",
        ],
        hardware="g.USBamp (g.Tec, Guger Technologies, Austria)",
        reference="earlobe",
        ground="AFz",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=16,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="cvep",
        task_type="p_ary_cvep",
        n_classes=11,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.71569/025s-eq10",
        description="P-ary m-sequence c-VEP dataset with non-binary encoding",
        investigators=[
            "V. Martínez-Cagigal",
            "E. Santamaría-Vázquez",
            "S. Pérez-Velasco",
            "D. Marcos-Martínez",
            "S. Moreno-Calderón",
            "R. Hornero",
        ],
        institution="University of Valladolid",
        country="ES",
        repository="UVaDoc",
        data_url="https://uvadoc.uva.es/handle/10324/70945",
        license="CC BY 4.0",
        publication_year=2023,
    ),
    sessions_per_subject=5,
    runs_per_session=1,
)

# =============================================================================
# Resting State Datasets
# =============================================================================

CATTAN2019_PHMD_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=17,
        channel_types={"eeg": 16, "stim": 1},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="rstate",
        task_type="eyes_open_closed",
        n_classes=2,
        trial_duration=60.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.2617084",
        description="Passive Head-Mounted Display resting state EEG dataset",
        investigators=["G. Cattan", "P.L.C. Rodrigues", "M. Congedo"],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2617085",
        license="CC BY 4.0",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

HINSS2021_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=62,
        channel_types={"eeg": 62},
        sensor_type="active Ag-AgCl electrodes",
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=15,
        health_status="healthy",
        gender={"female": 6, "male": 9},
        age_mean=25.0,
    ),
    experiment=ExperimentMetadata(
        paradigm="rstate",
        task_type="cognitive_workload",
        n_classes=4,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.5055046",
        description="Neuroergonomic dataset for mental state monitoring",
        investigators=["M. Hinss", "B. Somon", "F. Dehais", "R.N. Roy"],
        institution="ISAE-SUPAERO",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/records/5055046",
        license="CC BY 4.0",
        publication_year=2021,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

RODRIGUES2017_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=16,
        channel_types={"eeg": 16},
        sensors=[
            "Fp1",
            "Fp2",
            "Fc5",
            "Fz",
            "Fc6",
            "T7",
            "Cz",
            "T8",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "O1",
            "Oz",
            "O2",
        ],
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=20,
        health_status="healthy",
        gender={"female": 7, "male": 13},
        age_mean=25.8,
        age_std=5.27,
        age_min=19,
        age_max=44,
    ),
    experiment=ExperimentMetadata(
        paradigm="rstate",
        task_type="eyes_open_closed",
        n_classes=2,
        trial_duration=10.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.2348892",
        description="Alpha Waves - Resting state EEG dataset for BCI",
        investigators=["Cattan, G.", "Rodrigues, P.L.C.", "Congedo, M."],
        institution="GIPSA-lab",
        country="FR",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2348892",
        license="CC BY 4.0",
        publication_year=2018,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)


# =============================================================================
# Metadata Catalog Dictionary
# =============================================================================

DATASET_METADATA_CATALOG = {
    # Motor Imagery
    "AlexMI": ALEXMI_METADATA,
    "BNCI2014_001": BNCI2014_001_METADATA,
    "BNCI2014_002": BNCI2014_002_METADATA,
    "BNCI2014_004": BNCI2014_004_METADATA,
    "BNCI2015_001": BNCI2015_001_METADATA,
    "BNCI2015_004": BNCI2015_004_METADATA,
    "BNCI2003_004": BNCI2003_004_METADATA,
    "BNCI2019_001": BNCI2019_001_METADATA,
    "BNCI2020_001": BNCI2020_001_METADATA,
    "BNCI2024_001": BNCI2024_001_METADATA,
    "BNCI2022_001": BNCI2022_001_METADATA,
    "BNCI2025_001": BNCI2025_001_METADATA,
    "BNCI2025_002": BNCI2025_002_METADATA,
    "Beetl2021_A": BEETL2021_A_METADATA,
    "Beetl2021_B": BEETL2021_B_METADATA,
    "Cho2017": CHO2017_METADATA,
    "Dreyer2023": DREYER2023_METADATA,
    "Dreyer2023A": DREYER2023A_METADATA,
    "Dreyer2023B": DREYER2023B_METADATA,
    "Dreyer2023C": DREYER2023C_METADATA,
    "GrosseWentrup2009": GROSSEWENTRUP2009_METADATA,
    "Lee2019_MI": LEE2019_MI_METADATA,
    "Liu2024": LIU2024_METADATA,
    "Ofner2017": OFNER2017_METADATA,
    "PhysionetMI": PHYSIONETMI_METADATA,
    "Schirrmeister2017": SCHIRRMEISTER2017_METADATA,
    "Shin2017A": SHIN2017A_METADATA,
    "Shin2017B": SHIN2017B_METADATA,
    "Stieger2021": STIEGER2021_METADATA,
    "Weibo2014": WEIBO2014_METADATA,
    "Zhou2016": ZHOU2016_METADATA,
    # P300/ERP
    "BI2012": BI2012_METADATA,
    "BI2013a": BI2013A_METADATA,
    "BI2014a": BI2014A_METADATA,
    "BI2014b": BI2014B_METADATA,
    "BI2015a": BI2015A_METADATA,
    "BI2015b": BI2015B_METADATA,
    "BNCI2014_008": BNCI2014_008_METADATA,
    "BNCI2014_009": BNCI2014_009_METADATA,
    "BNCI2015_003": BNCI2015_003_METADATA,
    "BNCI2015_006": BNCI2015_006_METADATA,
    "BNCI2015_007": BNCI2015_007_METADATA,
    "BNCI2015_008": BNCI2015_008_METADATA,
    "BNCI2015_009": BNCI2015_009_METADATA,
    "BNCI2015_010": BNCI2015_010_METADATA,
    "BNCI2015_012": BNCI2015_012_METADATA,
    "BNCI2015_013": BNCI2015_013_METADATA,
    "BNCI2016_002": BNCI2016_002_METADATA,
    "BNCI2020_002": BNCI2020_002_METADATA,
    "Cattan2019_VR": CATTAN2019_VR_METADATA,
    "DemonsP300": DEMONSP300_METADATA,
    "EPFLP300": EPFLP300_METADATA,
    "ErpCore2021_ERN": ERPCORE2021_ERN_METADATA,
    "ErpCore2021_LRP": ERPCORE2021_LRP_METADATA,
    "ErpCore2021_MMN": ERPCORE2021_MMN_METADATA,
    "ErpCore2021_N170": ERPCORE2021_N170_METADATA,
    "ErpCore2021_N2pc": ERPCORE2021_N2PC_METADATA,
    "ErpCore2021_N400": ERPCORE2021_N400_METADATA,
    "ErpCore2021_P3": ERPCORE2021_P3_METADATA,
    "Huebner2017": HUEBNER2017_METADATA,
    "Huebner2018": HUEBNER2018_METADATA,
    "Kojima2024A": KOJIMA2024A_METADATA,
    "Kojima2024B": KOJIMA2024B_METADATA,
    "Lee2019_ERP": LEE2019_ERP_METADATA,
    "RomaniBF2025ERP": ROMANIBF2025ERP_METADATA,
    "Sosulski2019": SOSULSKI2019_METADATA,
    # SSVEP
    "Kalunga2016": KALUNGA2016_METADATA,
    "Lee2019_SSVEP": LEE2019_SSVEP_METADATA,
    "MAMEM1": MAMEM1_METADATA,
    "MAMEM2": MAMEM2_METADATA,
    "MAMEM3": MAMEM3_METADATA,
    "Nakanishi2015": NAKANISHI2015_METADATA,
    "Wang2016": WANG2016_METADATA,
    # cVEP
    "CastillosBurstVEP40": CASTILLOS_BURSTVEP40_METADATA,
    "CastillosBurstVEP100": CASTILLOS_BURSTVEP100_METADATA,
    "CastillosCVEP40": CASTILLOS_CVEP40_METADATA,
    "CastillosCVEP100": CASTILLOS_CVEP100_METADATA,
    "MartinezCagigal2023Checker": MARTINEZCAGIGAL2023_CHECKER_METADATA,
    "MartinezCagigal2023Pary": MARTINEZCAGIGAL2023_PARY_METADATA,
    "Thielen2015": THIELEN2015_METADATA,
    "Thielen2021": THIELEN2021_METADATA,
    # Resting State
    "Cattan2019_PHMD": CATTAN2019_PHMD_METADATA,
    "Hinss2021": HINSS2021_METADATA,
    "Rodrigues2017": RODRIGUES2017_METADATA,
}


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Get metadata for a specific dataset by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "BNCI2014_001", "Lee2019_MI").

    Returns
    -------
    DatasetMetadata
        Metadata object for the specified dataset.

    Raises
    ------
    KeyError
        If the dataset name is not found in the catalog.
    """
    if dataset_name not in DATASET_METADATA_CATALOG:
        available = ", ".join(sorted(DATASET_METADATA_CATALOG.keys()))
        raise KeyError(
            f"Dataset '{dataset_name}' not found in catalog. "
            f"Available datasets: {available}"
        )
    return DATASET_METADATA_CATALOG[dataset_name]
