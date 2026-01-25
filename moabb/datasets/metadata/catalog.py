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
        n_channels=16,
        channel_types={"eeg": 16},
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
        events={"right_hand": 2, "feet": 3, "rest": 4},
        n_classes=3,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        description="Motor imagery dataset from A. Barachant PhD dissertation",
        investigators=["Alexandre Barachant"],
        institution="University of Grenoble",
        country="France",
        repository="Zenodo",
        data_url="https://zenodo.org/record/806023",
        publication_year=2012,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2014_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=22,
        channel_types={"eeg": 22},
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
        events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
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
        country="Austria",
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
        n_channels=15,
        channel_types={"eeg": 15},
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
        events={"right_hand": 1, "feet": 2},
        n_classes=2,
        trials_per_class={"right_hand": 80, "feet": 80},
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2012.00055",
        description="Motor imagery dataset - right hand and feet kinesthetic MI",
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
        n_channels=3,
        channel_types={"eeg": 3},
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=4.5,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2012.00055",
        description="BCI Competition IV Dataset 2b - 2-class motor imagery with EOG artifacts",
        investigators=["R. Leeb", "C. Brunner", "G.R. Müller-Putz", "G. Pfurtscheller"],
        institution="Graz University of Technology",
        country="Austria",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/004-2014/",
        license="CC BY 4.0",
        publication_year=2008,
    ),
    sessions_per_subject=5,
    runs_per_session=1,
)

BNCI2015_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=13,
        channel_types={"eeg": 13},
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
        events={"right_hand": 1, "feet": 2},
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        description="BNCI 2015-001 Motor imagery dataset",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/001-2015/",
        license="CC BY 4.0",
        publication_year=2010,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BNCI2015_004_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={
            "right_hand": 1,
            "feet": 2,
            "navigation": 3,
            "subtraction": 4,
            "word_ass": 5,
        },
        n_classes=5,
        trial_duration=7.0,
    ),
    documentation=DocumentationMetadata(
        description="5-class mental tasks for users with disability",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/004-2015/",
        license="CC BY 4.0",
        publication_year=2013,
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
        events={"right_hand": 0, "feet": 1},
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
        country="Germany",
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
        n_channels=64,
        channel_types={"eeg": 64},
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trials_per_class={"left_hand": 100, "right_hand": 100},
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/gix034",
        description="Motor imagery BCI EEG dataset with EMG from GigaScience",
        investigators=["H. Cho", "M. Ahn", "S. Ahn", "M. Kwon", "S.C. Jun"],
        institution="Gwangju Institute of Science and Technology",
        country="South Korea",
        repository="GigaDB",
        data_url="http://dx.doi.org/10.5524/100295",
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
        events={"left_hand": 2, "right_hand": 1},
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
        country="South Korea",
        repository="GigaDB",
        data_url="http://deepbci.korea.ac.kr/opensource/opendb/",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

PHYSIONETMI_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=160.0,
        n_channels=64,
        channel_types={"eeg": 64},
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
        events={"left_hand": 2, "right_hand": 3, "hands": 4, "feet": 5, "rest": 1},
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
        country="USA",
        repository="PhysioNet",
        data_url="https://physionet.org/content/eegmmidb/1.0.0/",
        license="Open Data Commons Attribution License v1.0",
        publication_year=2004,
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
        events={"right_hand": 1, "left_hand": 2, "rest": 3, "feet": 4},
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
        country="Germany",
        repository="GIN",
        data_url="https://gin.g-node.org/robintibor/high-gamma-dataset",
        publication_year=2017,
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=7.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TBME.2008.2009768",
        description="Beamforming in Noninvasive Brain-Computer Interfaces",
        investigators=["M. Grosse-Wentrup", "C. Liefhold", "K. Gramann", "M. Buss"],
        institution="Max Planck Institute for Biological Cybernetics",
        country="Germany",
        repository="Zenodo",
        data_url="https://zenodo.org/records/1217449",
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=10.0,
    ),
    documentation=DocumentationMetadata(
        description="EEG/fNIRS hybrid motor imagery dataset - Modality A",
        country="South Korea",
        publication_year=2017,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

WEIBO2014_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=200.0,
        n_channels=60,
        channel_types={"eeg": 60},
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
        events={
            "left_hand": 1,
            "right_hand": 2,
            "left_foot": 3,
            "right_foot": 4,
            "rest": 5,
            "left_hand_right_foot": 6,
            "right_hand_left_foot": 7,
        },
        n_classes=7,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0114853",
        description="Simple and compound limb motor imagery EEG patterns",
        investigators=["W. Yi", "S. Qiu", "H. Qi", "L. Zhang", "B. Wan", "D. Ming"],
        country="China",
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
        events={"left_hand": 1, "right_hand": 2, "feet": 3},
        n_classes=3,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        description="Motor imagery dataset - 3 class",
        country="China",
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
        events={"left_hand": 1, "right_hand": 2, "tongue": 3, "feet": 4},
        n_classes=4,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-021-00883-1",
        description="Large-scale continuous sensorimotor rhythm BCI learning dataset",
        investigators=["J.R. Stieger", "S.A. Engel", "B. He"],
        institution="Carnegie Mellon University",
        country="USA",
        repository="Figshare",
        data_url="https://doi.org/10.6084/m9.figshare.13123148",
        publication_year=2021,
    ),
    sessions_per_subject=11,
    runs_per_session=1,
)

LIU2024_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=29,
        channel_types={"eeg": 29},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=50,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="left_right_hand",
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        description="Motor imagery dataset",
        country="China",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

DREYER2023_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=27,
        channel_types={"eeg": 27},
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
        events={"left_hand": 1, "right_hand": 2},
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
        country="France",
        repository="Zenodo",
        data_url="https://doi.org/10.5281/zenodo.8089820",
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
        events={
            "elbow_flexion": 1,
            "elbow_extension": 2,
            "forearm_supination": 3,
            "forearm_pronation": 4,
            "hand_close": 5,
            "hand_open": 6,
            "rest": 7,
        },
        n_classes=7,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0182578",
        description="Upper limb movements decoded from low-frequency EEG time-domain",
        investigators=["P. Ofner", "A. Schwarz", "J. Pereira", "G.R. Müller-Putz"],
        institution="Graz University of Technology, BCI-Lab",
        country="Austria",
        repository="BNCI Horizon 2020",
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
        events={
            "supination": 776,
            "pronation": 777,
            "hand_open": 779,
            "palmar_grasp": 925,
            "lateral_grasp": 926,
        },
        n_classes=5,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41598-019-43594-9",
        description="Motor imagery of grasp movements in spinal cord injury patients",
        investigators=["A. Schwarz", "P. Ofner", "J. Pereira", "G.R. Müller-Putz"],
        institution="Graz University of Technology",
        country="Austria",
        repository="BNCI Horizon 2020",
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
        n_subjects=15,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="reach_and_grasp",
        events={
            "palmar_grasp": 503587,
            "lateral_grasp": 503588,
            "rest": 768,
        },
        n_classes=3,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2020.00849",
        description="Reach-and-grasp electrode comparison (gel, water, dry)",
        investigators=["A. Schwarz", "J. Pereira", "G.R. Müller-Putz"],
        institution="Graz University of Technology",
        country="Austria",
        repository="BNCI Horizon 2020",
        publication_year=2020,
    ),
    sessions_per_subject=1,
    runs_per_session=4,
)

BNCI2024_001_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=64,
        channel_types={"eeg": 60, "eog": 4},
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
        events={
            "letter_a": 1,
            "letter_d": 2,
            "letter_e": 3,
            "letter_f": 4,
            "letter_j": 5,
            "letter_n": 6,
            "letter_o": 7,
            "letter_s": 8,
            "letter_t": 9,
            "letter_v": 10,
        },
        n_classes=10,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.compbiomed.2024.109132",
        description="Handwritten character classification from EEG",
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
        events={
            "trajectory_start": 1,
            "waypoint_miss": 16,
            "waypoint_hit": 48,
            "trajectory_end": 255,
        },
        n_classes=4,
        trial_duration=90.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/THMS.2020.3038339",
        description="EEG correlates of difficulty level in simulated drone piloting",
        investigators=["P.-K. Jao", "R. Chavarriaga", "J. d. R. Millán"],
        institution="EPFL",
        country="Switzerland",
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
        n_channels=64,
        channel_types={"eeg": 60, "eog": 4},
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
        events={
            "up_slow_near": 1,
            "up_slow_far": 2,
            "up_fast_near": 3,
            "up_fast_far": 4,
            "down_slow_near": 5,
            "down_slow_far": 6,
            "down_fast_near": 7,
            "down_fast_far": 8,
            "left_slow_near": 9,
            "left_slow_far": 10,
            "left_fast_near": 11,
            "left_fast_far": 12,
            "right_slow_near": 13,
            "right_slow_far": 14,
            "right_fast_near": 15,
            "right_fast_far": 16,
        },
        n_classes=16,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2552/ada0ea",
        description="Simultaneous encoding of speed, distance, and direction in discrete reaching",
        investigators=["N. Srisrisawang", "G. R. Müller-Putz"],
        institution="Graz University of Technology",
        country="Austria",
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
        n_subjects=20,
        health_status="healthy",
        gender={"male": 10, "female": 10},
        age_mean=24.0,
        age_std=5.0,
        handedness={"right": 20},
    ),
    experiment=ExperimentMetadata(
        paradigm="imagery",
        task_type="continuous_2d_trajectory_decoding",
        events={
            "snakerun": 1,
            "freerun": 2,
            "eyerun": 3,
        },
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
        country="Austria",
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
        events={"subtraction": 3, "rest": 4},
        n_classes=2,
        trial_duration=10.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2016.2628057",
        description="EEG/fNIRS hybrid mental arithmetic dataset - Modality B",
        institution="Korea Advanced Institute of Science and Technology",
        country="South Korea",
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
        events={"rest": 0, "left_hand": 1, "right_hand": 2, "feet": 3},
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        description="BEETL Challenge Dataset A - Motor Imagery",
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
        events={"left_hand": 0, "right_hand": 1, "feet": 2, "rest": 3},
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        description="BEETL Challenge Dataset B - Motor Imagery",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

DREYER2023A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=27,
        channel_types={"eeg": 27},
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Dreyer2023 Dataset A - Large MI dataset user profile study",
        investigators=["P. Dreyer", "A. Roc", "L. Pillette", "S. Rimbert", "F. Lotte"],
        institution="Inria Bordeaux",
        country="France",
        repository="Zenodo",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

DREYER2023B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=27,
        channel_types={"eeg": 27},
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Dreyer2023 Dataset B - Large MI dataset user profile study",
        investigators=["P. Dreyer", "A. Roc", "L. Pillette", "S. Rimbert", "F. Lotte"],
        institution="Inria Bordeaux",
        country="France",
        repository="Zenodo",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

DREYER2023C_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=27,
        channel_types={"eeg": 27},
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
        events={"left_hand": 1, "right_hand": 2},
        n_classes=2,
        trial_duration=5.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41597-023-02445-z",
        description="Dreyer2023 Dataset C - Large MI dataset user profile study",
        investigators=["P. Dreyer", "A. Roc", "L. Pillette", "S. Rimbert", "F. Lotte"],
        institution="Inria Bordeaux",
        country="France",
        repository="Zenodo",
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
        n_channels=8,
        channel_types={"eeg": 8},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=8,
        health_status="patients",
        clinical_population="ALS",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="speller",
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        description="P300 speller for ALS patients",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/008-2014/",
        license="CC BY 4.0",
        publication_year=2012,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BNCI2014_009_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=16,
        channel_types={"eeg": 16},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        description="P300 speller dataset",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/009-2014/",
        license="CC BY 4.0",
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BNCI2015_003_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=8,
        channel_types={"eeg": 8},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        description="BNCI 2015-003 P300 dataset",
        repository="BNCI Horizon 2020",
        data_url="http://bnci-horizon-2020.eu/database/data-sets/003-2015/",
        license="CC BY 4.0",
    ),
    sessions_per_subject=1,
    runs_per_session=2,
)

BNCI2015_006_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=100.0,
        n_channels=64,
        channel_types={"eeg": 64},
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
        events={"attended_deviant": 1, "unattended_deviant": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/11/2/026009",
        description="Music BCI - Auditory attention detection with ERP",
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
        n_channels=63,
        channel_types={"eeg": 63},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=16,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="motion_vep_speller",
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/9/4/045006",
        description="Motion VEP Speller - mVEP based BCI speller",
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
        n_channels=63,
        channel_types={"eeg": 63},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/8/6/066003",
        description="Center Speller P300 - gaze-independent BCI speller",
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
        sampling_rate=1000.0,
        n_channels=62,
        channel_types={"eeg": 60, "eog": 2},
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
        events={"NonTarget": 1, "Target": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0009813",
        description="AMUSE - Auditory Multi-class Spatial ERP BCI",
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
        sampling_rate=1000.0,
        n_channels=63,
        channel_types={"eeg": 63},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=0.8,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.clinph.2012.12.050",
        description="RSVP - Rapid Serial Visual Presentation BCI",
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
        sampling_rate=1000.0,
        n_channels=63,
        channel_types={"eeg": 63},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=12,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="pass2d_auditory_speller",
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2011.00099",
        description="PASS2D - Auditory Spatial Speller BCI",
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
        n_channels=64,
        channel_types={"eeg": 64},
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
        events={"correct": 1, "error": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/TNSRE.2010.2053387",
        description="Error-Related Potentials (ErrP) for BCI error detection",
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
        events={"car_brake": 1, "react_emg": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1088/1741-2560/8/5/056001",
        description="Emergency braking during simulated driving - ERP detection",
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
        n_channels=31,
        channel_types={"eeg": 29, "eog": 2},
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
        events={"NonTarget": 1, "Target": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.3389/fnins.2020.591777",
        description="Attention Shift - Covert Spatial Attention BCI",
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
        events={"Target": 2, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.2649006",
        description="Brain Invaders P300 dataset 2012 - experimental validation",
        investigators=["G. Van Veen", "A. Barachant", "M. Congedo"],
        institution="GIPSA-lab, University of Grenoble Alpes",
        country="France",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2649069",
        publication_year=2012,
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
        events={"Target": 33285, "NonTarget": 33286},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.1494163",
        description="Brain Invaders adaptive vs non-adaptive P300 BCI",
        investigators=["E. Vaineau", "A. Barachant", "M. Congedo"],
        institution="GIPSA-lab, CNRS, Grenoble-INP",
        country="France",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2669187",
        publication_year=2013,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2014A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=16,
        channel_types={"eeg": 16},
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
        events={"Target": 2, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3266222",
        description="Brain Invaders calibration-less P300 BCI with dry electrodes",
        investigators=["L. Korczowski", "E. Ostaschenko", "M. Congedo"],
        institution="GIPSA-lab",
        country="France",
        repository="Zenodo",
        publication_year=2014,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2014B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"Target": 2, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3267301",
        description="Brain Invaders Solo vs Collaboration multi-user P300 BCI",
        investigators=["L. Korczowski", "E. Ostaschenko", "M. Congedo"],
        institution="GIPSA-lab",
        country="France",
        repository="Zenodo",
        publication_year=2014,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

BI2015A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"Target": 2, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3266929",
        description="Brain Invaders with modulation of flash duration",
        investigators=["L. Korczowski", "M. Cederhout", "M. Congedo"],
        institution="GIPSA-lab",
        country="France",
        repository="Zenodo",
        publication_year=2015,
    ),
    sessions_per_subject=3,
    runs_per_session=1,
)

BI2015B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"Target": 2, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.3267307",
        description="Brain Invaders Cooperative vs Competitive multi-user P300",
        investigators=["L. Korczowski", "M. Cederhout", "M. Congedo"],
        institution="GIPSA-lab",
        country="France",
        repository="Zenodo",
        publication_year=2015,
    ),
    sessions_per_subject=1,
    runs_per_session=4,
)

CATTAN2019_VR_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=16,
        channel_types={"eeg": 16},
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
        events={"Target": 2, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.5281/zenodo.2605204",
        description="EEG-based BCI in Virtual Reality using P300",
        investigators=["G. Cattan", "A. Andreev", "P.L.C. Rodrigues", "M. Congedo"],
        institution="GIPSA-lab",
        country="France",
        repository="Zenodo",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=5,
)

EPFLP300_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=2048.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        description="EPFL P300 BCI dataset for disabled users",
        institution="EPFL",
        country="Switzerland",
        repository="BNCI Horizon 2020",
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/giz002",
        description="OpenBMI ERP dataset - investigation into BCI illiteracy",
        investigators=["M.H. Lee", "S.W. Lee"],
        institution="Korea University",
        country="South Korea",
        repository="GigaDB",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=2,
)

DEMONSP300_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=8,
        channel_types={"eeg": 8},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=60,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="speller",
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        description="Demons P300 dataset",
        repository="Neiry",
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ERPCORE2021_N170_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1024.0,
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
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
        country="USA",
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
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE MMN - Mismatch Negativity (passive auditory oddball)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="USA",
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
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE P3 - Active visual oddball paradigm",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="USA",
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
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE N400 - Semantic processing (word pair judgement)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="USA",
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
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE ERN - Error-related negativity (Flankers task)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="USA",
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
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE LRP - Lateralized Readiness Potential",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="USA",
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
        n_channels=30,
        channel_types={"eeg": 30},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2020.117465",
        description="ERP CORE N2pc - Attention-related N2pc (visual search)",
        investigators=["E.S. Kappenman", "S.J. Luck"],
        institution="University of California Davis",
        country="USA",
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
        events={"Target": 10002, "NonTarget": 10001},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0175856",
        description="Visual Matrix Speller LLP dataset 2017",
        investigators=["D. Huebner", "T. Verhoeven", "K.R. Müller", "P.J. Kindermans"],
        country="Germany",
        publication_year=2017,
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
        events={"Target": 10002, "NonTarget": 10001},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1109/MCI.2018.2807039",
        description="Visual Matrix Speller LLP dataset 2018",
        investigators=["D. Huebner", "T. Verhoeven", "K.R. Müller", "P.J. Kindermans"],
        country="Germany",
        publication_year=2018,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

KOJIMA2024A_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=64,
        channel_types={"eeg": 64},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=11,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="auditory_3_class",
        events={"Target": 1, "NonTarget": 0},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.7910/DVN/MQOVEY",
        description="Auditory BCI dataset A - 3-class auditory attention",
        repository="Harvard Dataverse",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

KOJIMA2024B_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=1000.0,
        n_channels=64,
        channel_types={"eeg": 64},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=15,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="p300",
        task_type="auditory_4_class",
        events={"Target": 1, "NonTarget": 0},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.7910/DVN/1UJDV6",
        description="Auditory BCI dataset B - 4-class auditory attention",
        repository="Harvard Dataverse",
        publication_year=2024,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

ROMANIBF2025ERP_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=8,
        channel_types={"eeg": 8},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.48550/arXiv.2510.10169",
        description="BrainForm ERP dataset",
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
        events={"Target": 21, "NonTarget": 1},
        n_classes=2,
        trial_duration=1.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6094/UNIFR/154576",
        description="Auditory Oddball with SOA variations",
        institution="University of Freiburg",
        country="Germany",
        publication_year=2019,
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
        events={"12.0": 1, "8.57": 2, "6.67": 3, "5.45": 4},
        n_classes=4,
        trial_duration=4.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1093/gigascience/giz002",
        description="OpenBMI SSVEP dataset - investigation into BCI illiteracy",
        investigators=["M.H. Lee", "S.W. Lee"],
        institution="Korea University",
        country="South Korea",
        repository="GigaDB",
        publication_year=2019,
    ),
    sessions_per_subject=2,
    runs_per_session=1,
)

KALUNGA2016_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=256.0,
        n_channels=8,
        channel_types={"eeg": 8},
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
        events={"13": 2, "17": 4, "21": 3, "rest": 1},
        n_classes=4,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neucom.2016.01.007",
        description="SSVEP Exoskeleton dataset using Riemannian geometry",
        investigators=["E.K. Kalunga", "S. Chevallier", "Q. Barthelemy"],
        institution="University of Versailles",
        country="France",
        repository="Zenodo",
        data_url="https://zenodo.org/record/2392979",
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
        n_subjects=10,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="ssvep",
        task_type="12_frequency_jfpm",
        events={
            "9.25": 1,
            "11.25": 2,
            "13.25": 3,
            "9.75": 4,
            "11.75": 5,
            "13.75": 6,
            "10.25": 7,
            "12.25": 8,
            "14.25": 9,
            "10.75": 10,
            "12.75": 11,
            "14.75": 12,
        },
        n_classes=12,
        trial_duration=4.15,
    ),
    documentation=DocumentationMetadata(
        doi="10.1371/journal.pone.0140703",
        description="12-class JFPM SSVEP dataset comparing CCA-based methods",
        investigators=["M. Nakanishi", "Y. Wang", "Y.T. Wang", "T.P. Jung"],
        institution="UCSD / Swartz Center for Computational Neuroscience",
        country="USA",
        repository="GitHub",
        data_url="https://github.com/mnakanishi/12JFPM_SSVEP",
        publication_year=2015,
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
        n_subjects=35,
        health_status="healthy",
        gender={"female": 17, "male": 18},
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
        country="China",
        repository="Tsinghua BCI Lab",
        data_url="http://bci.med.tsinghua.edu.cn/download.html",
        publication_year=2016,
    ),
    sessions_per_subject=1,
    runs_per_session=6,
)

MAMEM1_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=256,
        channel_types={"eeg": 256},
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
        events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
        n_classes=5,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6084/m9.figshare.2068677.v5",
        description="MAMEM SSVEP Dataset I - 5 frequencies presented in isolation",
        institution="MAMEM Project (EU H2020)",
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
        n_channels=256,
        channel_types={"eeg": 256},
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
        events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
        n_classes=5,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6084/m9.figshare.3153409",
        description="MAMEM SSVEP Dataset II - 5 frequencies presented simultaneously",
        institution="MAMEM Project (EU H2020)",
        repository="Figshare",
        data_url="https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_II/3153409",
        license="CC BY 4.0",
        publication_year=2016,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

MAMEM3_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=128.0,
        n_channels=14,
        channel_types={"eeg": 14},
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
        events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
        n_classes=5,
        trial_duration=3.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.6084/m9.figshare.3413851",
        description="MAMEM SSVEP Dataset III - consumer-grade EEG (Emotiv EPOC)",
        institution="MAMEM Project (EU H2020)",
        repository="Figshare",
        data_url="https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_III/3413851",
        license="CC BY 4.0",
        publication_year=2016,
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
        n_channels=64,
        channel_types={"eeg": 64},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=36,
        trial_duration=4.2,
    ),
    documentation=DocumentationMetadata(
        doi="10.1038/s41598-017-15373-x",
        description="cVEP dataset with Gold codes and spatiotemporal beamformer",
        investigators=["J. Thielen", "P. Desain", "J. Farquhar"],
        institution="Radboud University, Donders Institute",
        country="Netherlands",
        publication_year=2015,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

THIELEN2021_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=8,
        channel_types={"eeg": 8},
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
        events={"Target": 1, "NonTarget": 2},
        n_classes=20,
        trial_duration=31.5,
    ),
    documentation=DocumentationMetadata(
        doi="10.34973/1ecz-1232",
        description="cVEP noise-tagging BCI dataset from zero training study",
        investigators=["J. Thielen", "P. Desain"],
        institution="Radboud University, Donders Institute",
        country="Netherlands",
        repository="Donders Repository",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_BURSTVEP40_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"0": 100, "1": 101},
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="Burst VEP at 40 Hz refresh rate",
        investigators=["V. Martínez-Cagigal", "J. Thielen", "E. Santamaría-Vázquez"],
        country="Spain",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_BURSTVEP100_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"0": 100, "1": 101},
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="Burst VEP at 100 Hz refresh rate",
        investigators=["V. Martínez-Cagigal", "J. Thielen", "E. Santamaría-Vázquez"],
        country="Spain",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_CVEP40_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"0": 100, "1": 101},
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="cVEP at 40 Hz refresh rate",
        investigators=["V. Martínez-Cagigal", "J. Thielen", "E. Santamaría-Vázquez"],
        country="Spain",
        publication_year=2023,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

CASTILLOS_CVEP100_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=500.0,
        n_channels=32,
        channel_types={"eeg": 32},
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
        events={"0": 100, "1": 101},
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.1016/j.neuroimage.2023.120446",
        description="cVEP at 100 Hz refresh rate",
        investigators=["V. Martínez-Cagigal", "J. Thielen", "E. Santamaría-Vázquez"],
        country="Spain",
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
        events={"0": 100, "1": 101},
        n_classes=2,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        doi="10.71569/7c67-v596",
        description="Checkerboard c-VEP dataset",
        investigators=["V. Martínez-Cagigal", "E. Santamaría-Vázquez"],
        institution="University of Valladolid",
        country="Spain",
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
        description="P-ary c-VEP dataset with multiple base conditions",
        investigators=["V. Martínez-Cagigal", "E. Santamaría-Vázquez"],
        institution="University of Valladolid",
        country="Spain",
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
        n_channels=16,
        channel_types={"eeg": 16},
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
        events={"eyes_open": 1, "eyes_closed": 2},
        n_classes=2,
        trial_duration=60.0,
    ),
    documentation=DocumentationMetadata(
        description="Passive Head-Mounted Display resting state dataset",
        investigators=["G. Cattan"],
        institution="GIPSA-lab",
        country="France",
        publication_year=2019,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

HINSS2021_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=250.0,
        n_channels=62,
        channel_types={"eeg": 62},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=15,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="rstate",
        task_type="4_class_mental_state",
        n_classes=4,
        trial_duration=2.0,
    ),
    documentation=DocumentationMetadata(
        description="Mental state monitoring dataset",
        publication_year=2021,
    ),
    sessions_per_subject=1,
    runs_per_session=1,
)

RODRIGUES2017_METADATA = DatasetMetadata(
    acquisition=AcquisitionMetadata(
        sampling_rate=512.0,
        n_channels=16,
        channel_types={"eeg": 16},
        montage="standard_1005",
        line_freq=50.0,
    ),
    participants=ParticipantMetadata(
        n_subjects=20,
        health_status="healthy",
    ),
    experiment=ExperimentMetadata(
        paradigm="rstate",
        task_type="eyes_open_closed",
        events={"eyes_open": 1, "eyes_closed": 2},
        n_classes=2,
        trial_duration=10.0,
    ),
    documentation=DocumentationMetadata(
        description="Resting state EEG dataset",
        publication_year=2017,
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
