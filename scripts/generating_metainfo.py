from argparse import ArgumentParser
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import moabb
from moabb.datasets.utils import dataset_search
from moabb.utils import set_download_dir


columns_name = [
    "Dataset",
    "#Subj",
    "#Chan",
    "#Classes",
    "trials/events",
    "Window Size (s)",
    "Freq (Hz)",
    "#Session",
    "#Runs",
    "Total_trials",
]


def parser_init():
    parser = ArgumentParser(description="Getting the meta-information script for MOABB")

    parser.add_argument(
        "-mne_p",
        "--mne_data",
        dest="mne_data",
        default=Path.home() / "mne_data",
        type=Path,
        help="Folder where to save and load the datasets with mne structure.",
    )

    return parser


def process_trial_freq(trials_per_events, prdgm):
    """Function to process the trial frequency. Getting the median value if the
    paradigm is MotorImagery.

    Parameters
    ----------
    trials_per_events: dict
    prdgm: str

    Returns
    -------
    trial_freq: str
    """
    class_per_trial = list(trials_per_events.values())

    if prdgm == "imagery" or prdgm == "ssvep":
        return f"{int(np.median(class_per_trial))}"
    elif prdgm == "p300":
        not_target = max(trials_per_events.values())
        target = min(trials_per_events.values())
        return f"NT{not_target} / T {target}"


def get_meta_info(dataset, dataset_name, paradigm, prdgm_name):
    """Function to get the meta-information of a dataset.

    Parameters
    ----------
    dataset: BaseDataset
        Dataset object
    dataset_name: str
        Dataset name
    paradigm: BaseParadigm
         Paradigm object to process the dataset
    prdgm_name: str
        Paradigm name

    Returns
    -------
    """
    subjects = len(dataset.subject_list)
    session = dataset.n_sessions

    X, _, metadata = paradigm.get_data(dataset, [1], return_epochs=True)

    sfreq = int(X.info["sfreq"])
    nchan = X.info["nchan"]
    runs = len(metadata["run"].unique())
    classes = len(X.event_id)
    epoch_size = X.tmax - X.tmin

    trials_per_events = mne.count_events(X.events)
    total_trials = int(sum(trials_per_events.values()))
    trial_class = process_trial_freq(trials_per_events, prdgm_name)

    info_dataset = pd.Series(
        [
            dataset_name,
            subjects,
            nchan,
            classes,
            trial_class,
            epoch_size,
            sfreq,
            session,
            runs,
            session * runs * total_trials * subjects,
        ],
        index=columns_name,
    )

    return info_dataset


if __name__ == "__main__":
    mne.set_log_level(False)

    parser = parser_init()
    options = parser.parse_args()
    mne_path = Path(options.mne_data)

    set_download_dir(mne_path)

    paradigms = {}
    paradigms["imagery"] = moabb.paradigms.MotorImagery()
    paradigms["ssvep"] = moabb.paradigms.SSVEP()
    paradigms["p300"] = moabb.paradigms.P300()

    for prdgm_name, paradigm in paradigms.items():
        dataset_list = dataset_search(paradigm=prdgm_name)

        metainfo = []
        for dataset in dataset_list:
            dataset_name = str(dataset).split(".")[-1].split(" ")[0]

            dataset_path = f"{mne_path.parent}/metainfo/metainfo_{dataset_name}.csv"

            if not dataset_path.exists():
                print(
                    "Trying to get the meta information from the "
                    f"dataset {dataset} with {prdgm_name}"
                )

                try:
                    info_dataset = get_meta_info(
                        dataset, dataset_name, paradigm, prdgm_name
                    )
                    print(
                        "Saving the meta information for the dataset in the file: ",
                        dataset_path,
                    )
                    info_dataset.to_csv(dataset_path)
                    metainfo.append(info_dataset)

                except Exception as ex:
                    print(f"Error with {dataset} with {prdgm_name} paradigm", end=" ")
                    print(f"Error: {ex}")

                    if prdgm_name == "imagery":
                        print("Trying with the LeftRightImagery paradigm")
                        prdgm2 = moabb.paradigms.LeftRightImagery()
                        try:
                            info_dataset = get_meta_info(
                                dataset, dataset_name, prdgm2, prdgm_name
                            )
                            print(
                                "Saving the meta information for the dataset in the file: ",
                                dataset_path,
                            )
                            info_dataset.to_csv(dataset_path)
                            metainfo.append(info_dataset)

                        except Exception as ex:
                            print(
                                f"Error with {dataset} with {prdgm_name} paradigm",
                                end=" ",
                            )
                            print(f"Error: {ex}")
            else:
                print(f"Loading the meta information from {dataset_path}")
                info_dataset = pd.read_csv(dataset_path)
                metainfo.append(info_dataset)

        paradigm_df = pd.concat(metainfo, axis=1).T

        paradigm_df.columns = columns_name
        dataset_path = mne_path.parent / "metainfo" / f"metainfo_{dataset_name}.csv"
        print(f"Saving the meta information for the paradigm {dataset_path}")

        paradigm_df.to_csv(dataset_path, index=None)
