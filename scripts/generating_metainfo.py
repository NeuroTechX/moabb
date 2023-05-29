import os.path
from argparse import ArgumentParser
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import moabb
from moabb.datasets.utils import dataset_search
from moabb.utils import set_download_dir


def parser_init():
    parser = ArgumentParser(description="Getting the meta-information script for MOABB")

    parser.add_argument(
        "-mne_p",
        "--mne_data",
        dest="mne_data",
        default="/mnt/beegfs/projects/moabb/mne_data/",
        type=str,
        help="Folder where to save and load the datasets with mne structure.",
    )

    return parser


def process_trial_freq(trials_per_events, parad_name):
    """
    Function to process the trial frequency.
    Getting the median value if the paradigm is MotorImagery.
    Parameters
    ----------
    trials_per_events: dict
    parad_name: str

    Returns
    -------

    """
    class_per_trial = list(trials_per_events.values())

    if parad_name == "imagery" or parad_name == "ssvep":
        return f"{int(np.median(class_per_trial))}"
    elif parad_name == "p300":
        not_target = max(trials_per_events.values())
        target = min(trials_per_events.values())
        return f"NT{not_target} / T {target}"


def get_meta_info(dataset, dataset_name, parad_obj, parad_name):
    """
    Function to get the meta-information of a dataset.
    Parameters
    ----------
    dataset: moabb.datasets.base.BaseDataset
    dataset_name: str
    parad_obj: moabb.paradigms.base.BaseParadigm

    Returns
    -------

    """
    subjects = len(dataset.subject_list)
    session = dataset.n_sessions

    X, y, metadata = parad_obj.get_data(dataset, [1], return_epochs=True)

    sfreq = int(X.info["sfreq"])
    nchan = X.info["nchan"]
    runs = len(metadata["run"].unique())
    classes = len(X.event_id)
    epoch_size = X.tmax - X.tmin

    trials_per_events = mne.count_events(X.events)
    total_trials = int(sum(trials_per_events.values()))
    trial_class = process_trial_freq(trials_per_events, parad_name)

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
        index=[
            "Dataset",
            "#Subj",
            "#Chan",
            "#Classes",
            "trials/events",
            "Window Size",
            "Freq",
            "#Session",
            "#Runs",
            "Total_trials",
        ],
    )

    return info_dataset


if __name__ == "__main__":
    mne.set_log_level(False)

    parser = parser_init()
    options = parser.parse_args()
    mne_path = Path(options.mne_data)

    set_download_dir(mne_path)

    paradigms = {}
    paradigms.update({"imagery": moabb.paradigms.MotorImagery()})
    paradigms.update({"ssvep": moabb.paradigms.SSVEP()})
    paradigms.update({"p300": moabb.paradigms.P300()})

    for parad_name, parad_obj in paradigms.items():
        dataset_list = dataset_search(paradigm=parad_name)

        metainfo = []
        for dataset in dataset_list:
            dataset_name = str(dataset).split(".")[-1].split(" ")[0]

            dataset_path = f"{mne_path.parent}/metainfo/metainfo_{dataset_name}.csv"

            if not os.path.exists(dataset_path):
                print(
                    f"Trying to get the meta information from the "
                    f"dataset {dataset} with {parad_name}"
                )

                try:
                    info_dataset = get_meta_info(
                        dataset, dataset_name, parad_obj, parad_name
                    )
                    print(
                        "Saving the meta information for the dataset in the file: ",
                        dataset_path,
                    )
                    info_dataset.to_csv(dataset_path)
                    metainfo.append(info_dataset)

                except Exception as ex:
                    print(f"Error with {dataset} with {parad_name} paradigm", end=" ")
                    print(f"Error: {ex}")

                    if parad_name == "imagery":
                        print("Trying with the LeftRightImagery paradigm")
                        parad_obj_2 = moabb.paradigms.LeftRightImagery()
                        try:
                            info_dataset = get_meta_info(
                                dataset, dataset_name, parad_obj_2, parad_name
                            )
                            print(
                                "Saving the meta information for the dataset in the file: ",
                                dataset_path,
                            )
                            info_dataset.to_csv(dataset_path)
                            metainfo.append(info_dataset)

                        except Exception as ex:
                            print(
                                f"Error with {dataset} with {parad_name} paradigm",
                                end=" ",
                            )
                            print(f"Error: {ex}")
            else:
                print(f"Loading the meta information from {dataset_path}")
                info_dataset = pd.read_csv(dataset_path)
                metainfo.append(info_dataset)

        paradigm_df = pd.concat(metainfo, axis=1).T

        paradigm_df.columns = [
            "Dataset",
            "#Subj",
            "#Chan",
            "#Classes",
            "#Trials_per_subject",
            "trials_ids",
            "Window Size (s)",
            "Freq (Hz)",
            "#Session",
            "#Runs",
            "Total_trials",
        ]

        paradigm_path = f"{mne_path.parent}/metainfo/metainfo_{parad_name}.csv"
        print(f"Saving the meta information for the paradigm {paradigm_path}")

        paradigm_df.to_csv(paradigm_path, index=None)
