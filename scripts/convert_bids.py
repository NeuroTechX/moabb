import os.path as op
from argparse import ArgumentParser
from pathlib import Path

import mne
from mne_bids import BIDSPath, get_anonymization_daysback, write_raw_bids

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


if __name__ == "__main__":
    mne.set_log_level(False)

    parser = parser_init()
    options = parser.parse_args()
    mne_path = Path(options.mne_data)

    set_download_dir(mne_path)

    paradigms = {}
    paradigms.update({"imagery": moabb.paradigms.MotorImagery(fmin=0.1, fmax=None)})
    # paradigms.update({"ssvep": moabb.paradigms.SSVEP()})
    # paradigms.update({"p300": moabb.paradigms.P300()})

    for parad_name, parad_obj in paradigms.items():
        dataset_list = dataset_search(paradigm=parad_name)

        metainfo = []
        for dataset in dataset_list[:1]:
            epoch_list = list()
            raw_list = list()

            bids_list = list()
            dataset_name = str(dataset).split(".")[-1].split(" ")[0]
            for subject_id in dataset.subject_list:
                print(f"Processing {dataset_name}-{subject_id}")
                epoch, y, metadata = parad_obj.get_data(
                    dataset, [subject_id], return_epochs=True
                )
                raw, _, _ = parad_obj.get_data(
                    dataset, [subject_id], return_epochs=False, return_raws=True
                )

                epoch[0].info[
                    "line_freq"
                ] = 50  # specify power line frequency as required by BIDS
                epoch[0].info["subject_info"] = {
                    "his_id": subject_id
                }  # specify subject info as required by BIDS
                epoch[0].info["device_info"] = {
                    "type": "eeg"
                }  # specify device info as required by BIDS

                raw[0].info[
                    "line_freq"
                ] = 50  # specify power line frequency as required by BIDS
                raw[0].info["subject_info"] = {
                    "his_id": subject_id
                }  # specify subject info as required by BIDS
                raw[0].info["device_info"] = {
                    "type": "eeg"
                }  # specify device info as required by BIDS
                raw[0].set_meas_date(None)
                epoch_list.append(epoch[0])
                raw_list.append(raw[0])

                bids_root = op.join(mne_path, f"MNE-{dataset_name.upper()}-BIDS")

                bids_path = BIDSPath(
                    subject=f"{subject_id:03}", task="MotorImagery", root=bids_root
                )

                bids_list.append(bids_path)

            daysback_min, daysback_max = get_anonymization_daysback(epoch_list)

            for raw, epoch, bids_path in zip(raw_list, epoch_list, bids_list):
                # By using the same anonymization `daysback` number we can
                # preserve the longitudinal structure of multiple sessions for a
                # single subject and the relation between subjects. Be sure to
                # change or delete this number before putting code online, you
                # wouldn't want to inadvertently de-anonymize your data.
                #
                # Note that we do not need to pass any events, as the dataset is already
                # equipped with annotations, which will be converted to BIDS events
                # automatically.
                write_raw_bids(
                    raw,
                    bids_path,
                    anonymize=dict(daysback=daysback_min + 2117),
                    overwrite=True,
                    allow_preload=True,
                    format="EDF",
                    events=epoch.events,
                    event_id=epoch.event_id,
                )
