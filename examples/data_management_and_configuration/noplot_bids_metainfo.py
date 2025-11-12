"""
====================================
Tutorial 6: Setting BIDS Metainfo
====================================
"""

# Authors: Inga Schöyen, Loekie Slütter, Femke van Venrooij
#
# In this tutorial, we will show how to use the bids_metainfo function
# to set the metadata for a BIDS dataset using json
# The function is shown below (commented out) and takes a BIDS path as input
#  and returns the metainfo as a json dictionary
#
#
#

import tempfile

import Path

from moabb.datasets import AlexMI
from moabb.datasets.utils import bids_metainfo


##############################################################################
# Outline of the Tutorial
# ------------------------------------------------------------------
# 1. Create a BIDS dataset
# 2. Save the metadata dictionary as a json file
# 3. Load the metadata dictionary from the json file
# 4. Advanced usage of the metadata dictionary

# ##############################################################################
#  Below is the function bids_metainfo as defined in moabb.datasets.utils
#
#
# def bids_metainfo(bids_path: Path) -> dict:
#     """Create metadata for the BIDS dataset.

#     To allow lazy loading of the metadata, we store the metadata in a JSON file
#     in the root of the BIDS dataset.

#     Parameters
#     ----------
#     bids_path : Path
#         The path to the BIDS dataset.
#     """
#     json_data = {}

#     paths = mne_bids.find_matching_paths(
#         root=bids_path,
#         datatypes="eeg",
#     )

#     for path in paths:
#         uid = path.fpath.name
#         json_data[uid] = path.entities
#         json_data[uid]["fpath"] = str(path.fpath)

#     return json_data

# as we can see, the function takes a BIDS path as input, collects all matching datasets
# and fills a dictionary with the metadata gathered from each matching dataset

# to use the function, we first need to create a BIDS dataset, either by sourcing it
# from a local path or by using the moabb.datasets module to fetch a dataset and convert
# it to BIDS using the get_data() method


##############################################################################
# 1. Fetch a moabb dataset


dataset = AlexMI()

##############################################################################
# 2. Tranform the dataset to BIDS format, using a temporary directory

temp_dir = Path(tempfile.mkdtemp())

# by using a temporary directory, and setting save_raw to False, we can avoid
# saving the raw data to and consequently overcluttering the disk

# if you want to save the raw data, set save_raw to True
# and set the path to a directory of your choice

_ = dataset.get_data(cache_config = dict(path=temp_dir, save_raw=False))


##############################################################################
# 3. Get the metadata using the bids_metainfo function

metadata = bids_metainfo(temp_dir)

##############################################################################
#  4. Print the metadata dictionary
print(metadata)

# ##############################################################################
# Output:

# sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '1', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '1', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '1', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '1', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '1', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-1/ses-0/eeg/sub-1_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '2', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-2/ses-0/eeg/sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '2', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-2/ses-0/eeg/sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '2', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-2/ses-0/eeg/sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '2', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-2/ses-0/eeg/sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '2', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-2/ses-0/eeg/sub-2_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '3', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-3/ses-0/eeg/sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '3', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-3/ses-0/eeg/sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '3', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-3/ses-0/eeg/sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '3', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-3/ses-0/eeg/sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '3', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-3/ses-0/eeg/sub-3_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '4', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-4/ses-0/eeg/sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '4', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-4/ses-0/eeg/sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '4', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-4/ses-0/eeg/sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '4', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-4/ses-0/eeg/sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '4', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-4/ses-0/eeg/sub-4_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '5', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-5/ses-0/eeg/sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '5', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-5/ses-0/eeg/sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '5', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-5/ses-0/eeg/sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '5', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-5/ses-0/eeg/sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '5', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-5/ses-0/eeg/sub-5_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '6', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-6/ses-0/eeg/sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '6', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-6/ses-0/eeg/sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '6', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-6/ses-0/eeg/sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '6', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-6/ses-0/eeg/sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '6', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-6/ses-0/eeg/sub-6_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '7', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-7/ses-0/eeg/sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '7', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-7/ses-0/eeg/sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '7', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-7/ses-0/eeg/sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '7', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-7/ses-0/eeg/sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '7', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-7/ses-0/eeg/sub-7_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}
# sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv: {'subject': '8', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-8/ses-0/eeg/sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_channels.tsv'}
# sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf: {'subject': '8', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-8/ses-0/eeg/sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.edf'}
# sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json: {'subject': '8', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-8/ses-0/eeg/sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_eeg.json'}
# sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json: {'subject': '8', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-8/ses-0/eeg/sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.json'}
# sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv: {'subject': '8', 'session': '0', 'task': 'imagery', 'acquisition': None, 'run': '0', 'processing': None, 'space': None, 'recording': None, 'split': None, 'description': '2f4a2e6207d30d406bb04b5b1aae5195', 'fpath': '/var/folders/d0/y1vcxz_x22g764clc80c8bmr0000gn/T/tmpxvydmbsb/sub-8/ses-0/eeg/sub-8_ses-0_task-imagery_run-0_desc-2f4a2e6207d30d406bb04b5b1aae5195_events.tsv'}

# as we can see, the metadata dictionary contains the metadata for each dataset
# in the BIDS dataset, including the path to the raw data file
# since we did not specify the samplifng configuration when calling the dataset (line 73),
# the configuration is set to None
