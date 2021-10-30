"""
===========================
Change Download Directory
===========================

This is a minimal Example to show how to change the default data download directory to a custom
path/location.
"""
# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

import os.path as osp

from mne import get_config

from moabb.utils import set_download_dir


# You can choose to change the download directory to any path of your choice.
# If the path/folder doesn't exist, it will be created for you.

original_path = get_config("MNE_DATA")
print(f"The download directory is currently {original_path}")
new_path = osp.join(osp.expanduser("~"), "mne_data_test")
set_download_dir(new_path)

# To see if the mne config has been changed correctly
check_path = get_config("MNE_DATA")
print(f"Now the download directory has been changed to {check_path}")

# Set back to usual
set_download_dir(original_path)
