import pooch
import os
from pooch import Unzip

FILES = []
FILES.append("https://dataverse.harvard.edu/api/access/datafile/2499178")

base_path = os.path.join(os.path.expanduser("~"), "mne_data", "WEIBO", "MNE-weibo-2014")
if not os.path.isdir(base_path):
    os.makedirs(base_path)

print(f"Downloading to {base_path}")
pooch.retrieve(
    FILES[0],
    None,
    "data0.zip",
    base_path,
    processor=Unzip(),
    progressbar=True,
)
print("Download finished")
