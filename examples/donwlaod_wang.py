"""Download Wang2026 archives concurrently, then convert them to BIDS.

The four source archives require roughly 65 GB in total. BIDS conversion needs
additional disk space and is kept sequential to limit peak memory use.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from moabb.datasets import (
    Wang2026BCI2000Control,
    Wang2026EEGNetControl,
    Wang2026JointLearning,
    Wang2026TactileControl,
)


bids_output_root = Path.home() / "Desktop" / "datasets_to_jz"

dataset_classes = [
    Wang2026JointLearning,
    Wang2026TactileControl,
    Wang2026EEGNetControl,
    Wang2026BCI2000Control,
]


def download_archive(dataset_class):
    """Cache one complete group archive by requesting its first subject."""
    dataset = dataset_class()
    subject = dataset.subject_list[0]
    paths = dataset.data_path(
        subject=subject,
        force_update=False,
        verbose=False,
    )
    return dataset_class.__name__, subject, len(paths)


def download_archives_in_parallel():
    """Download the four independent Wang2026 archives concurrently."""
    errors = []

    with ThreadPoolExecutor(max_workers=len(dataset_classes)) as executor:
        futures = {
            executor.submit(download_archive, dataset_class): dataset_class
            for dataset_class in dataset_classes
        }

        for future in as_completed(futures):
            dataset_class = futures[future]
            try:
                name, subject, n_files = future.result()
                print(
                    f"Downloaded {name}; extracted subject {subject} "
                    f"with {n_files} MAT files."
                )
            except Exception as error:
                errors.append((dataset_class.__name__, error))
                print(f"Failed to download {dataset_class.__name__}: {error}")

    if errors:
        failed = ", ".join(name for name, _ in errors)
        raise RuntimeError(f"Failed Wang2026 downloads: {failed}") from errors[0][1]


def convert_datasets_to_bids():
    """Convert sequentially to limit peak memory use."""
    bids_output_root.mkdir(parents=True, exist_ok=True)

    for dataset_class in dataset_classes:
        print(f"Converting {dataset_class.__name__} to BIDS...")
        dataset = dataset_class()
        bids_root = dataset.convert_to_bids(path=bids_output_root, overwrite=True)
        print(f"Finished {dataset_class.__name__}: {bids_root}")


if __name__ == "__main__":
    download_archives_in_parallel()
    convert_datasets_to_bids()
