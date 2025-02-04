from moabb.paradigms import MotorImagery
from moabb.datasets import utils

paradigm = MotorImagery()
compatible_datasets = utils.dataset_search(paradigm=paradigm)
print("\nCompatible datasets for Motor Imagery paradigm:")
for dataset in compatible_datasets:
    print(f"- {dataset.code}: {dataset.n_classes} classes")