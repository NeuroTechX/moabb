import glob
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


# Classes that are not exported at the top-level ``moabb.datasets`` namespace
# and therefore need a fully-qualified module path for Sphinx cross-references.
_FULL_PATH_CLASSES = {
    "ErpCore2021": "moabb.datasets.erpcore2021.ErpCore2021",
}


def prepare_table(df: pd.DataFrame):
    def _class_ref(name):
        full = _FULL_PATH_CLASSES.get(name)
        if full:
            return f":class:`~{full}`"
        return f":class:`{name}`"

    df["Dataset"] = df["Dataset"].apply(_class_ref)


def main(source_dir: str, target_dir: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(Path(source_dir) / "*.csv"))
    for f in files:
        target_file = target_dir / Path(f).name
        print(f"Processing {f} -> {target_file}")
        df = pd.read_csv(f, index_col=False, header=0, skipinitialspace=True)
        prepare_table(df)
        df.to_csv(target_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()
    main(args.source_dir, args.target_dir)
    print(args.target_dir)
