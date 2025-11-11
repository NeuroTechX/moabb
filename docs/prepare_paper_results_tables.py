from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


pipelines_files = {
    "SSVEP_CCA": "CCA-SSVEP.yml",
    "SSVEP_MsetCCA": "MsetCCA-SSVEP.yml",
    "SSVEP_MDM": "MDM-SSVEP.yml",
    "SSVEP_TS+LR": "TSLR-SSVEP.yml",
    "SSVEP_TS+SVM": "TSSVM_grid.yml",
    "SSVEP_TRCA": "TRCA-SSVEP.yml",
    "XDAWN+LDA": "xDAWN_LDA.yml",
    "XDAWNCov+MDM": "XdawnCov_MDM.yml",
    "XDAWNCov+TS+SVM": "XdawnCov_TS_SVM.yml",
    "ERPCov+MDM": "ERPCov_MDM.yml",
    "ERPCov(svd_n=4)+MDM": "ERPCov_MDM.yml",
    "ACM+TS+SVM": "AUG_TANG_SVM_grid.yml",
    "CSP+LDA": "CSP.yml",
    "CSP+SVM": "CSP_SVM_grid.yml",
    "DLCSPauto+shLDA": "regCSP%2BshLDA.yml",
    "DeepConvNet": "Keras_DeepConvNet.yml",
    "EEGITNet": "Keras_EEGITNet.yml",
    "EEGNeX": "Keras_EEGNeX.yml",
    "EEGNet_8_2": "Keras_EEGNet_8_2.yml",
    "EEGTCNet": "Keras_EEGITNet.yml",
    "FilterBank+SVM": "FBCSP.py",
    "FgMDM": "FgMDM.yml",
    "LogVariance+LDA": "LogVar_grid.yml",
    "LogVariance+SVM": "LogVar_grid.yml#L7",
    "MDM": "MDM.yml",
    "ShallowConvNet": "Keras_ShallowConvNet.yml",
    "TRCSP+LDA": "WTRCSP.py",
    "TS+EL": "EN_grid.yml",
    "TS+LR": "TSLR.yml",
    "TS+SVM": "TSSVM_grid.yml",
}


def wrap_pipeline(name: str) -> str:
    name = name.split("`")[1]
    file_name = pipelines_files.get(name)
    if file_name is not None:
        # Required due to Keras removal from MOABB; prevents broken links.
        branch = "develop" if "Keras" not in file_name else "v1.1.2"
        url = f"https://github.com/NeuroTechX/moabb/blob/{branch}/pipelines/{file_name}"
        return f'<a href="{url}">{name}</a>'
    return name


def wrap_dataset(name: str) -> str:
    if name == "Pipeline":
        return name

    name = name.split("`")[1]
    url = f'<a class="reference internal" href="generated/moabb.datasets.{name}.html#moabb.datasets.{name}" title="moabb.datasets.{name}"><code class="xref py py-class docutils literal notranslate"><span class="pre">{name}</span></code></a>'
    return url


def main(source_dir: str, target_dir: str) -> None:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    for file in Path(source_dir).glob("*.csv"):
        target_file = target_path / file.name
        print(f"Processing {file} -> {target_file}")
        df = pd.read_csv(file, index_col=False, header=0, skipinitialspace=True)
        df.columns = df.columns.map(wrap_dataset)
        df["Pipeline"] = df["Pipeline"].apply(wrap_pipeline)
        html_table = df.to_html(
            index=False,
            classes=["moabb-table", "sortable", "hover", "row-border", "order-column"],
            escape=False,
            table_id=file.stem,
        )
        with open(f"{target_path}/{file.stem}.html", "w", encoding="utf-8") as f:
            f.write(html_table)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_dir", type=str)
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()
    main(args.source_dir, args.target_dir)
    print(args.target_dir)
