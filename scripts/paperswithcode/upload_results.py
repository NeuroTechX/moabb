import pickle
from argparse import ArgumentParser
from dataclasses import dataclass
from math import isnan

import pandas as pd
from paperswithcode import PapersWithCodeClient
from paperswithcode.models import (
    EvaluationTableCreateRequest,
    EvaluationTableSyncRequest,
    MetricSyncRequest,
    ResultSyncRequest,
)


@dataclass
class Task:
    id: str
    name: str
    description: str
    area: str
    parent_task: str


_metrics = {
    "time": "training time (s)",
    "carbon_emission": "CO2 Emission (g)",
}


def make_table(results_csv, metric):
    df = pd.read_csv(results_csv)
    df = (
        df.groupby(
            ["dataset", "paradigm", "evaluation", "pipeline"],
        )[["score", "time", "carbon_emission"]]
        .mean()
        .reset_index()
    )
    df.score = df.score * 100
    columns = dict(**_metrics, score=metric)
    df.rename(columns=columns, inplace=True)
    df.paradigm = df.paradigm.replace(
        {"FilterBankMotorImagery": "MotorImagery", "LeftRightImagery": "MotorImagery"}
    )
    print(df.head())
    return df


def upload_subtable(client, df, dataset, task, paper, evaluated_on):
    kwargs = dict(
        task=task.id,
        dataset=dataset.id,
        description=task.description,
        mirror_url="http://moabb.neurotechx.com/docs/benchmark_summary.html",
    )
    client.evaluation_create(EvaluationTableCreateRequest(**kwargs))

    r = EvaluationTableSyncRequest(
        **kwargs,
        metric=[
            MetricSyncRequest(name=metric, is_loss=metric in _metrics.values())
            for metric in df.columns
        ],
        results=[
            ResultSyncRequest(
                metrics={k: str(v) for k, v in row.to_dict().items() if not isnan(v)},
                paper=paper,
                methodology=pipeline,
                external_id=f"{dataset.id}-{task.id}-{pipeline}",
                evaluated_on=evaluated_on,
                # external_source_url="http://moabb.neurotechx.com/docs/benchmark_summary.html",
                # TODO: maybe update url with the exact row of the result
            )
            for pipeline, row in df.iterrows()
        ],
    )
    leaderboard_id = client.evaluation_synchronize(r)
    print(f"{leaderboard_id=}")
    return leaderboard_id


def upload_table(client, df, datasets, tasks, paper, evaluated_on, subsubtask):
    gp_cols = ["dataset", "paradigm", "evaluation"]
    df_gp = df.groupby(gp_cols)
    ids = []
    for (dataset_name, paradigm_name, evaluation_name), sub_df in df_gp:
        dataset = datasets[dataset_name]
        task_key = (paradigm_name, evaluation_name)
        if subsubtask is not None:
            task_key += (subsubtask,)
        task = tasks[task_key]
        id = upload_subtable(
            client,
            sub_df.set_index("pipeline").drop(columns=gp_cols),
            dataset,
            task,
            paper,
            evaluated_on,
        )
        ids.append(id)
    return ids


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("token", type=str, help="PapersWithCode API token")
    parser.add_argument("results_csv", type=str, help="CSV file with results to upload")
    parser.add_argument(
        "metric",
        type=str,
        help="Metric used in the results CSV (see PapersWithCode metrics)",
    )
    parser.add_argument(
        "-s",
        "--subsubtask",
        type=str,
        default=None,
        help="If relevant, the type of motor imagery task (see create_datasets_and_tasks.py)",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        help="Pickle file created by create_datasets_and_tasks.py",
        default="paperswithcode_datasets_and_tasks.pickle",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Pickle output file",
        default="paperswithcode_results.pickle",
    )
    parser.add_argument("-p", "--paper", type=str, help="Paper URL", default="")
    parser.add_argument(
        "-e",
        "--evaluated_on",
        type=str,
        help="Results date YYYY-MM-DD",
        default="2024-04-09",
    )
    args = parser.parse_args()

    with open(args.datasets, "rb") as f:
        datasets = pickle.load(f)
    summary_table = make_table(args.results_csv, metric=args.metric)

    client = PapersWithCodeClient(token=args.token)

    upload_table(
        client,
        summary_table,
        datasets["datasets"],
        datasets["tasks"],
        args.paper,
        args.evaluated_on,
        args.subsubtask,
    )
