import pickle
import re
from argparse import ArgumentParser

from paperswithcode import PapersWithCodeClient
from paperswithcode.models import DatasetCreateRequest, TaskCreateRequest


def dataset_name(dataset):
    return f"{dataset.__name__} MOABB"


def dataset_full_name(dataset):
    s = dataset.__doc__.split("\n\n")[0]
    s = re.sub(r" \[\d+\]_", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def dataset_url(dataset):
    return f"http://moabb.neurotechx.com/docs/generated/moabb.datasets.{dataset.__name__}.html"


def valid_datasets():
    from moabb.datasets.utils import dataset_list
    from moabb.utils import aliases_list

    deprecated_names = [n[0] for n in aliases_list]
    return [
        d()
        for d in dataset_list
        if (d.__name__ not in deprecated_names) and ("Fake" not in d.__name__)
    ]


_paradigms = {
    "MotorImagery": (
        "Motor Imagery",
        ["all classes", "left hand vs. right hand", "right hand vs. feet"],
        "Motor Imagery",
    ),
    "P300": ("ERP", None, "Event-Related Potential (ERP)"),
    "SSVEP": ("SSVEP", None, "Steady-State Visually Evoked Potential (SSVEP)"),
    "CVEP": ("c-VEP", None, "Code-Modulated Visual Evoked Potential (c-VEP)"),
}
_evaluations = {
    "WithinSession": "Within-Session",
    "CrossSession": "Cross-Session",
    "CrossSubject": "Cross-Subject",
}


def create_tasks(client: PapersWithCodeClient):
    tasks = {}
    for paradigm_class, (
        paradigm_name,
        subparadigms,
        paradigm_fullname,
    ) in _paradigms.items():
        description = f"Classification of examples recorded under the {paradigm_fullname} paradigm, as part of Brain-Computer Interfaces (BCI)."
        task = client.task_add(
            TaskCreateRequest(
                name=paradigm_name,
                description=description,
                area="medical",
                parent_task="brain-computer-interface",
            )
        )
        tasks[paradigm_class] = task
        for evaluation_class, evaluation in _evaluations.items():
            eval_url = f'http://moabb.neurotechx.com/docs/generated/moabb.evaluations.{evaluation.replace("-", "")}Evaluation.html'
            subtask = client.task_add(
                TaskCreateRequest(
                    name=f"{evaluation} {paradigm_name}",
                    description=f"MOABB's {evaluation} evaluation for the {paradigm_name} paradigm.\n\nEvaluation details: {eval_url}",
                    area="medical",
                    parent_task=task.id,
                )
            )
            tasks[(paradigm_class, evaluation_class)] = subtask
            if subparadigms is not None:
                for subparadigm in subparadigms:
                    client.task_add(
                        TaskCreateRequest(
                            name=f"{evaluation} {paradigm_name} ({subparadigm})",
                            description=f"MOABB's {evaluation} evaluation for the {paradigm_name} paradigm ({subparadigm}).\n\nEvaluation details: {eval_url}",
                            area="medical",
                            parent_task=subtask.id,
                        )
                    )
                    tasks[(paradigm_class, evaluation_class, subparadigm)] = subtask
    return tasks


def create_datasets(client):
    datasets = valid_datasets()
    pwc_datasets = {}
    for dataset in datasets:
        pwc_dataset = client.dataset_add(
            DatasetCreateRequest(
                name=dataset_name(dataset),
                full_name=dataset_full_name(dataset),
                url=dataset_url(dataset),
            )
        )
        pwc_datasets[dataset.code] = pwc_dataset
    return pwc_datasets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("token", type=str, help="PapersWithCode API token")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Pickle output file",
        default="paperswithcode_datasets_and_tasks.pickle",
    )
    args = parser.parse_args()

    client = PapersWithCodeClient(token=args.token)

    # create tasks
    tasks = create_tasks(client)

    # create datasets
    datasets = create_datasets(client)
    obj = {"datasets": datasets, "tasks": tasks}

    with open(args.output, "wb") as f:
        pickle.dump(obj, f)
    print(f"Datasets and tasks saved to {args.output}")
