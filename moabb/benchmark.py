import logging

import mne
import pandas as pd
import yaml

from moabb import paradigms as moabb_paradigms
from moabb.analysis import analyze
from moabb.evaluations import (
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
    WithinSessionEvaluation,
)
from moabb.pipelines.utils import generate_paradigms, parse_pipelines_from_directory


log = logging.getLogger(__name__)


def benchmark(
    pipelines="./pipelines/",
    evaluations=None,
    results="./results/",
    force=False,
    output="./",
    threads=1,
    plot=False,
    contexts=None,
    include_datasets=None,
    exclude_datasets=None,
):
    """
    Function to run benchmark from scripts.
    Load from saved pipeline configurations
    Take input of paradigms, option to exclude certain datasets

    In case of using include_datasets or exclude_dataset, you can get possible dataset options
    for each paradigm as shown in this example -

    # Replace paradigm name as required

    p = moabb.paradigms.SSVEP()

    # Alternatively

    p = moabb.datasets.utils.dataset_search("ssvep")

    # For the class names

    print(p.datasets)

    # For the key_codes

    print([d.code for d in p.datasets])

    Parameters
    ----------
    pipelines: str
        Folder containing the pipelines to evaluate
    evaluations: list of str
        If to restrict the types of evaluations to be run. By default all 3 base types are run
    results: str
        Folder to store the results
    force: bool
        Force evaluation of cached pipelines
    output: str
        Folder to store the analysis results
    threads: int
        Number of threads to use for running parallel jobs
    plot: bool
        Plot results after computing
    contexts: str
        File path to context.yml file that describes context parameters.
        If none, assumes all defaults. Must contain an entry for all
        paradigms described in the pipelines
    include_datasets: list of str
        Datasets to include in the benchmark run. By default all suitable datasets are taken.
        If arguments are given for both include_datasets as well as exclude_datasets,
        include_datasets will take precedence and exclude_datasets will be neglected.
    exclude_datasets: list of str
        Datasets to exclude from the benchmark run.

    Returns
    -------

    """
    # set logs
    if evaluations is None:
        evaluations = ["WithinSession", "CrossSession", "CrossSubject"]

    eval_type = {
        "WithinSession": WithinSessionEvaluation,
        "CrossSession": CrossSessionEvaluation,
        "CrossSubject": CrossSubjectEvaluation,
    }

    mne.set_log_level(False)
    # logging.basicConfig(level=logging.WARNING)

    pipeline_configs = parse_pipelines_from_directory(pipelines)

    context_params = {}
    if contexts is not None:
        with open(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)

    paradigms = generate_paradigms(pipeline_configs, context_params, log)

    if len(context_params) == 0:
        for paradigm in paradigms:
            context_params[paradigm] = {}

    # Looping over the evaluations to be done
    for evaluation in evaluations:
        eval_results = dict()
        for paradigm in paradigms:
            # get the context
            log.debug(f"{paradigm}: {context_params[paradigm]}")
            p = getattr(moabb_paradigms, paradigm)(**context_params[paradigm])
            log.debug(f"Datasets in this paradigm {[d.code for d in p.datasets]}")
            print(f"Datasets in this paradigm {[d.code for d in p.datasets]}")
            # List of dataset class instances
            datasets = p.datasets
            d = _inc_exc_datasets(datasets, include_datasets, exclude_datasets)

            # if len(d) = 0, raise warning that no suitable datasets were present after the
            # arguments were satisfied
            if len(d) == 0:
                log.debug("No datasets matched the include_datasets or exclude_datasets")
                print("No datasets matched the include_datasets or exclude_datasets")

            context = eval_type[evaluation](
                paradigm=p,
                datasets=d,
                random_state=42,
                hdf5_path=results,
                n_jobs=threads,
                overwrite=force,
            )
            paradigm_results = context.process(pipelines=paradigms[paradigm])
            eval_results[f"{paradigm}"] = paradigm_results

        # Combining the FilterBank and the base Paradigm
        combine_paradigms = ["SSVEP", "MotorImagery"]
        for p in combine_paradigms:
            if f"FilterBank{p}" in eval_results.keys() and f"{p}" in eval_results.keys():
                eval_results[f"{p}"] = pd.concat(
                    [eval_results[f"{p}"], eval_results[f"FilterBank{p}"]]
                )
                del eval_results[f"FilterBank{p}"]

        for paradigm_result in eval_results.values():
            analyze(paradigm_result, output, plot=plot)


def _inc_exc_datasets(datasets, include_datasets, exclude_datasets):
    d = list()
    if include_datasets is not None:
        # Assert if the inputs are key_codes
        if isinstance(datasets[0], str):
            # Map from key_codes to class instances
            datasets_codes = [d.code for d in datasets]
            # Get the indices of the matching datasets
            for incdat in include_datasets:
                if incdat in datasets_codes:
                    d.append(datasets[datasets_codes.index(incdat)])

        else:
            # The case where the class instances have been given
            # can be passed on directly
            d = include_datasets

    elif exclude_datasets is not None:
        d = datasets
        # Assert if the inputs are not key_codes i.e expected to be dataset class objects
        if not isinstance(datasets[0], str):
            # Convert the input to key_codes
            exclude_datasets = [e.code for e in exclude_datasets]

        # Map from key_codes to class instances
        datasets_codes = [d.code for d in datasets]
        for excdat in exclude_datasets:
            del d[datasets_codes.index(excdat)]
    else:
        d = datasets
    return d
