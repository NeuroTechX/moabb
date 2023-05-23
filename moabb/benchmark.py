import logging
import os
import os.path as osp
from pathlib import Path

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
from moabb.pipelines.utils import (
    generate_paradigms,
    generate_param_grid,
    parse_pipelines_from_directory,
)


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


log = logging.getLogger(__name__)


def benchmark(  # noqa: C901
    pipelines="./pipelines/",
    evaluations=None,
    paradigms=None,
    results="./results/",
    overwrite=False,
    output="./benchmark/",
    n_jobs=-1,
    plot=False,
    contexts=None,
    include_datasets=None,
    exclude_datasets=None,
):
    """Run benchmarks for selected pipelines and datasets

    Load from saved pipeline configurations to determine associated paradigms. It is
    possible to include or exclude specific datasets and to choose the type of
    evaluation.

    If particular paradigms are mentioned through select_paradigms, only the pipelines corresponding to those paradigms
    will be run. If no paradigms are mentioned, all pipelines will be run.

    Pipelines stored in a file named braindecode_xxx.py will be recognized as Braindecode architectures
    and they will receive epochs as input, instead of numpy array.

    To define the include_datasets or exclude_dataset, you could start from the full dataset list,
    using for example the following code:
    > # Choose your paradigm
    > p = moabb.paradigms.SSVEP()
    > # Get the class names
    > print(p.datasets)
    > # Get the dataset code
    > print([d.code for d in p.datasets])

    Parameters
    ----------
    pipelines: str
        Folder containing the pipelines to evaluate
    evaluations: list of str
        If to restrict the types of evaluations to be run. By default, all 3 base types are run
        Can be a list of these elements ["WithinSession", "CrossSession", "CrossSubject"]
    paradigms: list of str
        To restrict the paradigms on which evaluations should be run.
        Can be a list of these elements ['LeftRightImagery', 'MotorImagery', 'FilterBankSSVEP', 'SSVEP',
        'FilterBankMotorImagery']
    results: str
        Folder to store the results
    overwrite: bool
        Force evaluation of cached pipelines
    output: str
        Folder to store the analysis results
    n_jobs: int
        Number of threads to use for running parallel jobs
    plot: bool
        Plot results after computing
    contexts: str
        File path to context.yml file that describes context parameters.
        If none, assumes all defaults. Must contain an entry for all
        paradigms described in the pipelines.
    include_datasets: list of str or Dataset object
        Datasets (dataset.code or object) to include in the benchmark run.
        By default, all suitable datasets are included. If both include_datasets
        and exclude_datasets are specified, raise an error.
    exclude_datasets: list of str or Dataset object
        Datasets to exclude from the benchmark run

    Returns
    -------
    eval_results: DataFrame
        Results of benchmark for all considered paradigms

    Notes
    -----
    .. versionadded:: 0.5.0
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

    output = Path(output)
    if not osp.isdir(output):
        os.makedirs(output)

    pipeline_configs = parse_pipelines_from_directory(pipelines)

    context_params = {}
    if contexts is not None:
        with open(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)

    prdgms = generate_paradigms(pipeline_configs, context_params, log)
    if paradigms is not None:
        prdgms = {p: prdgms[p] for p in paradigms}

    param_grid = generate_param_grid(pipeline_configs, context_params, log)

    log.debug(f"The paradigms being run are {prdgms.keys()}")

    if len(context_params) == 0:
        for paradigm in prdgms:
            context_params[paradigm] = {}

    # Looping over the evaluations to be done
    df_eval = []
    for evaluation in evaluations:
        eval_results = dict()
        for paradigm in prdgms:
            # get the context
            log.debug(f"{paradigm}: {context_params[paradigm]}")
            p = getattr(moabb_paradigms, paradigm)(**context_params[paradigm])
            # List of dataset class instances
            datasets = p.datasets
            d = _inc_exc_datasets(datasets, include_datasets, exclude_datasets)
            log.debug(
                f"Datasets considered for {paradigm} paradigm {[dt.code for dt in d]}"
            )

            ppl_with_epochs, ppl_with_array = {}, {}
            for pn, pv in prdgms[paradigm].items():
                if "braindecode" in pn:
                    ppl_with_epochs[pn] = pv
                else:
                    ppl_with_array[pn] = pv

            if len(ppl_with_epochs) > 0:
                # Braindecode pipelines require return_epochs=True
                context = eval_type[evaluation](
                    paradigm=p,
                    datasets=d,
                    random_state=42,
                    hdf5_path=results,
                    n_jobs=1,
                    overwrite=overwrite,
                    return_epochs=True,
                )
                paradigm_results = context.process(
                    pipelines=ppl_with_epochs, param_grid=param_grid
                )
                paradigm_results["paradigm"] = f"{paradigm}"
                paradigm_results["evaluation"] = f"{evaluation}"
                eval_results[f"{paradigm}"] = paradigm_results
                df_eval.append(paradigm_results)

            # Other pipelines, that use numpy arrays
            if len(ppl_with_array) > 0:
                context = eval_type[evaluation](
                    paradigm=p,
                    datasets=d,
                    random_state=42,
                    hdf5_path=results,
                    n_jobs=n_jobs,
                    overwrite=overwrite,
                )
                paradigm_results = context.process(
                    pipelines=ppl_with_array, param_grid=param_grid
                )
                paradigm_results["paradigm"] = f"{paradigm}"
                paradigm_results["evaluation"] = f"{evaluation}"
                eval_results[f"{paradigm}"] = paradigm_results
                df_eval.append(paradigm_results)

        # Combining FilterBank and direct paradigms
        eval_results = _combine_paradigms(eval_results)

        _save_results(eval_results, output, plot)

    df_eval = pd.concat(df_eval)
    _display_results(df_eval)

    return df_eval


def _display_results(results):
    """Print results after computation"""
    tab = []
    for d in results["dataset"].unique():
        for p in results["pipeline"].unique():
            for e in results["evaluation"].unique():
                r = {
                    "dataset": d,
                    "evaluation": e,
                    "pipeline": p,
                    "avg score": results[
                        (results["dataset"] == d)
                        & (results["pipeline"] == p)
                        & (results["evaluation"] == e)
                    ]["score"].mean(),
                }
                if _carbonfootprint:
                    r["carbon emission"] = results[
                        (results["dataset"] == d)
                        & (results["pipeline"] == p)
                        & (results["evaluation"] == e)
                    ]["carbon_emission"].sum()
                tab.append(r)
    tab = pd.DataFrame(tab)
    print(tab)


def _combine_paradigms(prdgm_results):
    """Combining FilterBank and direct paradigms

    Applied only on SSVEP for now.

    Parameters
    ----------
    prdgm_results: dict of DataFrame
        Results of benchmark for all considered paradigms

    Returns
    -------
    eval_results: dict of DataFrame
        Results with filterbank and direct paradigms combined
    """
    eval_results = prdgm_results.copy()
    combine_paradigms = ["SSVEP"]
    for p in combine_paradigms:
        if f"FilterBank{p}" in eval_results.keys() and f"{p}" in eval_results.keys():
            eval_results[f"{p}"] = pd.concat(
                [eval_results[f"{p}"], eval_results[f"FilterBank{p}"]]
            )
            del eval_results[f"FilterBank{p}"]
    return eval_results


def _save_results(eval_results, output, plot):
    """Save results in specified folder

    Parameters
    ----------
    eval_results: dict of DataFrame
        Results of benchmark for all considered paradigms
    output: str or Path
        Folder to store the analysis results
    plot: bool
        Plot results after computing
    """
    for prdgm, prdgm_result in eval_results.items():
        prdgm_path = Path(output) / prdgm
        if not osp.isdir(prdgm_path):
            prdgm_path.mkdir()
        analyze(prdgm_result, str(prdgm_path), plot=plot)


def _inc_exc_datasets(datasets, include_datasets, exclude_datasets):
    d = list()
    if include_datasets is not None:
        # Assert if the inputs are key_codes
        if isinstance(include_datasets[0], str):
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
        if exclude_datasets is not None:
            raise AttributeError(
                "You could not specify both include and exclude datasets"
            )

    elif exclude_datasets is not None:
        d = datasets
        # Assert if the inputs are not key_codes i.e. expected to be dataset class objects
        if not isinstance(exclude_datasets[0], str):
            # Convert the input to key_codes
            exclude_datasets = [e.code for e in exclude_datasets]

        # Map from key_codes to class instances
        datasets_codes = [d.code for d in datasets]
        for excdat in exclude_datasets:
            del d[datasets_codes.index(excdat)]
    else:
        d = datasets
    return d
