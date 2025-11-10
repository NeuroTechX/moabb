import logging
import os
import os.path as osp
from pathlib import Path

import mne
import pandas as pd
import yaml
from mne.utils import _open_lock

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
    n_splits=None,
    cache_config=None,
    optuna=False,
):
    """Run benchmarks for selected pipelines and datasets.

    Load from saved pipeline configurations to determine associated paradigms. It is
    possible to include or exclude specific datasets and to choose the type of
    evaluation.

    If particular paradigms are mentioned through select_paradigms, only the pipelines corresponding to those paradigms
    will be run. If no paradigms are mentioned, all pipelines will be run.

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
    pipelines: str or list of dict
       Folder containing the pipelines to evaluate or path to a single pipeline file,
       or a list of scikit-learn pipelines with format:

       pipelines = [
                    {
                        "paradigms": ["SomeParadigm"],
                        "pipeline": make_pipeline(Transformer1(), Transformer2(), Classifier()),
                        "name": "PipelineName"
                    },
                    {
                        "paradigms": ["AnotherParadigm"],
                        "pipeline": make_pipeline(TransformerA(), ClassifierB()),
                        "name": "AnotherPipelineName"
                    }
                   ]
       Each entry is a dictionary with 3 keys: "name", "pipeline", "paradigms".
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
    n_splits: int or None, default=None
        This parameter only works for CrossSubjectEvaluation. It defines the
        number of splits to be done in the cross-validation. If None,
        the number of splits is equal to the number of subjects in the dataset.
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
    optuna: Enable Optuna for the hyperparameter search

    Returns
    -------
    eval_results: DataFrame
        Results of benchmark for all considered paradigms

    Notes
    -----
    .. versionadded:: 1.1.1
        Includes the possibility to use Optuna for hyperparameter search.

    .. versionadded:: 0.5.0
        Create the function to run the benchmark
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

    if isinstance(pipelines, str):
        pipeline_configs = parse_pipelines_from_directory(pipelines)
    elif isinstance(pipelines, list):
        pipeline_configs = pipelines
    else:
        raise TypeError(f"Unsupported pipelines type {type(pipelines)}.")

    context_params = {}
    if contexts is not None:
        with _open_lock(contexts, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)

    pipeline_prdgms = generate_paradigms(pipeline_configs, context_params, log)
    
    # Filter requested benchmark paradigms vs available in provided pipelines
    prdgms = filter_paradigms(pipeline_prdgms, paradigms, log)
    

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
            print(
                f"Datasets considered for {paradigm} paradigm {[dt.code for dt in d]}"
            )

            ppl_with_epochs, ppl_with_array = {}, {}
            for pn, pv in prdgms[paradigm].items():
                ppl_with_array[pn] = pv

            if len(ppl_with_epochs) > 0:
                # Keras pipelines require return_epochs=True
                context = eval_type[evaluation](
                    paradigm=p,
                    datasets=d,
                    random_state=42,
                    hdf5_path=results,
                    n_jobs=n_jobs,
                    overwrite=overwrite,
                    return_epochs=True,
                    n_splits=n_splits,
                    cache_config=cache_config,
                    optuna=optuna,
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
                    n_splits=n_splits,
                    cache_config=cache_config,
                    optuna=optuna,
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
    """Print results after computation."""
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
    """Combining FilterBank and direct paradigms.

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
    """Save results in specified folder.

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


def _inc_exc_datasets(datasets, include_datasets=None, exclude_datasets=None):
    """
    Filter datasets based on include_datasets and exclude_datasets.

    Parameters
    ----------
    datasets : list
        List of dataset class instances (each with a `.code` attribute).
    include_datasets : list[str or Dataset], optional
        List of dataset codes or dataset class instances to include.
    exclude_datasets : list[str or Dataset], optional
        List of dataset codes or dataset class instances to exclude.

    Returns
    -------
    list
        Filtered list of dataset class instances.

    """
    # --- Safety checks ---
    if include_datasets is not None and exclude_datasets is not None:
        raise ValueError("Cannot specify both include_datasets and exclude_datasets.")

    all_codes = [ds.code for ds in datasets]
    d = list(datasets)

    # --- Helper to validate and normalize inputs ---
    def _validate_dataset_list(ds_list, list_name):
        """Ensure list is consistent and corresponds to existing datasets."""
        if not isinstance(ds_list, (list, tuple)):
            raise TypeError(f"{list_name} must be a list or tuple.")

        # Empty list edge case
        if len(ds_list) == 0:
            raise ValueError(f"{list_name} cannot be an empty list.")

        # All strings or all class instances — not a mix
        all_str = all(isinstance(x, str) for x in ds_list)
        all_obj = all(hasattr(x, "code") for x in ds_list)
        if not (all_str or all_obj):
            raise TypeError(f"{list_name} must contain either all strings or all dataset objects, not a mix.")

        # Convert all to codes
        if all_str:
            # Check uniqueness
            if len(ds_list) != len(set(ds_list)):
                raise ValueError(f"{list_name} contains duplicate dataset codes.")

            # Check validity
            invalid = [x for x in ds_list if x not in all_codes]
            if invalid:
                raise ValueError(f"Invalid dataset codes in {list_name}: {invalid}")
            return ds_list

        elif all_obj:
            # Ensure they are unique by code
            codes = [x.code for x in ds_list]
            if len(codes) != len(set(codes)):
                raise ValueError(f"{list_name} contains duplicate dataset instances.")
            # Check that all objects exist in available datasets
            invalid = [x.code for x in ds_list if x.code not in all_codes]
            if invalid:
                raise ValueError(f"Some datasets in {list_name} are not part of available datasets: {invalid}")
            return codes

    # --- Inclusion logic ---
    if include_datasets is not None:
        include_codes = _validate_dataset_list(include_datasets, "include_datasets")
        # Keep only included datasets
        filtered = [ds for ds in datasets if ds.code in include_codes]
        return filtered

    # --- Exclusion logic ---
    if exclude_datasets is not None:
        exclude_codes = _validate_dataset_list(exclude_datasets, "exclude_datasets")
        # Remove excluded datasets
        filtered = [ds for ds in datasets if ds.code not in exclude_codes]
        return filtered

    return d
    
    
 def filter_paradigms(pipeline_prdgms, paradigms, logger):
    """
    Filter a dictionary of paradigms and their pipelines based on user selection.

    Parameters
    ----------
    pipeline_prdgms : dict
        Dictionary mapping paradigm names to pipeline definitions.
        Example: {"MotorImagery": {...}, "SSVEP": {...}}
    paradigms : list[str] or None
        List of target paradigms to include (e.g., ["MotorImagery", "SSVEP"]).
        If None, all paradigms are kept.
    logger : logging.Logger
        Logger instance used to print warnings or info messages.

    Returns
    -------
    dict
        Filtered dictionary containing only paradigms that exist in `pipeline_prdgms`.

    if paradigms is None:
        logger.debug("No paradigms filter specified; using all available paradigms.")
        return pipeline_prdgms

    # Collect available vs. requested paradigms
    available_paradigms = set(pipeline_prdgms.keys())
    requested_paradigms = set(paradigms)

    valid_paradigms = requested_paradigms & available_paradigms
    missing_paradigms = requested_paradigms - available_paradigms

    # Log warnings for missing ones
    if missing_paradigms:
        logger.warning(
            "The following paradigms were requested but have no corresponding pipelines: "
            f"{sorted(missing_paradigms)}"
        )

    # Keep only valid ones
    filtered_prdgms = {p: pipeline_prdgms[p] for p in valid_paradigms}

    # Error if nothing left
    if not filtered_prdgms:
        raise ValueError(
            f"No pipelines correspond to the requested paradigms {sorted(paradigms)} for benchmark. "
            f"Available paradigms based on pipelines are: {sorted(available_paradigms)}"
        )

    return filtered_prdgms
