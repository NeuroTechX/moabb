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
    verbose=False,
    debug=False,
    output="./",
    threads=1,
    plot=False,
    contexts=None,
):
    """
    Function to run benchmark from scripts.
    Load from saved pipeline configurations, identify sui
    Take input of paradigms, option to exclude certain datasets

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
    verbose:
        verbosity level
    debug: bool
        Print debug level parse statements. Overrides verbose
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
            context = eval_type[evaluation](
                paradigm=p, random_state=42, n_jobs=threads, overwrite=force
            )
            results = context.process(pipelines=paradigms[paradigm])
            eval_results[f"{paradigm}"] = results

        # Combining the FilterBank and the base Paradigm
        combine_paradigms = ["SSVEP", "MotorImagery"]
        for p in combine_paradigms:
            if f"FilterBank{p}" in eval_results.keys() and f"{p}" in eval_results.keys():
                eval_results[f"{p}"] = pd.concat(
                    [eval_results[f"{p}"], eval_results[f"FilterBank{p}"]]
                )
                del eval_results[f"FilterBank{p}"]

        for paradigm_result in eval_results.values():
            analyze(pd.concat(paradigm_result, ignore_index=True), output, plot=plot)
