import logging

import mne
import pandas as pd
import yaml

from moabb import paradigms as moabb_paradigms
from moabb.analysis import analyze
from moabb.evaluations import WithinSessionEvaluation
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

    all_results = []
    # TODO : Extend to run other evaluations based on evaluations param list
    # If statement? Or would that be wastage in run time
    # Loop paradigms within evaluations
    for paradigm in paradigms:
        # get the context
        log.debug("{}: {}".format(paradigm, context_params[paradigm]))
        p = getattr(moabb_paradigms, paradigm)(**context_params[paradigm])
        context = WithinSessionEvaluation(
            paradigm=p, random_state=42, n_jobs=threads, overwrite=force
        )
        # TODO : If filterbank type paradigm is run, concatenate the results of the base and
        #  filterbank. Don't append everything else. Need to keep paradigm results separate.
        results = context.process(pipelines=paradigms[paradigm])
        all_results.append(results)
    analyze(pd.concat(all_results, ignore_index=True), output, plot=plot)
