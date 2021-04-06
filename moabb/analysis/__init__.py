import logging
import os
import platform
from datetime import datetime

from moabb.analysis import plotting as plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
from moabb.analysis.results import Results  # noqa: F401


log = logging.getLogger(__name__)


def analyze(results, out_path, name="analysis", plot=False):
    """Analyze results.

    Given a results dataframe, generates a folder with
    results and a dataframe of the exact data used to generate those results,
    aswell as introspection to return information on the computer

    parameters
    ----------
    out_path: location to store analysis folder

    results: Dataframe generated from Results object

    path: string/None

    plot: whether to plot results

    Either path or results is necessary

    """
    # input checks #
    if not isinstance(out_path, str):
        raise ValueError("Given out_path argument is not string")
    elif not os.path.isdir(out_path):
        raise IOError("Given directory does not exist")
    else:
        analysis_path = os.path.join(out_path, name)

    unique_ids = [plt._simplify_names(x) for x in results.pipeline.unique()]
    simplify = True
    print(unique_ids)
    print(set(unique_ids))
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Pipeline names are too similar, turning off name shortening")
        simplify = False

    os.makedirs(analysis_path, exist_ok=True)
    # TODO: no good cross-platform way of recording CPU info?
    with open(os.path.join(analysis_path, "info.txt"), "a") as f:
        dt = datetime.now()
        f.write("Date: {:%Y-%m-%d}\n Time: {:%H:%M}\n".format(dt, dt))
        f.write("System: {}\n".format(platform.system()))
        f.write("CPU: {}\n".format(platform.processor()))

    results.to_csv(os.path.join(analysis_path, "data.csv"))

    stats = compute_dataset_statistics(results)
    stats.to_csv(os.path.join(analysis_path, "stats.csv"))
    P, T = find_significant_differences(stats)
    if plot:
        fig, color_dict = plt.score_plot(results)
        fig.savefig(os.path.join(analysis_path, "scores.pdf"))
        fig = plt.summary_plot(P, T, simplify=simplify)
        fig.savefig(os.path.join(analysis_path, "ordering.pdf"))
