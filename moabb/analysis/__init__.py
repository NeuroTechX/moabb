import importlib
import importlib.util
import logging
import os
import platform
from datetime import datetime


log = logging.getLogger(__name__)


def analyze(results, out_path, name="analysis", plot=False):
    """Analyze results.

    Given a results dataframe, generates a folder with
    results and a dataframe of the exact data used to generate those results,
    as well as introspection to return information on the computer

    Parameters
    ----------
    out_path: location to store analysis folder

    results: Dataframe generated from Results object

    path: string/None

    plot: whether to plot results

    Either path or results is necessary
    """
    from mne.utils import _open_lock

    from moabb.analysis import plotting as plt
    from moabb.analysis.meta_analysis import (
        compute_dataset_statistics,
        find_significant_differences,
    )

    # input checks #
    if not isinstance(out_path, str):
        raise ValueError("Given out_path argument is not string")
    elif not os.path.isdir(out_path):
        raise IOError("Given directory does not exist")
    else:
        analysis_path = os.path.join(out_path, name)

    unique_ids = [plt._simplify_names(x) for x in results.pipeline.unique()]
    simplify = True
    if len(unique_ids) != len(set(unique_ids)):
        log.warning("Pipeline names are too similar, turning off name shortening")
        simplify = False

    os.makedirs(analysis_path, exist_ok=True)
    # TODO: no good cross-platform way of recording CPU info?
    with _open_lock(os.path.join(analysis_path, "info.txt"), "a") as f:
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


# ``plotting`` applies a seaborn theme to the global matplotlib rcParams at
# import time. Re-exporting its names lazily keeps that off the path of anyone
# who reaches this package for something else -- ``moabb.datasets`` imports
# ``moabb.analysis.results`` for ``get_digest``, so an eager import here
# restyled the figures of every caller who merely imported a dataset.
_LAZY_IMPORTS = {
    "adjusted_chance_level": ".chance_level",
    "chance_by_chance": ".chance_level",
    "compute_dataset_statistics": ".meta_analysis",
    "find_significant_differences": ".meta_analysis",
    "Results": ".results",
    "MOABB_PALETTE": ".style",
    "apply_moabb_style": ".style",
    "class_balance_svg": ".timeline",
    "plot_class_balance": ".timeline",
    "plot_session_structure": ".timeline",
    "plot_stimulus_timeline": ".timeline",
    "session_structure_svg": ".timeline",
    "stimulus_timeline_svg": ".timeline",
    "codecarbon_plot": ".plotting",
    "distribution_plot": ".plotting",
    "emissions_summary": ".plotting",
    "plot_critical_difference": ".plotting",
}

if importlib.util.find_spec("plotly") is not None:
    _LAZY_IMPORTS["generate_neural_signature"] = ".neural_signatures"
    _LAZY_IMPORTS["neural_signature_html"] = ".neural_signatures"


def __getattr__(name):
    if name == "plotting":
        plotting = importlib.import_module("moabb.analysis.plotting")
        globals()[name] = plotting
        return plotting
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    try:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{name}":
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS) | {"plotting"})

