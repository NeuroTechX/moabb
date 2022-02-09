import logging
from argparse import ArgumentParser

import mne
import pandas as pd
import yaml

from moabb import paradigms as moabb_paradigms
from moabb.analysis import analyze
from moabb.evaluations import WithinSessionEvaluation
from moabb.pipelines.utils import generate_paradigms, parse_pipelines_from_directory


log = logging.getLogger(__name__)


def parser_init():
    parser = ArgumentParser(description="Main run script for MOABB")
    parser.add_argument(
        "-p",
        "--pipelines",
        dest="pipelines",
        type=str,
        default="./pipelines/",
        help="Folder containing the pipelines to evaluates.",
    )
    parser.add_argument(
        "-r",
        "--results",
        dest="results",
        type=str,
        default="./results/",
        help="Folder to store the results.",
    )
    parser.add_argument(
        "-f",
        "--force-update",
        dest="force",
        action="store_true",
        default=False,
        help="Force evaluation of cached pipelines.",
    )

    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Print debug level parse statements. Overrides verbose",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default="./",
        help="Folder to put analysis results",
    )
    parser.add_argument(
        "--threads", dest="threads", type=int, default=1, help="Number of threads to run"
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot results after computing. Defaults false",
    )
    parser.add_argument(
        "-c",
        "--contexts",
        dest="context",
        type=str,
        default=None,
        help="File path to context.yml file that describes context parameters."
        "If none, assumes all defaults. Must contain an entry for all "
        "paradigms described in the pipelines",
    )
    return parser


if __name__ == "__main__":
    # TODO: replace by call to moabb.benchmark
    # set logs
    mne.set_log_level(False)
    # logging.basicConfig(level=logging.WARNING)

    parser = parser_init()
    options = parser.parse_args()

    pipeline_configs = parse_pipelines_from_directory(options.pipelines)

    context_params = {}
    if options.context is not None:
        with open(options.context, "r") as cfile:
            context_params = yaml.load(cfile.read(), Loader=yaml.FullLoader)

    paradigms = generate_paradigms(pipeline_configs, context_params)

    if len(context_params) == 0:
        for paradigm in paradigms:
            context_params[paradigm] = {}

    all_results = []
    for paradigm in paradigms:
        # get the context
        log.debug("{}: {}".format(paradigm, context_params[paradigm]))
        p = getattr(moabb_paradigms, paradigm)(**context_params[paradigm])
        context = WithinSessionEvaluation(
            paradigm=p, random_state=42, n_jobs=options.threads, overwrite=options.force
        )
        results = context.process(pipelines=paradigms[paradigm])
        all_results.append(results)
    analyze(pd.concat(all_results, ignore_index=True), options.output, plot=options.plot)
