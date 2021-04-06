import importlib
import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import mne
import pandas as pd
import yaml
from sklearn.base import BaseEstimator

from moabb import paradigms as moabb_paradigms
from moabb.analysis import analyze
from moabb.analysis.results import get_string_rep
from moabb.evaluations import WithinSessionEvaluation

# moabb specific imports
from moabb.pipelines.utils import create_pipeline_from_config


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


def parse_pipelines_from_directory(d):
    """
    Given directory, returns generated pipeline config dictionaries. Each entry
    has structure:
    'name': string
    'pipeline': sklearn.BaseEstimator
    'paradigms': list of class names that are compatible with said pipeline
    """
    assert os.path.isdir(
        os.path.abspath(d)
    ), "Given pipeline path {} is not valid".format(d)

    # get list of config files
    yaml_files = glob(os.path.join(d, "*.yml"))

    pipeline_configs = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as _file:
            content = _file.read()

            # load config
            config_dict = yaml.load(content, Loader=yaml.FullLoader)
            ppl = create_pipeline_from_config(config_dict["pipeline"])
            pipeline_configs.append(
                {
                    "paradigms": config_dict["paradigms"],
                    "pipeline": ppl,
                    "name": config_dict["name"],
                }
            )

    # we can do the same for python defined pipeline
    python_files = glob(os.path.join(d, "*.py"))

    for python_file in python_files:
        spec = importlib.util.spec_from_file_location("custom", python_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        pipeline_configs.append(foo.PIPELINE)
    return pipeline_configs


def generate_paradigms(pipeline_configs, context=None):
    context = context or {}
    paradigms = OrderedDict()
    for config in pipeline_configs:

        if "paradigms" not in config.keys():
            log.error("{} must have a 'paradigms' key.".format(config))
            continue

        # iterate over paradigms

        for paradigm in config["paradigms"]:

            # check if it is in the context parameters file
            if len(context) > 0:
                if paradigm not in context.keys():
                    log.debug(context)
                    log.warning(
                        "Paradigm {} not in context file {}".format(
                            paradigm, context.keys()
                        )
                    )

            if isinstance(config["pipeline"], BaseEstimator):
                pipeline = deepcopy(config["pipeline"])
            else:
                log.error(config["pipeline"])
                raise (ValueError("pipeline must be a sklearn estimator"))

            # append the pipeline in the paradigm list
            if paradigm not in paradigms.keys():
                paradigms[paradigm] = {}

            # FIXME name are not unique
            log.debug("Pipeline: \n\n {} \n".format(get_string_rep(pipeline)))
            paradigms[paradigm][config["name"]] = pipeline

    return paradigms


if __name__ == "__main__":
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
