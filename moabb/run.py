#!/usr/bin/env python3

import os
import yaml
import mne
import logging
import coloredlogs
import importlib

from glob import glob
from argparse import ArgumentParser
from collections import OrderedDict
from sklearn.base import BaseEstimator
from copy import deepcopy

# moabb specific imports
from moabb.pipelines.utils import create_pipeline_from_config
from moabb import paradigms as moabb_paradigms
from moabb.evaluations import WithinSessionEvaluation
# set logs
mne.set_log_level(False)
# logging.basicConfig(level=logging.WARNING)
log = logging.getLogger()
# coloredlogs.install(level=logging.WARNING)

parser = ArgumentParser(description="Main run script for MOABB")
parser.add_argument(
    "-p",
    "--pipelines",
    dest="pipelines",
    type=str,
    default='./pipelines/',
    help="Folder containing the pipelines to evaluates.")
parser.add_argument(
    "-r",
    "--results",
    dest="results",
    type=str,
    default='./results/',
    help="Folder to store the results.")
parser.add_argument(
    "-f",
    "--force-update",
    dest="force",
    action="store_true",
    default=False,
    help="Force evaluation of cached pipelines.")

parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    default=False)
parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    default=False,
    help="Print debug level parse statements. Overrides verbose")

parser.add_argument(
    "-c",
    "--contexts",
    dest="context",
    type=str,
    default=None,
    help="File path to context.yml file that describes context parameters."
         "If none, assumes all defaults. Must contain an entry for all "
         "paradigms described in the pipelines")
options = parser.parse_args()

assert os.path.isdir(os.path.abspath(options.pipelines)
                     ), "Given pipeline path {} is not valid".format(options.pipelines)

if options.debug:
    coloredlogs.install(level=logging.DEBUG)
elif options.verbose:
    coloredlogs.install(level=logging.INFO)
else:
    coloredlogs.install(level=logging.WARNING)


# get list of config files
yaml_files = glob(os.path.join(options.pipelines, '*.yml'))

pipeline_configs = []
for yaml_file in yaml_files:
    with open(yaml_file, 'r') as _file:
        content = _file.read()

        # load config
        pipeline_configs.append(yaml.load(content))

# we can do the same for python defined pipeline
python_files = glob(os.path.join(options.pipelines, '*.py'))

for python_file in python_files:
    spec = importlib.util.spec_from_file_location("custom", python_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    pipeline_configs.append(foo.PIPELINE)

context_params = {}
if options.context is not None:
    with open(options.context, 'r') as cfile:
        context_params = yaml.load(cfile.read())

paradigms = OrderedDict()
for config in pipeline_configs:

    if 'paradigms' not in config.keys():
        log.error("{} must have a 'paradigms' key.".format(config))
        continue

    # iterate over paradigms

    for paradigm in config['paradigms']:

        # check if it is in the context parameters file
        if len(context_params) > 0:
            if paradigm not in context_params.keys():
                log.debug(context_params)
                log.warning("Paradigm {} not in context file {}".format(
                    paradigm, context_params.keys()))

        if isinstance(config['pipeline'], list):
            pipeline = create_pipeline_from_config(config['pipeline'])
        elif isinstance(config['pipeline'], BaseEstimator):
            pipeline = deepcopy(config['pipeline'])
        else:
            log.error(config['pipeline'])
            raise(ValueError('pipeline must be a list or a sklearn estimator'))

        # append the pipeline in the paradigm list
        if paradigm not in paradigms.keys():
            paradigms[paradigm] = {}

        # FIXME name are not unique
        log.debug('Pipeline: \n\n {} \n'.format(repr(pipeline.get_params())))
        paradigms[paradigm][config['name']] = pipeline

for paradigm in paradigms:
    # get the context
    if len(context_params) == 0:
        context_params[paradigm] = {}
    log.debug('{}: {}'.format(paradigm, context_params[paradigm]))
    p = getattr(moabb_paradigms, paradigm)(**context_params[paradigm])
    context = WithinSessionEvaluation(paradigm=p, random_state=42)
    results = context.process(pipelines=paradigms[paradigm])
