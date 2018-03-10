#!/usr/bin/env python3

import os
import yaml
import mne
import logging
import coloredlogs

from glob import glob
from optparse import OptionParser
from collections import OrderedDict

# moabb specific imports
from moabb.pipelines.utils import create_pipeline_from_config
from moabb import paradigms as para
from moabb.evaluations import WithinSessionEvaluation

# set logs
mne.set_log_level(False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
coloredlogs.install(level=logging.INFO)

parser = OptionParser()
parser.add_option(
    "-p",
    "--pipelines",
    dest="pipelines",
    type='str',
    default='./pipelines/',
    help="Folder containing the pipelines to evaluates.")
parser.add_option(
    "-r",
    "--results",
    dest="results",
    type='str',
    default='./results/',
    help="Folder to store the results.")
parser.add_option(
    "-f",
    "--force-update",
    dest="force",
    action="store_true",
    default=False,
    help="Force evaluation of cached pipelines.")

(options, args) = parser.parse_args()

paradigms = OrderedDict()

# get list of config files
yaml_files = glob(os.path.join(options.pipelines, '*.yml'))

for yaml_file in yaml_files:
    with open(yaml_file, 'r') as yml:
        content = yml.read()

        # load config
        config = yaml.load(content)

        # iterate over paradigms
        for paradigm in config['paradigms']:

            pipeline = create_pipeline_from_config(config['pipeline'])

            # append the pipeline in the paradigm list
            if paradigm not in paradigms.keys():
                paradigms[paradigm] = {}

            # FIXME name are not unique
            paradigms[paradigm][config['name']] = pipeline


# we can do the same for python defined pipeline
python_files = glob(os.path.join(options.pipelines, '*.py'))

for paradigm in paradigms:
    # get the context

    p = getattr(para, paradigm)()
    context = WithinSessionEvaluation(paradigm=p, random_state=42)
    results = context.process(pipelines=paradigms[paradigm])
