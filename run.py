#!/usr/bin/env python3

import os
import yaml
import hashlib

from glob import glob
from optparse import OptionParser
from collections import OrderedDict

# moabb specific imports
from moabb.pipelines.utils import create_pipeline_from_config
from moabb import contexts
from moabb.contexts.evaluations import WithinSessionEvaluation

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

        # get digest
        digest = hashlib.md5(content.encode('utf8')).hexdigest()
        outdir = os.path.join(options.results,
                              digest + '_(' + config['name'] + ')')

        # iterate over paradigms
        for paradigm in config['paradigms']:
            outpath = os.path.join(outdir, paradigm)

            # if folder exist and we not forcing it, we go to the next step
            # we should do something smarter with caching on the dataset level
            # so that we dont run everything twice.
            if os.path.isdir(outpath) & (not options.force):
                continue

            pipe = create_pipeline_from_config(config['pipeline'])

            pipeline = {
                'pipeline': pipe,
                'name': config['name'],
                'path': outpath
            }

            # append the pipeline in the paradigm list
            if paradigm not in paradigms.keys():
                paradigms[paradigm] = []

            paradigms[paradigm].append(pipeline)

        if os.path.isdir(outdir):
            continue

        # create folder
        os.makedirs(outdir)

        # save config file
        with open(os.path.join(outdir, 'config.yml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

# we can do the same for python defined pipeline
python_files = glob(os.path.join(options.pipelines, '*.py'))

for paradigm in paradigms:
    # get the context
    # FIXME name are not unique
    pipelines = {p['name']: p['pipeline'] for p in paradigms[paradigm]}
    context = getattr(contexts, paradigm)(
        pipelines=pipelines, evaluator=WithinSessionEvaluation())
    context.process()

    for pipe in paradigms[paradigm]:
        os.makedirs(pipe['path'])
        results[pipe['name']].to_csv(os.path.join(pipe['path'], 'results.csv'))
