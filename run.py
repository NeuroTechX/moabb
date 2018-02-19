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

from moabb.datasets.bnci import (BNCI2014001, BNCI2014002,
                                 BNCI2014004, BNCI2015001, BNCI2015004)

from moabb.datasets.alex_mi import AlexMI
from moabb.datasets.bbci_eeg_fnirs import BBCIEEGfNIRS
from moabb.datasets.gigadb import GigaDbMI
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.openvibe_mi import OpenvibeMI

parser = OptionParser()
parser.add_option("-p", "--pipelines",
                  dest="pipelines", type='str', default='./pipelines/',
                  help="Folder containing the pipelines to evaluates.")
parser.add_option("-r", "--results",
                  dest="results", type='str', default='./results/',
                  help="Folder to store the results.")
parser.add_option("-f", "--force-update",
                  dest="force", action="store_true", default=False,
                  help="Force evaluation of cached pipelines.")

(options, args) = parser.parse_args()

# FIXME : get automatically datset compatibles with some contexts
DATASETS = {'MotorImageryTwoClasses': [AlexMI(),
                                       OpenvibeMI(),
                                       BNCI2015004(motor_imagery=True),
                                       PhysionetMI(),
                                       GigaDbMI(),
                                       BBCIEEGfNIRS()],

            'MotorImageryMultiClasses': [AlexMI(with_rest=True),
                                         #BNCI2014001(),
                                         PhysionetMI(with_rest=True,
                                                     feets=False)]}

paradigms = OrderedDict()

# get list of config files
yaml_files = glob(os.path.join(options.pipelines, '*.yml'))

for yaml_file in yaml_files:
    with open(yaml_file, 'r') as yml:
        content = yml.read()

        # get digest
        digest = hashlib.md5(content.encode('utf8')).hexdigest()
        outdir = os.path.join(options.results, digest)

        # load config
        config = yaml.load(content)

        # iterate over paradigms
        for paradigm in config['paradigms']:
            outpath = os.path.join(outdir, paradigm)

            # if folder exist and we not forcing it, we go to the next step
            # we should do something smarter with caching on the dataset level
            # so that we dont run everything twice.
            if os.path.isdir(outpath) & (not options.force):
                continue

            pipe = create_pipeline_from_config(config['pipeline'])

            pipeline = {'pipeline': pipe,
                        'name': config['name'],
                        'path': outpath}

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
    pipelines = {p['name']: p['pipeline'] for p in paradigms[paradigm]}
    datasets = DATASETS[paradigm]
    context = getattr(contexts, paradigm)(pipelines=pipelines,
                                          datasets=datasets)
    results = context.evaluate(verbose=True)

    for pipe in paradigms[paradigm]:
        os.makedirs(pipe['path'])
        results[pipe['name']].to_csv(os.path.join(pipe['path'], 'results.csv'))
