import importlib
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import numpy as np
import scipy.signal as scp
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from moabb.analysis.results import get_string_rep


log = logging.getLogger(__name__)


def create_pipeline_from_config(config):
    """Create a pipeline from a config file.

    takes a config dict as input and return the corresponding pipeline.

    If the pipeline is a Tensorflow pipeline it convert also the optimizer function and the callbacks.

    Parameters
    ----------
    config : Dict.
        Dict containing the config parameters.

    Returns
    -------
    pipeline : Pipeline
        sklearn Pipeline

    """
    components = []

    for component in config:
        # load the package
        mod = __import__(component["from"], fromlist=[component["name"]])
        # create the instance
        if "parameters" in component.keys():
            params = component["parameters"]
            if "optimizer" in component["parameters"].keys():
                for optm in component["parameters"]["optimizer"]:
                    mod_optm = __import__(name=optm["from"], fromlist=[optm["name"]])
                    params_optm = optm["parameters"]
                    instance = getattr(mod_optm, optm["name"])(**params_optm)
                    component["parameters"]["optimizer"] = instance

            if "callbacks" in component["parameters"].keys():
                cb = []
                for callbacks in component["parameters"]["callbacks"]:
                    mod_callbacks = __import__(
                        name=callbacks["from"], fromlist=[callbacks["name"]]
                    )
                    params_callbacks = callbacks["parameters"]
                    instance = getattr(mod_callbacks, callbacks["name"])(
                        **params_callbacks
                    )
                    cb.append(instance)
                component["parameters"]["callbacks"] = cb

        else:
            params = {}
        instance = getattr(mod, component["name"])(**params)
        components.append(instance)

    pipeline = make_pipeline(*components)
    return pipeline


def parse_pipelines_from_directory(dir_path):
    """
    Takes in the path to a directory with pipeline configuration files and returns a dictionary
    of pipelines.
    Parameters
    ----------
    dir_path: str
        Path to directory containing pipeline config .yml or .py files

    Returns
    -------
    pipeline_configs: dict
        Generated pipeline config dictionaries. Each entry has structure:
        'name': string
        'pipeline': sklearn.BaseEstimator
        'paradigms': list of class names that are compatible with said pipeline
    """
    assert os.path.isdir(
        os.path.abspath(dir_path)
    ), "Given pipeline path {} is not valid".format(dir_path)

    # get list of config files
    yaml_files = glob(os.path.join(dir_path, "*.yml"))

    pipeline_configs = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as _file:
            content = _file.read()

            # load config
            config_dict = yaml.load(content, Loader=yaml.FullLoader)
            ppl = create_pipeline_from_config(config_dict["pipeline"])
            if "param_grid" in config_dict:
                pipeline_configs.append(
                    {
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                        "param_grid": config_dict["param_grid"],
                    }
                )
            else:
                pipeline_configs.append(
                    {
                        "paradigms": config_dict["paradigms"],
                        "pipeline": ppl,
                        "name": config_dict["name"],
                    }
                )

    # we can do the same for python defined pipeline
    # TODO for python pipelines
    python_files = glob(os.path.join(dir_path, "*.py"))

    for python_file in python_files:
        spec = importlib.util.spec_from_file_location("custom", python_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        pipeline_configs.append(foo.PIPELINE)
    return pipeline_configs


def generate_paradigms(pipeline_configs, context=None, logger=log):
    """
    Takes in a dictionary of pipelines configurations as returned by
    parse_pipelines_from_directory and returns a dictionary of unique paradigms with all pipeline
    configurations compatible with that paradigm.
    Parameters
    ----------
    pipeline_configs:
        dictionary of pipeline configurations
    context:
        TODO:add description
    logger:
        logger

    Returns
    -------
    paradigms: dict
        Dictionary of dictionaries with the unique paradigms and the configuration of the
        pipelines compatible with the paradigm

    """
    context = context or {}
    paradigms = OrderedDict()
    for config in pipeline_configs:
        if "paradigms" not in config.keys():
            logger.error("{} must have a 'paradigms' key.".format(config))
            continue

        # iterate over paradigms

        for paradigm in config["paradigms"]:
            # check if it is in the context parameters file
            if len(context) > 0:
                if paradigm not in context.keys():
                    logger.debug(context)
                    logger.warning(
                        "Paradigm {} not in context file {}".format(
                            paradigm, context.keys()
                        )
                    )

            if isinstance(config["pipeline"], BaseEstimator):
                pipeline = deepcopy(config["pipeline"])
            else:
                logger.error(config["pipeline"])
                raise (ValueError("pipeline must be a sklearn estimator"))

            # append the pipeline in the paradigm list
            if paradigm not in paradigms.keys():
                paradigms[paradigm] = {}

            # FIXME name are not unique
            logger.debug("Pipeline: \n\n {} \n".format(get_string_rep(pipeline)))
            paradigms[paradigm][config["name"]] = pipeline

    return paradigms


def generate_param_grid(pipeline_configs, context=None, logger=log):
    context = context or {}
    param_grid = {}
    for config in pipeline_configs:
        if "paradigms" not in config:
            logger.error("{} must have a 'paradigms' key.".format(config))
            continue

        # iterate over paradigms
        if "param_grid" in config:
            param_grid[config["name"]] = config["param_grid"]

    return param_grid


class FilterBank(BaseEstimator, TransformerMixin):
    """Apply a given indentical pipeline over a bank of filter.

    The pipeline provided with the constrictor will be appield on the 4th
    axis of the input data. This pipeline should be used with a FilterBank
    paradigm.

    This can be used to build a filterbank CSP, for example::

        pipeline = make_pipeline(FilterBank(estimator=CSP()), LDA())

    Parameters
    ----------
    estimator: sklean Estimator
        the sklearn pipeline to apply on each band of the filter bank.
    flatten: bool (True)
        If True, output of each band are concatenated together on the feature
        axis. if False, output are stacked.
    """

    def __init__(self, estimator, flatten=True):
        self.estimator = estimator
        self.flatten = flatten

    def fit(self, X, y=None):
        assert X.ndim == 4
        self.models = [
            deepcopy(self.estimator).fit(X[..., i], y) for i in range(X.shape[-1])
        ]
        return self

    def transform(self, X):
        assert X.ndim == 4
        out = [self.models[i].transform(X[..., i]) for i in range(X.shape[-1])]
        assert out[0].ndim == 2, (
            "Each band must return a two dimensional "
            f" matrix, currently have {out[0].ndim}"
        )
        if self.flatten:
            return np.concatenate(out, axis=1)
        else:
            return np.stack(out, axis=2)

    def __repr__(self):
        estimator_name = type(self).__name__
        estimator_prms = self.estimator.get_params()
        return "{}(estimator={}, flatten={})".format(
            estimator_name, estimator_prms, self.flatten
        )


def filterbank(X, sfreq, idx_fb, peaks):
    """
    Filter bank design for decomposing EEG data into sub-band components [1]_

    Parameters
    ----------
    X: ndarray of shape (n_trials, n_channels, n_samples) or (n_channels, n_samples)
        EEG data to be processed

    sfreq: int
        Sampling frequency of the data.

    idx_fb: int
        Index of filters in filter bank analysis

    peaks : list of len (n_classes)
        Frequencies corresponding to the SSVEP components.

    Returns
    -------
    y: ndarray of shape (n_trials, n_channels, n_samples)
        Sub-band components decomposed by a filter bank

    Reference:
      .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using
           task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.

    Code based on the Matlab implementation from authors of [1]_
    (https://github.com/mnakanishi/TRCA-SSVEP).
    """

    # Calibration data comes in batches of trials
    if X.ndim == 3:
        num_chans = X.shape[1]
        num_trials = X.shape[0]

    # Testdata come with only one trial at the time
    elif X.ndim == 2:
        num_chans = X.shape[0]
        num_trials = 1

    sfreq = sfreq / 2

    min_freq = np.min(peaks)
    max_freq = np.max(peaks)

    if max_freq < 40:
        top = 100
    else:
        top = 115
    # Check for Nyquist
    if top >= sfreq:
        top = sfreq - 10

    diff = max_freq - min_freq
    # Lowcut frequencies for the pass band (depends on the frequencies of SSVEP)
    # No more than 3dB loss in the passband

    passband = [min_freq - 2 + x * diff for x in range(7)]

    # At least 40db attenuation in the stopband
    if min_freq - 4 > 0:
        stopband = [
            min_freq - 4 + x * (diff - 2) if x < 3 else min_freq - 4 + x * diff
            for x in range(7)
        ]
    else:
        stopband = [2 + x * (diff - 2) if x < 3 else 2 + x * diff for x in range(7)]

    Wp = [passband[idx_fb] / sfreq, top / sfreq]
    Ws = [stopband[idx_fb] / sfreq, (top + 7) / sfreq]

    N, Wn = scp.cheb1ord(Wp, Ws, 3, 40)  # Chebyshev type I filter order selection.

    B, A = scp.cheby1(N, 0.5, Wn, btype="bandpass")  # Chebyshev type I filter design

    y = np.zeros(X.shape)
    if num_trials == 1:  # For testdata
        for ch_i in range(num_chans):
            try:
                # The arguments 'axis=0, padtype='odd', padlen=3*(max(len(B),len(A))-1)' correspond
                # to Matlab filtfilt (https://dsp.stackexchange.com/a/47945)
                y[ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    X[ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
            except Exception as e:
                print(e)
                print(num_chans)
    else:
        for trial_i in range(num_trials):  # Filter each trial sequentially
            for ch_i in range(num_chans):  # Filter each channel sequentially
                y[trial_i, ch_i, :] = scp.filtfilt(
                    B,
                    A,
                    X[trial_i, ch_i, :],
                    axis=0,
                    padtype="odd",
                    padlen=3 * (max(len(B), len(A)) - 1),
                )
    return y
