from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy


def create_pipeline_from_config(config):
    """Create a pipeline from a config file.

    takes a config dict as input and return the coresponding pipeline.

    Parameters
    ----------
    config : Dict.
        Dict containing the config parameters.

    Return
    ------
    pipeline : Pipeline
        sklearn Pipeline

    """
    components = []

    for component in config:
        # load the package
        mod = __import__(component['from'], fromlist=[component['name']])
        # create the instance
        if 'parameters' in component.keys():
            params = component['parameters']
        else:
            params = {}
        instance = getattr(mod, component['name'])(**params)
        components.append(instance)

    pipeline = make_pipeline(*components)
    return pipeline


class FilterBank(BaseEstimator, TransformerMixin):
    """Class that applies a given pipeline to multiple frequency bands from
    MotorImageryMultiPass.

    TODO: should this be forced to be a transformer?
    """

    def __init__(self, estimator, flatten=True):
        """
        estimator: BaseEstimator instance, applied on the 4th axis
        flatten: bool, whether to flatten output of estimator
        """
        assert issubclass(type(estimator), BaseEstimator)
        self.estimator = estimator
        self.flatten = flatten

    def fit(self, X, y=None):
        """fit"""
        assert X.ndim == 4
        self.models = [deepcopy(self.estimator).fit(X[..., i], y)
                       for i in range(X.shape[-1])]
        return self

    def transform(self, X):
        assert X.ndim == 4
        out = [self.models[i].transform(X[..., i]) for i in range(X.shape[-1])]
        assert out[0].ndim == 2, 'output is of dim {}'.format(out[0].ndim)
        if self.flatten:
            return np.concatenate(out, axis=1)
        else:
            return np.stack(out, axis=2)

    def __repr__(self):
        est = self.estimator.get_params()
        return '{}(estimator={}, flatten={})'.format(type(self).__name__,
                                                     est,
                                                     self.flatten)
