from collections import defaultdict
from warnings import warn
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.utils import tosequence
from sklearn.utils.metaestimators import if_delegate_has_method
import sklearn.pipeline as pipeline

class MultitaskPipeline(pipeline.Pipeline):
    """Class that extends the sklearn Pipeline to have the ability to take in lists
    of datasets and run transforms and classifiers that use offline data in a
    multi-task manner. This is done by calling pipeline.pre_fit, which then runs
    through the list of transforms and pre-fits them as they require, then
    pre-trains the classifier. It can then be used like a normal Pipeline, in
    that the fit method is stateless *given* that pre_fit has already been
    called. 

    Parameters
    ----------
    steps : list
        List of (name, pre_transform) tuples (implementing pre_fit/pre_transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    """

    # BaseEstimator interface
    @staticmethod
    def _name_estimators(estimators):
        """Generate names for estimators."""

        names = [type(estimator).__name__.lower() for estimator in estimators]
        namecount = defaultdict(int)
        for est, name in zip(estimators, names):
            namecount[name] += 1

        for k, v in list(six.iteritems(namecount)):
            if v == 1:
                del namecount[k]

        for i in reversed(range(len(estimators))):
            name = names[i]
            if name in namecount:
                names[i] += "-%d" % namecount[name]
                namecount[name] -= 1

        return list(zip(names, estimators))    

    def _pre_validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        pre_transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in pre_transformers:
            if t is None:
                continue
            if (not (hasattr(t, "pre_fit") or hasattr(t, "pre_fit_transform")) or not
                    hasattr(t, "pre_transform")):
                raise TypeError("All intermediate steps should be "
                                "pre_transformers and implement pre_fit and pre_transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity pre_transformation
        if estimator is not None and not hasattr(estimator, "pre_fit"):
            raise TypeError("Last step of Pipeline should implement pre_fit. "
                            "'%s' (type %s) doesn't"
                            % (estimator, type(estimator)))

    # Estimator interface

    def _pre_fit(self, X, y=None, **pre_fit_params):
        self._pre_validate_steps()
        pre_fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(pre_fit_params):
            step, param = pname.split('__', 1)
            pre_fit_params_steps[step][param] = pval
        Xt = X
        for name, pre_transform in self.steps[:-1]:
            if pre_transform is None:
                pass
            elif hasattr(pre_transform, "pre_fit_transform"):
                Xt = pre_transform.pre_fit_transform(Xt, y, **pre_fit_params_steps[name])
            else:
                Xt = pre_transform.pre_fit(Xt, y, **pre_fit_params_steps[name]) \
                              .pre_transform(Xt)
        if self._final_estimator is None:
            return Xt, {}
        return Xt, pre_fit_params_steps[self.steps[-1][0]]

    def pre_fit(self, X, y=None, **pre_fit_params):
        """Pre_Fit the model
        Pre_Fit all the pre_transforms one after the other and pre_transform the
        data, then pre_fit the pre_transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **pre_fit_params : dict of string -> object
            Parameters passed to the ``pre_fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, pre_fit_params = self._pre_fit(X, y, **pre_fit_params)
        if self._final_estimator is not None:
            self._final_estimator.pre_fit(Xt, y, **pre_fit_params)
        return self

    def pre_fit_transform(self, X, y=None, **pre_fit_params):
        """Pre_Fit the model and pre_transform with the final estimator
        Pre_Fits all the pre_transforms one after the other and pre_transforms the
        data, then uses pre_fit_pre_transform on pre_transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **pre_fit_params : dict of string -> object
            Parameters passed to the ``pre_fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_pre_transformed_features]
            Pre_Transformed samples
        """
        last_step = self._final_estimator
        Xt, pre_fit_params = self._pre_fit(X, y, **pre_fit_params)
        if hasattr(last_step, 'pre_fit_transform'):
            return last_step.pre_fit_transform(Xt, y, **pre_fit_params)
        elif last_step is None:
            return Xt
        else:
            return last_step.pre_fit(Xt, y, **pre_fit_params).pre_transform(Xt)

    @property
    def pre_transform(self):
        """Apply pre_transforms, and pre_transform with the final estimator
        This also works where final estimator is ``None``: all prior
        pre_transformations are applied.
        Parameters
        ----------
        X : iterable
            Data to pre_transform. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_pre_transformed_features]
        """
        # _final_estimator is None or has pre_transform, otherwise attribute error
        if self._final_estimator is not None:
            #...is this a bug? BUG
            self._final_estimator.pre_transform
        return self._pre_transform

    def _pre_transform(self, X):
        Xt = X
        for name, pre_transform in self.steps:
            if pre_transform is not None:
                Xt = pre_transform.pre_transform(Xt)
        return Xt

    @property
    def inverse_pre_transform(self):
        """Apply inverse pre_transformations in reverse order
        All estimators in the pipeline must support ``inverse_pre_transform``.
        Parameters
        ----------
        Xt : array-like, shape = [n_samples, n_pre_transformed_features]
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_pre_transform`` method.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_features]
        """
        # raise AttributeError if necessary for hasattr behaviour
        for name, pre_transform in self.steps:
            if pre_transform is not None:
                pre_transform.inverse_pre_transform
        return self._inverse_pre_transform

    def _inverse_pre_transform(self, X):
        if hasattr(X, 'ndim') and X.ndim == 1:
            warn("From version 0.19, a 1d X will not be reshaped in"
                 " pipeline.inverse_pre_transform any more.", FutureWarning)
            X = X[None, :]
        Xt = X
        for name, pre_transform in self.steps[::-1]:
            if pre_transform is not None:
                Xt = pre_transform.inverse_pre_transform(Xt)
        return Xt

    @staticmethod
    def make_pipeline(*steps):
        """Construct a Pipeline from the given estimators.
        This is a shorthand for the Pipeline constructor; it does not require, and
        does not permit, naming the estimators. Instead, their names will be set
        to the lowercase of their types automatically.
        Examples
        --------
        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.preprocessing import StandardScaler
        >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
        ...     # doctest: +NORMALIZE_WHITESPACE
        Pipeline(steps=[('standardscaler',
                         StandardScaler(copy=True, with_mean=True, with_std=True)),
                        ('gaussiannb', GaussianNB(priors=None))])
        Returns
        -------
        p : MultitaskPipeline
        """
        return MultitaskPipeline(MultitaskPipeline._name_estimators(steps))

