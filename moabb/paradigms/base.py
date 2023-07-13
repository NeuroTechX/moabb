import logging
from abc import ABCMeta, abstractmethod
from operator import methodcaller

import mne
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from moabb.datasets.preprocessing import (
    RawToEpochs,
    get_crop_pipeline,
    get_filter_pipeline,
    get_resample_pipeline,
)


log = logging.getLogger(__name__)


class BaseParadigm(metaclass=ABCMeta):
    """Base Paradigm."""

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def scoring(self):
        """Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        """
        pass

    @property
    @abstractmethod
    def datasets(self):
        """Property that define the list of compatible datasets"""
        pass

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an ERP dataset for motor imagery paradigm, or if the
        dataset does not contain any of the required events.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """
        pass

    def prepare_process(self, dataset):
        """Prepare processing of raw files

                This function allows to set parameter of the paradigm class prior to
                the preprocessing (process_raw). Does nothing by default and could be
                overloaded if needed.

                Parameters
                ----------

                dataset : dataset instance
                    The dataset corresponding to the raw file. mainly use to access
                    dataset specific i
        nformation.
        """
        if dataset is not None:
            pass

    def get_data(
        self,
        dataset,
        subjects=None,
        return_epochs=False,
        return_raws=False,
        processing_pipeline=None,
        cache_config=None,
    ):
        """
        Return the data for a list of subject.

        return the data, labels and a dataframe with metadata. the dataframe
        will contain at least the following columns

        - subject : the subject indice
        - session : the session indice
        - run : the run indice

        parameters
        ----------
        dataset:
            A dataset instance.
        subjects: List of int
            List of subject number
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs
        return_raws: boolean
            To return raw files and events, to ensure compatibility with braindecode.
            Mutually exclusive with return_epochs
        cache_config: dict
            Configuration for caching of datasets. See moabb.datasets.base.CacheConfig for details.

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
            if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata.
        """

        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)

        if return_epochs and return_raws:
            message = "Select only return_epochs or return_raws, not both"
            raise ValueError(message)

        if subjects is None:
            subjects = dataset.subject_list

        self.prepare_process(dataset)
        raw_pipelines = self._get_raw_pipelines()
        epochs_pipeline = self._get_epochs_pipeline(return_raws)
        array_pipeline = self._get_array_pipeline(
            return_epochs, return_raws, processing_pipeline, dataset
        )
        data = [
            dataset.get_data(
                subjects=subjects,
                cache_config=cache_config,
                raw_pipeline=raw_pipeline,
                epochs_pipeline=epochs_pipeline,
                array_pipeline=array_pipeline,
            )
            for raw_pipeline in raw_pipelines
        ]

        data = {
            subject: {
                session: {
                    run: [data_i[subject][session][run] for data_i in data]
                    for run in runs.keys()
                }
                for session, runs in sessions.items()
            }
            for subject, sessions in data[0].items()
        }

        X = [] if (return_epochs or return_raws) else np.array([])
        labels = []
        metadata = []
        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raws in runs.items():
                    proc = self.process_raws(raws, dataset, return_epochs, return_raws)
                    # TODO: get the labels and metadata
                    if proc is None:
                        # this mean the run did not contain any selected event
                        # go to next
                        continue

                    x, lbs, met = proc
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    metadata.append(met)

                    # grow X and labels in a memory efficient way. can be slow
                    if return_epochs:
                        x.metadata = (
                            met.copy()
                            if len(self.filters) == 1
                            else pd.concat(
                                [met.copy()] * len(self.filters), ignore_index=True
                            )
                        )
                        X.append(x)
                    elif return_raws:
                        X.append(x)
                    else:
                        X = np.append(X, x, axis=0) if len(X) else x
                    labels = np.append(labels, lbs, axis=0)

        metadata = pd.concat(metadata, ignore_index=True)
        if return_epochs:
            X = mne.concatenate_epochs(X)
        return X, labels, metadata

    def _get_raw_pipelines(self):
        return [get_filter_pipeline(fmin, fmax) for fmin, fmax in self.filters]

    def _get_epochs_pipeline(self, return_raws, dataset):
        if return_raws:
            return None

        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        baseline = self.baseline
        if baseline is not None:
            baseline = (
                self.baseline[0] + dataset.interval[0],
                self.baseline[1] + dataset.interval[0],
            )
            bmin = baseline[0] if baseline[0] < tmin else tmin
            bmax = baseline[1] if baseline[1] > tmax else tmax
        else:
            bmin = tmin
            bmax = tmax
        steps = []
        steps.append(
            (
                "epoching",
                RawToEpochs(
                    event_id=self.events,
                    tmin=bmin,
                    tmax=bmax,
                    baseline=baseline,
                    channels=self.channels,
                ),
            )
        )
        if bmin < tmin or bmax > tmax:
            steps.append(("crop", get_crop_pipeline(tmin=tmin, tmax=tmax)))
        if self.resample is not None:
            steps.append(("resample", get_resample_pipeline(self.resample)))
        return Pipeline(steps)

    def _get_array_pipeline(
        self, return_epochs, return_raws, processing_pipeline, dataset
    ):
        steps = []
        if not return_epochs and not return_raws:
            steps.append(("get_data", FunctionTransformer(methodcaller("get_data"))))
            steps.append(
                (
                    "scaling",
                    FunctionTransformer(
                        np.multiply, kw_args=dict(x1=dataset.scaling_factor)
                    ),
                )
            )
        if processing_pipeline is not None:
            steps.append(("processing_pipeline", processing_pipeline))
        if len(steps) == 0:
            return None, ColumnTransformer
        return Pipeline(steps)
