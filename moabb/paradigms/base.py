from abc import ABC, abstractproperty, abstractmethod
import numpy as np
import pandas as pd


class BaseParadigm(ABC):
    """Base Paradigm.
    """

    def __init__(self):
        """init"""
        pass

    @abstractproperty
    def scoring(self):
        '''Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        '''
        pass

    @abstractproperty
    def datasets(self):
        '''Property that define the list of compatible datasets

        '''
        pass

    @abstractmethod
    def verify(self, dataset):
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

    @abstractmethod
    def process_raw(self, raw, dataset):
        """
        Process one raw data file.

        This function is apply the preprocessing and eventual epoching on the
        individual run, and return the data, labels and a dataframe with
        metadata.

        metadata is a dataframe with as many row as the length of the data
        and labels.

        returns
        -------
        X : np.ndarray
            the data that will be used as features for the model
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata

        """
        pass

    def get_data(self, dataset, subjects=None):
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

        returns
        -------
        X : np.ndarray
            the data that will be used as features for the model
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata.
        """

        self.verify(dataset)

        data = dataset.get_data(subjects)

        X = []
        labels = []
        metadata = []
        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    proc = self.process_raw(raw, dataset)

                    if proc is None:
                        # this mean the run did not contain any selected event
                        # go to next
                        continue

                    x, lbs, met = proc
                    met['subject'] = subject
                    met['session'] = session
                    met['run'] = run
                    metadata.append(met)

                    # grow X and labels in a memory efficient way. can be slow
                    if len(X) > 0:
                        X = np.append(X, x, axis=0)
                        labels = np.append(labels, lbs, axis=0)
                    else:
                        X = x
                        labels = lbs

        metadata = pd.concat(metadata, ignore_index=True)
        return X, labels, metadata
