from abc import ABCMeta, abstractproperty, abstractmethod
import numpy as np
import pandas as pd
import mne


class BaseParadigm(metaclass=ABCMeta):
    """Base Paradigm.
    """

    def __init__(self):
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
            dataset specific information.
        """
        pass

    def process_raw(self, raw, dataset, baseline=None, return_epochs=False):
        """
        Process one raw data file.

        This function apply the preprocessing and eventual epoching on the
        individual run, and return the data, labels and a dataframe with
        metadata.

        metadata is a dataframe with as many row as the length of the data
        and labels.

        Parameters
        ----------
        raw: mne.Raw instance
            the raw EEG data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.          
        baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline correction.
            If None, do not apply baseline correction. 
            If a tuple (a, b), the interval is between a and b (in seconds), including the endpoints.
            Correction is applied by computing the mean of the baseline period and subtracting it from the data (see mne.Epochs)
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs

        returns
        -------
        X : Union[np.ndarray, mne.Epochs]
            the data that will be used as features for the model
            Note: if return_epochs=True,  this is mne.Epochs
                  if return_epochs=False, this is np.ndarray
        labels: np.ndarray
            the labels for training / evaluating the model
        metadata: pd.DataFrame
            A dataframe containing the metadata

        """
        # get events id
        event_id = self.used_events(dataset)

        # find the events, first check stim_channels then annotations
        stim_channels = mne.utils._get_stim_channel(None, raw.info,
                                                    raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            try:
                events, _ = mne.events_from_annotations(raw,
                                                        event_id=event_id,
                                                        verbose=False)
            except ValueError:
                events, _ = mne.events_from_annotations(raw, verbose=False)

        # picks channels
        if self.channels is None:
            picks = mne.pick_types(raw.info, eeg=True, stim=False)
        else:
            picks = mne.pick_types(raw.info, stim=False, include=self.channels)

        # pick events, based on event_id
        try:
            events = mne.pick_events(events, include=list(event_id.values()))
        except RuntimeError:
            # skip raw if no event found
            return

        # get interval
        tmin = self.tmin + dataset.interval[0]
        if self.tmax is None:
            tmax = dataset.interval[1]
        else:
            tmax = self.tmax + dataset.interval[0]

        X = []
        for bandpass in self.filters:
            fmin, fmax = bandpass
            # filter data
            raw_f = raw.copy().filter(fmin, fmax, method='iir',
                                      picks=picks, verbose=False)
            # epoch data
            if baseline is not None:
                bmin = baseline[0] if baseline[0] < tmin else tmin
                bmax = baseline[1] if baseline[1] > tmax else tmax
            else:
                bmin = tmin
                bmax = tmax
            epochs = mne.Epochs(raw_f, events, event_id=event_id, proj=False,
                                tmin=baseline[0], tmax=dataset.interval[1],
                                baseline=baseline, preload=True,
                                verbose=False, picks=picks,
                                event_repeated='drop',
                                on_missing='ignore')
            if bmin < tmin or bmax > tmax:
                epochs.crop(tmin=tmin, tmax=tmax)
            if self.resample is not None:
                epochs = epochs.resample(self.resample)
            # rescale to work with uV
            if return_epochs:
                X.append(epochs)
            else:
                X.append(dataset.unit_factor * epochs.get_data())

        inv_events = {k: v for v, k in event_id.items()}
        labels = np.array([inv_events[e] for e in epochs.events[:, -1]])

        # if only one band, return a 3D array, otherwise return a 4D
        if len(self.filters) == 1:
            X = X[0]
        else:
            X = np.array(X).transpose((1, 2, 3, 0))

        metadata = pd.DataFrame(index=range(len(labels)))
        return X, labels, metadata

    def get_data(self, dataset, subjects=None, baseline=None,
                 return_epochs=False):
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
        baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline correction.
            If None, do not apply baseline correction. 
            If a tuple (a, b), the interval is between a and b (in seconds), including the endpoints.
            Correction is applied by computing the mean of the baseline period and subtracting it from the data (see mne.Epochs)
        return_epochs: boolean
            This flag specifies whether to return only the data array or the
            complete processed mne.Epochs

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
            message = "Dataset {} is not valid for paradigm".format(
                dataset.code)
            raise AssertionError(message)

        data = dataset.get_data(subjects)
        self.prepare_process(dataset)

        X = []
        labels = []
        metadata = []
        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    proc = self.process_raw(raw, dataset, baseline=baseline,
                                            return_epochs=return_epochs)

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
                    if not return_epochs:
                        if len(X) > 0:
                            X = np.append(X, x, axis=0)
                            labels = np.append(labels, lbs, axis=0)
                        else:
                            X = x
                            labels = lbs
                    else:
                        X.append(x)

        metadata = pd.concat(metadata, ignore_index=True)
        return X, labels, metadata
