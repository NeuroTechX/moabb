"""
===========================
Creating a dataset class in MOABB
===========================
"""
# Authors: Pedro L. C. Rodrigues, Marco Congedo
# Sylvain Chevallier
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

import matplotlib.pyplot as plt
import numpy as np
import pyriemann
from scipy.io import savemat, loadmat
import mne

from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

from sklearn.pipeline import make_pipeline

##############################################################################
# Dataset class creation
# ----------------------
# 
# To illustrate the creation of a dataset class in MOABB, we first create an 
# example dataset saved in .mat file. It contains a single fake recording on 
# 8 channels lasting for 150 seconds (sampling frequency 256 Hz). We've included
# the script that creates this dataset and have uploaded it online. It is 
# available at the Zenodo website on the link : https://sandbox.zenodo.org/record/369543


def create_example_dataset():

    fsample = 256
    Tsample = 1.0/fsample
    Trecording = 150
    Ttrial = 1 # duration of a trial
    intertrial = 2 # time between finishing one trial and beginning another one
    Nchannels = 8

    x = np.zeros((Nchannels+1, Trecording * fsample)) # electrodes + stimulus
    stim = np.zeros(Trecording * fsample)
    toffset = 1.0 # offset where the trials start
    Ntrials = 40

    signal = np.sin(2 * np.pi / Ttrial * np.linspace(0, 4 * Ttrial, Ttrial * fsample))
    for n in range(Ntrials):
        label = n % 2 + 1 # alternate between class 0 and class 1
        tn = int(toffset * fsample + n * (Ttrial+intertrial) * fsample)
        stim[tn] = label
        noise = 0.1 * np.random.randn(Nchannels, len(signal))
        x[:-1, tn:(tn+Ttrial*fsample)] = label * signal + noise
    x[-1,:] = stim    
    
    return x, fsample

for subject in [1, 2, 3]:
    
    x, fs = create_example_dataset()
    filename = 'subject_' + str(subject).zfill(2) + '.mat'
    mdict = {}
    mdict['x'] = x
    mdict['fs'] = fs
    savemat(filename, mdict)


ExampleDataset_URL = 'https://sandbox.zenodo.org/record/369543/files/'

class ExampleDataset(BaseDataset):
    
    '''
    Dataset used to exemplify the creation of a dataset class in MOABB. 
    The data samples have been simulated and has no physiological meaning whatsoever.
    '''
    
    def __init__(self):
        super().__init__(
            subjects=[1, 2, 3],
            sessions_per_subject=1,
            events={'left_hand':1, 'right_hand':2},
            code='Example dataset',
            interval=[0, 0.75],
            paradigm='imagery',
            doi='')

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        
        data = loadmat(file_path_list[0])
        x = data['x']
        fs = data['fs']
        ch_names = ['ch' + str(i) for i in range(8)] + ['stim'] 
        ch_types = ['eeg' for i in range(8)] + ['stim']
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(x, info)
        
        sessions = {}
        sessions['session_1'] = {}
        sessions['session_1']['run_1'] = raw

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        url = '{:s}subject_0{:d}.mat'.format(ExampleDataset_URL, subject)
        path = dl.data_path(url, 'ExampleDataset')
        
        return [path] # it has to return a list    
    
dataset = ExampleDataset()

paradigm = LeftRightImagery()
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=dataset, overwrite=True)
pipelines = {}
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM(metric='riemann'))
scores = evaluation.process(pipelines)

print(scores)