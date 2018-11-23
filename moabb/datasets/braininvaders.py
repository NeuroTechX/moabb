import mne
from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl
import os
import glob
import zipfile
import yaml

BI2013a_URL = 'https://zenodo.org/record/1494240/files/'


class BrainInvaders2013a(BaseDataset):
    '''P300 dataset from a Brain Invaders experiment in 2013.

    Dataset following the setup from [1]_.

    **Dataset Description**

    This dataset concerns an experiment carried out at GIPSA-lab in Grenoble
    using the Brain Invaders P300-based Brain-Computer Interface [1].
    The recordings were done on 24 subjects in total. Subjects 1 to 7
    participated in 8 sessions and the others only did one session.

    Each session consisted in a NonAdaptive and an Adaptive experimental
    condition, which had a Training (calibration) phase and an Online phase.
    For both conditions, the data from the Training phase was used for
    classifying the trials on the Online phase, which served as a feedback for
    the user. However, in the Adaptive experimental condition, the new trials
    available during the Online phase were also used for the classification.

    Because of the experimental setup of the Brain Invaders paradigm, there
    is 1 Target trial to 5 NonTarget trials (i.e., the classes are unbalanced).
    This means that for quantifying the performance of a classification method
    using this dataset one should prefer the AUC score instead of accuracy.

    Data were acquired with a sampling frequency of 512 Hz and using
    16 electrodes at positions FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz,
    P4, P8, O1, Oz, O2 according to the 10/20 international system, using as
    reference the left ear-lobe.

    The reader may check Ref.[2] for an example of study using this dataset.

    References
    ----------

    .. [1] Congedo, M., Goyat, M., Tarrin, N., Ionescu, G., Rivet, B.,
           Varnet, L., Rivet, B., Phlypo, R., Jrad, N., Acquadro, M.,
           Jutten, C., Brain Invaders : a prototype of an open-source
           P300-based video game working with the OpenViBE.
           Proc. IBCI Conf., Graz, Austria, 280-283.
           https://doi.org/10.1016/j.jneumeth.2007.03.005

    .. [2] Barachant, A., Congedo, M., A Plug & Play P300 BCI using Information
           Geometry, 2014, Available at arXiv (ref. arXiv:1409.0107)
    '''

    def __init__(
            self,
            NonAdaptive=True,
            Adaptive=False,
            Training=True,
            Online=False):
        super().__init__(
            subjects=list(range(1, 24 + 1)),
            sessions_per_subject='varying',
            events=dict(Target=33285, NonTarget=33286),
            code='Brain Invaders 2013a',
            interval=[0, 1],
            paradigm='p300',
            doi='')

        self.adaptive = Adaptive
        self.nonadaptive = NonAdaptive
        self.training = Training
        self.online = Online

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""

        file_path_list = self.data_path(subject)
        sessions = {}
        for file_path in file_path_list:

            session_number = file_path.split('/')[-2].strip('Session')
            session_name = 'session_' + session_number
            if session_name not in sessions.keys():
                sessions[session_name] = {}

            run_number = file_path.split('/')[-1]
            run_number = run_number.split('_')[-1]
            run_number = run_number.split('.gdf')[0]
            run_name = 'run_' + run_number

            raw_original = mne.io.read_raw_edf(file_path,
                                               montage='standard_1020',
                                               preload=True)

            sessions[session_name][run_name] = raw_original

        return sessions

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = '{:s}subject{:d}.zip'.format(BI2013a_URL, subject)
        path_zip = dl.data_path(url, 'BRAININVADERS')
        path_folder = path_zip.strip('subject{:d}.zip'.format(subject))

        # check if has to unzip
        if not(os.path.isdir(path_folder + 'subject{:d}/'.format(subject))):
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        # filter the data regarding the experimental conditions
        meta_file = 'subject{:d}/meta.yml'.format(subject)
        meta_path = path_folder + meta_file
        with open(meta_path, 'r') as stream:
            meta = yaml.load(stream)
        conditions = []
        if self.adaptive:
            conditions = conditions + ['adaptive']
        if self.nonadaptive:
            conditions = conditions + ['nonadaptive']
        types = []
        if self.training:
            types = types + ['training']
        if self.online:
            types = types + ['online']
        filenames = []
        for run in meta['runs']:
            run_condition = run['experimental_condition']
            run_type = run['type']
            if (run_condition in conditions) and (run_type in types):
                filenames = filenames + [run['filename']]

        # list the filepaths for this subject
        subject_paths = []
        for filename in filenames:
            subject_paths = subject_paths + \
                glob.glob(path_folder + 'subject{:d}/Session*/'.format(subject) + filename) # noqa

        return subject_paths
