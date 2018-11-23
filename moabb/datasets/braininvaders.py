import mne
from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl
import os
import glob
import zipfile
import yaml

BI2013a_URL = 'https://zenodo.org/record/1494240/files/'


class bi2013a(BaseDataset):
    '''P300 dataset bi2013a from a "Brain Invaders" experiment (2013)
    carried-out at University of Grenoble Alpes.

    Dataset following the setup from [1]_.

    **Dataset Description**

    This dataset concerns an experiment carried out at GIPSA-lab
    (University of Grenoble Alpes, CNRS, Grenoble-INP) in 2013.
    Principal Investigators: Erwan Vaineau, Dr. Alexandre Barachant
    Scientific Supervisor :  Dr. Marco Congedo
    Technical Supervisor : Anton Andreev

    The experiment uses the Brain Invaders P300-based Brain-Computer Interface
    [7], which uses the Open-ViBE platform for on-line EEG data acquisition and
    processing [1, 9]. For classification purposes the Brain Invaders
    implements on-line Riemannian MDM classifiers [2, 3, 4, 6]. This experiment
    features both a training-test (classical) mode of operation and a
    calibration-less mode of operation [4, 5, 6].

    The recordings concerned 24 subjects in total. Subjects 1 to 7 participated
    to eight sessions, run in different days, subject 8 to 24 participated to
    one session. Each session consisted in two runs, one in a Non-Adaptive
    (classical) and one in an Adaptive (calibration-less) mode of operation.
    The order of the runs was randomized for each session. In both runs there
    was a Training (calibration) phase and an Online phase, always passed in
    this order. In the non-Adaptive run the data from the Training phase was
    used for classifying the trials on the Online phase using the training-test
    version of the MDM algorithm [3, 4]. In the Adaptive run, the data from the
    training phase was not used at all, instead the classifier was initialized
    with generic class geometric means and continuously adapted to the incoming
    data using the Riemannian method explained in [4]. Subjects were completely
    blind to the mode of operation and the two runs appeared to them identical.

    In the Brain Invaders P300 paradigm, a repetition is composed of 12
    flashes, of which 2 include the Target symbol (Target flashes) and 10 do
    not (non-Target flash). Please see [7] for a description of the paradigm.
    For this experiment, in the Training phases the number of flashes is fixed
    (80 Target flashes and 400 non-Target flashes). In the Online phases the
    number of Target and non-Target still are in a ratio 1/5, however their
    number is variable because the Brain Invaders works with a fixed number of
    game levels, however the number of repetitions needed to destroy the target
    (hence to proceed to the next level) depends on the user’s performance
    [4, 5]. In any case, since the classes are unbalanced, an appropriate score
    must be used for quantifying the performance of classification methods
    (e.g., balanced accuracy, AUC methods, etc).

    Data were acquired with a Nexus (TMSi, The Netherlands) EEG amplifier:
    - Sampling Frequency: 512 samples per second
    - Digital Filter: no
    - Electrodes:  16 wet Silver/Silver Chloride electrodes positioned at
      FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2
      according to the 10/20 international system.
    - Reference: left ear-lobe.
    - Ground: N/A.

    References
    ----------

    .. [1] Arrouët C, Congedo M, Marvie J-E, Lamarche F, Lècuyer A, Arnaldi B
           (2005) Open-ViBE: a 3D Platform for Real-Time Neuroscience.
           Journal of Neurotherapy, 9(1), 3-25.
    .. [2] Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of
           covariance matrices using a Riemannian-based kernel for BCI
           applications. Neurocomputing 112, 172-178.
    .. [3] Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multi-Class Brain
           Computer Interface, Classification by Riemannian Geometry.
           IEEE Transactions on Biomedical Engineering 59(4), 920-928
    .. [4] Barachant A, Congedo M (2014) A Plug & Play P300 BCI using
           Information Geometry.
           arXiv:1409.0107.
    .. [5] Congedo M, Barachant A, Andreev A (2013) A New Generation of
           Brain-Computer Interface Based on Riemannian Geometry.
           arXiv:1310.8115.
    .. [6] Congedo M, Barachant A, Bhatia R (2017) Riemannian Geometry for
           EEG-based Brain-Computer Interfaces; a Primer and a Review.
           Brain-Computer Interfaces, 4(3), 155-174.
    .. [7] Congedo M, Goyat M, Tarrin N, Ionescu G, Rivet B,Varnet L, Rivet B,
           Phlypo R, Jrad N, Acquadro M, Jutten C (2011) “Brain Invaders”: a
           prototype of an open-source P300-based video game working with the
           OpenViBE platform. Proc. IBCI Conf., Graz, Austria, 280-283.
    .. [8] Congedo M, Korczowski L, Delorme A, Lopes da Silva F. (2016)
           Spatio-temporal common pattern: A companion method for ERP analysis
           in the time domain. Journal of Neuroscience Methods, 267, 74-88.
    .. [9] Renard Y, Lotte F, Gibert G, Congedo M, Maby E, Delannoy V, Bertrand
           O, Lécuyer A (2010) OpenViBE: An Open-Source Software Platform to
           Design, Test and Use Brain-Computer Interfaces in Real and Virtual
           Environments. PRESENCE : Teleoperators and Virtual Environments
           19(1), 35-53.
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
