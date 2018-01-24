from moabb.contexts.motor_imagery import MotorImageryTwoClasses

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier, MDM
from sklearn.pipeline import make_pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from moabb.datasets.bnci import (BNCI2014001, BNCI2014002,
                                 BNCI2014004, BNCI2015001, BNCI2015004)

from moabb.datasets.alex_mi import AlexMI
# from moabb.datasets.bbci_eeg_fnirs import BBCIEEGfNIRS
from moabb.datasets.gigadb import GigaDbMI
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.openvibe_mi import OpenvibeMI
pi = PhysionetMI()
pi.subject_list = range(1,5)
gb = GigaDbMI()
gb.subject_list = range(1,3)

datasets = [AlexMI(), OpenvibeMI(), BNCI2015004(motor_imagery=True), gb, pi]

pipelines = OrderedDict()
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM())
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())
# pipelines['CSP+SVM'] = make_pipeline(Covariances('oas'), CSP(8), SVC())

context = MotorImageryTwoClasses(datasets=datasets, pipelines=pipelines)

results = context.evaluate(verbose=True)

for p in results.keys():
    results[p].to_csv('../../results/MotorImagery/TwoClass/%s.csv2' % p)

results = pd.concat(results.values())
print(results.groupby('Pipeline').mean())

res = results.pivot(values='Score', columns='Pipeline')
sns.lmplot(data=res, x='CSP+LDA', y='TS', fit_reg=False)
plt.xlim(0.4, 1)
plt.ylim(0.4, 1)
plt.plot([0.4, 1], [0.4, 1], ls='--', c='k')
plt.show()
