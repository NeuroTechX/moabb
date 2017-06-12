from moabb.contexts.motor_imagery import MotorImageryMultiClasses

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier, MDM
from sklearn.pipeline import make_pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from moabb.datasets.bnci import BNCI2014001

from moabb.datasets.alex_mi import AlexMI
from moabb.datasets.physionet_mi import PhysionetMI

datasets = [AlexMI(with_rest=True),
            BNCI2014001(),
            PhysionetMI(with_rest=True, feets=False)]

pipelines = OrderedDict()
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM())
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())

context = MotorImageryMultiClasses(datasets=datasets, pipelines=pipelines)

results = context.evaluate(verbose=True)

for p in results.keys():
    results[p].to_csv('../../results/MotorImagery/MultiClass/%s.csv' % p)

results = pd.concat(results.values())
print(results.groupby('Pipeline').mean())

res = results.pivot(values='Score', columns='Pipeline')
sns.lmplot(data=res, x='CSP+LDA', y='TS', fit_reg=False)
plt.xlim(0.25, 1)
plt.ylim(0.25, 1)
plt.plot([0.25, 1], [0.25, 1], ls='--', c='k')
plt.show()
