from moabb.contexts.bnci_2014_001 import BNCI2014001MIHands

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier, MDM
from sklearn.pipeline import make_pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict

context = BNCI2014001MIHands()

pipelines = OrderedDict()
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM())
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())

results = context.evaluate(pipelines, verbose=True)

for p in results.keys():
    results[p].to_csv('../../results/MotorImagery/BNCI_2014_001/%s.csv' % p)

results = pd.concat(results.values())
print(results.groupby('Pipeline').mean())

res = results.pivot(values='Score', columns='Pipeline', index='Subject')
sns.lmplot(data=res, x='CSP+LDA', y='TS', fit_reg=False)
plt.xlim(0.4, 1)
plt.ylim(0.4, 1)
plt.plot([0.4, 1], [0.4, 1], ls='--', c='k')
plt.show()
