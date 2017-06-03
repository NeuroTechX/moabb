from moabb.contexts.gigadb_mi import GigaDbMI2Class

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.classification import TSclassifier, MDM
from sklearn.pipeline import make_pipeline

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict

context = GigaDbMI2Class()
# 32 and 46, 49 have trouble
a = list(range(1, 53))
[a.remove(ii) for ii in [32, 46, 49]]
context.dataset.subject_list = a

pipelines = OrderedDict()
pipelines['MDM'] = make_pipeline(Covariances('oas'), MDM())
pipelines['TS'] = make_pipeline(Covariances('oas'), TSclassifier())
pipelines['CSP+LDA'] = make_pipeline(Covariances('oas'), CSP(8), LDA())

results = context.evaluate(pipelines, verbose=True)

for p in results.keys():
    results[p].to_csv('../../results/MotorImagery/gigadb_mi/%s.csv' % p)

results = pd.concat(results.values())
print(results.groupby('Pipeline').mean())

res = results.pivot(values='Score', columns='Pipeline', index='Subject')
sns.lmplot(data=res, x='CSP+LDA', y='TS', fit_reg=False)
plt.xlim(0.4, 1)
plt.ylim(0.4, 1)
plt.plot([0.4, 1], [0.4, 1], ls='--', c='k')
plt.show()
