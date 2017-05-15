import pandas as pd
from time import time

class BaseContext():

    def __init__(self):
        pass

    def evaluate(self, pipelines, verbose=False):
        columns = ['Score', 'Dataset', 'Subject', 'Pipeline', 'Time']
        results = dict()
        for pipeline in pipelines:
            results[pipeline] = pd.DataFrame(columns=columns)

        dataset_name = self.dataset.get_name()
        subjects = self.dataset.get_subject_list()

        for subject in subjects:
            X, y, groups = self.prepare_data([subject])

            for pipeline in pipelines:
                clf = pipelines[pipeline]
                t_start = time()
                score = self.score(clf, X=X, y=y, groups=groups)
                duration = time() - t_start
                row = [score, dataset_name, subject, pipeline, duration]
                results[pipeline].loc[len(results[pipeline])] = row
                if verbose:
                    print(row)
        return results
