import pandas as pd
from time import time

from sklearn.base import BaseEstimator

from ..datasets.base import BaseDataset


class BaseContext():
    """Base Context.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.
    """

    def __init__(self, datasets, pipelines):
        """init"""
        # check dataset
        if not isinstance(datasets, list):
            if isinstance(datasets, BaseDataset):
                datasets = [datasets]
            else:
                raise(ValueError("datasets must be a list or a dataset instance"))

        for dataset in datasets:
            if not(isinstance(dataset, BaseDataset)):
                raise(ValueError("datasets must only contains dataset instance"))

        self.datasets = datasets

        # check pipelines
        if not isinstance(pipelines, dict):
            raise(ValueError("pipelines must be a dict or a Pipeline instance"))

        for name, pipeline in pipelines.items():
            if not(isinstance(pipeline, BaseEstimator)):
                raise(ValueError("pipelines must only contains Pipelines instance"))
        self.pipelines = pipelines


class WithinSubjectContext(BaseContext):
    """Within Subject evaluation Context.

    Evaluate performance of the pipeline on each subject independently.

    Parameters
    ----------
    datasets : List of Dataset instances.
        List of dataset instances on which the pipelines will be evaluated.
    pipelines : Dict of pipelines instances.
        Dictionary of pipelines. Keys identifies pipeline names, and values
        are scikit-learn pipelines instances.

    See Also
    --------
    BaseContext
    """

    def evaluate(self, verbose=False):
        """Evaluate performances

        Parameters
        ----------
        verbose: bool (defaul False)
            if true, print results durint the evaluation

        Returns
        -------
        results: Dict of panda DataFrame
            Return a dict of pandas dataframe, one for each pipeline

        """
        columns = ['Score', 'Dataset', 'Subject', 'Pipeline', 'Time']
        results = dict()
        for pipeline in self.pipelines:
            results[pipeline] = pd.DataFrame(columns=columns)

        for dataset in self.datasets:
            dataset_name = dataset.get_name()
            subjects = dataset.get_subject_list()

            for subject in subjects:
                X, y, groups = self.prepare_data(dataset, [subject])

                for pipeline in self.pipelines:
                    clf = self.pipelines[pipeline]
                    t_start = time()
                    score = self.score(clf, X=X, y=y, groups=groups)
                    duration = time() - t_start
                    row = [score, dataset_name, subject, pipeline, duration]
                    results[pipeline].loc[len(results[pipeline])] = row
                    if verbose:
                        print(row)
        return results
