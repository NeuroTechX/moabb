from abc import ABC
import pandas as pd


class Results(ABC):
    '''Class to hold results from the evaluation.evaluate method. Appropriate test
    would be to ensure the result of 'evaluate' is consistent and can be
    accepted by 'results.add'

    Saves dataframe per pipeline and can query to see if particular subject has
    already been run

    '''

    def __init__(self, evaluation, pipelines):
        """
        class that will abstract result storage
        """
        self.evaluation = evaluation
        self.data_columns = ['id', 'time', 'score',
                             'dataset', 'n_samples', 'n_channels']
        dfs = [[] for p in pipelines.keys()]
        self.data = dict(zip(pipelines.keys(), dfs))

    def add(self, data_dict, pipeline):
        if type(data_dict) is dict:
            data_dict = [data_dict]
        elif type(data_dict) is not list:
            raise ValueError('Results are given as neither dict nor list but {}'.format(
                type(data_dict).__name__))
        self.data[pipeline].extend(data_dict)

    def to_dataframe(self):
        for k in self.data.keys():
            df = pd.DataFrame.from_records(
                self.data[k], columns=self.data_columns)
            self.data[k] = df
