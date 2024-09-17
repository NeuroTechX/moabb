import logging
from abc import ABC, abstractmethod
from warnings import warn

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from moabb.analysis import Results
from moabb.datasets.base import BaseDataset
from moabb.evaluations.utils import _convert_sklearn_params_to_optuna
from moabb.paradigms.base import BaseParadigm


log = logging.getLogger(__name__)

# Making the optuna soft dependency
try:
    from optuna.integration import OptunaSearchCV

    optuna_available = True
except ImportError:
    optuna_available = False

if optuna_available:
    search_methods = {"grid": GridSearchCV, "optuna": OptunaSearchCV}
else:
    search_methods = {"grid": GridSearchCV}


class BaseEvaluation(ABC):
    """Base class that defines necessary operations for an evaluation.
    Evaluations determine what the train and test sets are and can implement
    additional data preprocessing steps for more complicated algorithms.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        ‘raise’, the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    return_raws: bool, default=False
        use MNE raw to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    n_splits: int, default=None
        Number of splits for cross-validation. If None, the number of splits
        is equal to the number of subjects.
    save_model: bool, default=False
        Save model after training, for each fold of cross-validation if needed
    cache_config: bool, default=None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.
    optuna:bool, default=False
        If optuna is enable it will change the GridSearch to a RandomizedGridSearch with 15 minutes of cut off time.
        This option is compatible with list of entries of type None, bool, int, float and string
    time_out: default=60*15
        Cut off time for the optuna search expressed in seconds, the default value is 15 minutes.
        Only used with optuna equal to True.

    Notes
    -----
    .. versionadded:: 1.1.0
       n_splits, save_model, cache_config parameters.
    .. versionadded:: 1.1.1
       optuna, time_out parameters.
    """

    def __init__(
        self,
        paradigm,
        datasets=None,
        random_state=None,
        n_jobs=1,
        overwrite=False,
        error_score="raise",
        suffix="",
        hdf5_path=None,
        additional_columns=None,
        return_epochs=False,
        return_raws=False,
        mne_labels=False,
        n_splits=None,
        save_model=False,
        cache_config=None,
        optuna=False,
        time_out=60 * 15,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.error_score = error_score
        self.hdf5_path = hdf5_path
        self.return_epochs = return_epochs
        self.return_raws = return_raws
        self.mne_labels = mne_labels
        self.n_splits = n_splits
        self.save_model = save_model
        self.cache_config = cache_config
        self.optuna = optuna
        self.time_out = time_out

        if self.optuna and not optuna_available:
            raise ImportError("Optuna is not available. Please install it first.")
        if (self.time_out != 60 * 15) and not self.optuna:
            warn(
                "time_out parameter is only used when optuna is enabled. "
                "Ignoring time_out parameter."
            )
        # check paradigm
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm

        # check labels
        if self.mne_labels and not self.return_epochs:
            raise (ValueError("mne_labels could only be set with return_epochs"))

        # if no dataset provided, then we get the list from the paradigm
        if datasets is None:
            datasets = self.paradigm.datasets

        if not isinstance(datasets, list):
            if isinstance(datasets, BaseDataset):
                datasets = [datasets]
            else:
                raise (ValueError("datasets must be a list or a dataset " "instance"))

        for dataset in datasets:
            if not (isinstance(dataset, BaseDataset)):
                raise (ValueError("datasets must only contains dataset " "instance"))
        rm = []
        for dataset in datasets:
            valid_for_paradigm = self.paradigm.is_valid(dataset)
            valid_for_eval = self.is_valid(dataset)
            if not valid_for_paradigm:
                log.warning(
                    f"{dataset} not compatible with "
                    "paradigm. Removing this dataset from the list."
                )
                rm.append(dataset)
            elif not valid_for_eval:
                log.warning(
                    f"{dataset} not compatible with evaluation. "
                    "Removing this dataset from the list."
                )
                rm.append(dataset)

        [datasets.remove(r) for r in rm]
        if len(datasets) > 0:
            self.datasets = datasets
        else:
            raise Exception(
                """No datasets left after paradigm
            and evaluation checks"""
            )

        self.results = Results(
            type(self),
            type(self.paradigm),
            overwrite=overwrite,
            suffix=suffix,
            hdf5_path=self.hdf5_path,
            additional_columns=additional_columns,
        )

    def process(self, pipelines, param_grid=None, postprocess_pipeline=None):
        """Runs all pipelines on all datasets.

        This function will apply all provided pipelines and return a dataframe
        containing the results of the evaluation.

        Parameters
        ----------
        pipelines : dict of pipeline instance.
            A dict containing the sklearn pipeline to evaluate.
        param_grid : dict of str
            The key of the dictionary must be the same as the associated pipeline.
        postprocess_pipeline: Pipeline | None
            Optional pipeline to apply to the data after the preprocessing.
            This pipeline will either receive :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
            or :func:`np.ndarray` as input, depending on the values of ``return_epochs``
            and ``return_raws``.
            This pipeline must return an ``np.ndarray``.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.


        Returns
        -------
        results: pd.DataFrame
            A dataframe containing the results.
        """

        # check pipelines
        if not isinstance(pipelines, dict):
            raise (ValueError("pipelines must be a dict"))

        for _, pipeline in pipelines.items():
            if not (isinstance(pipeline, BaseEstimator)):
                raise (ValueError("pipelines must only contains Pipelines " "instance"))

        res_per_db = []
        for dataset in self.datasets:
            log.info("Processing dataset: {}".format(dataset.code))
            process_pipeline = self.paradigm.make_process_pipelines(
                dataset,
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                postprocess_pipeline=postprocess_pipeline,
            )[0]
            # (we only keep the pipeline for the first frequency band, better ideas?)

            results = self.evaluate(
                dataset,
                pipelines,
                param_grid=param_grid,
                process_pipeline=process_pipeline,
                postprocess_pipeline=postprocess_pipeline,
            )
            for res in results:
                self.push_result(res, pipelines, process_pipeline)
            res_per_db.append(
                self.results.to_dataframe(
                    pipelines=pipelines, process_pipeline=process_pipeline
                )
            )

        return pd.concat(res_per_db, ignore_index=True)

    def push_result(self, res, pipelines, process_pipeline):
        message = "{} | ".format(res["pipeline"])
        message += "{} | {} | {}".format(
            res["dataset"].code, res["subject"], res["session"]
        )
        message += ": Score %.3f" % res["score"]
        log.info(message)
        self.results.add(
            {res["pipeline"]: res}, pipelines=pipelines, process_pipeline=process_pipeline
        )

    def get_results(self):
        return self.results.to_dataframe()

    @abstractmethod
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        """Evaluate results on a single dataset.

        This method return a generator. each results item is a dict with
        the following conversion::

            res = {'time': Duration of the training ,
                   'dataset': dataset id,
                   'subject': subject id,
                   'session': session id,
                   'score': score,
                   'n_samples': number of training examples,
                   'n_channels': number of channel,
                   'pipeline': pipeline name}
        """
        pass

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with evaluation.

        This method is called to verify dataset given in the constructor
        are compatible with the evaluation context.

        This method should return false if the dataset does not match the
        evaluation. This is for example the case if the dataset does not
        contain enough session for a cross-session eval.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """

    def _grid_search(self, param_grid, name, grid_clf, inner_cv):
        extra_params = {}
        if param_grid is not None:
            if name in param_grid:
                if self.optuna:
                    search = search_methods["optuna"]
                    param_grid[name] = _convert_sklearn_params_to_optuna(param_grid[name])
                    extra_params["timeout"] = self.time_out
                else:
                    search = search_methods["grid"]

                search = search(
                    grid_clf,
                    param_grid[name],
                    refit=True,
                    cv=inner_cv,
                    n_jobs=self.n_jobs,
                    scoring=self.paradigm.scoring,
                    return_train_score=True,
                    **extra_params,
                )
                return search
            else:
                return grid_clf

        else:
            return grid_clf
