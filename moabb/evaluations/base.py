import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from time import perf_counter
from typing import Optional, Union
from uuid import uuid4
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from moabb.analysis import Results
from moabb.datasets.base import (  # noqa: F401 - CacheConfig used in type hints
    BaseDataset,
    CacheConfig,
)
from moabb.evaluations.utils import (
    Emissions,
    _carbonfootprint,
    _convert_sklearn_params_to_optuna,
    _create_save_path,
    _create_scorer,
    _DictScorer,
    _ensure_fitted,
    _get_nchan,
    _pipeline_requires_epochs,
    _save_model_cv,
    _score_and_update,
    _update_result_with_scores,
    check_search_available,
)
from moabb.paradigms.base import BaseParadigm
from moabb.utils import verbose


search_methods, optuna_available = check_search_available()

log = logging.getLogger(__name__)

# Making the optuna soft dependency


def _grid_search_static(
    param_grid, name, grid_clf, inner_cv, scoring, n_jobs, optuna, time_out
):
    """Wrap a classifier in grid/optuna search if param_grid has an entry for name.

    Returns (wrapped_clf, is_search). Pure function for use in parallel workers.
    """
    extra_params = {}
    if param_grid is not None:
        if name in param_grid:
            if optuna:
                search = search_methods["optuna"]
                param_grid[name] = _convert_sklearn_params_to_optuna(param_grid[name])
                extra_params["timeout"] = time_out
            else:
                search = search_methods["grid"]

            if isinstance(scoring, dict):
                refit = next(iter(scoring))
            else:
                refit = True

            search = search(
                grid_clf,
                param_grid[name],
                refit=refit,
                cv=inner_cv,
                n_jobs=n_jobs,
                scoring=scoring,
                return_train_score=True,
                **extra_params,
            )
            return search, True
        else:
            return grid_clf, False
    else:
        return grid_clf, False


def _evaluate_fold(
    X,
    y,
    metadata,
    config,
    dataset,
    pipeline_name,
    pipeline,
    train_idx,
    test_idx,
    subject,
    session,
    cv_ind,
    split_metadata=None,
):
    """Evaluate a single CV fold. Pure function, no shared mutable state.

    Parameters
    ----------
    X : :class:`numpy.ndarray` or :class:`mne.Epochs`
        Full data.
    y : :class:`numpy.ndarray`
        Labels.
    metadata : :class:`pandas.DataFrame`
        Metadata.
    config : dict
        Evaluation-wide settings (scoring, error_score, random_state, etc.).
    dataset : :class:`~moabb.datasets.base.BaseDataset`
        Dataset instance.
    pipeline_name : str
        Pipeline name.
    pipeline : :class:`sklearn.base.BaseEstimator`
        Sklearn pipeline.
    train_idx, test_idx : :class:`numpy.ndarray`
        Indices for train/test split.
    subject, session : int|str, str
        Subject and session identifiers.
    cv_ind : int
        Cross-validation fold index.
    split_metadata : dict | None
        Extra metadata from the splitter.

    Returns
    -------
    list of dict
    """
    scoring = config["scoring"]
    error_score = config["error_score"]
    random_state = config["random_state"]
    param_grid = config["param_grid"]
    additional_columns = config["additional_columns"]
    score_per_session = config["score_per_session"]
    mne_labels = config["mne_labels"]
    codecarbon_config = config["codecarbon_config"]

    # Label encode per fold (matching old per-session/per-subject scoping)
    if not mne_labels:
        le = LabelEncoder()
        combined_y = np.concatenate([y[train_idx], y[test_idx]])
        le.fit(combined_y)
        y_train = le.transform(y[train_idx])
        y_test = le.transform(y[test_idx])
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]

    inner_cv = StratifiedKFold(3, shuffle=True, random_state=random_state)
    grid_clf, is_search = _grid_search_static(
        param_grid=param_grid,
        name=pipeline_name,
        grid_clf=clone(pipeline),
        inner_cv=inner_cv,
        scoring=scoring,
        n_jobs=config["n_jobs_grid"],
        optuna=config["optuna"],
        time_out=config["time_out"],
    )

    cvclf = clone(grid_clf) if is_search else grid_clf
    nchan = _get_nchan(X)

    # Set up emissions tracker
    tracker = None
    if _carbonfootprint and codecarbon_config is not None:
        emissions_obj = Emissions(codecarbon_config=codecarbon_config)
        tracker = emissions_obj.create_tracker()
        tracker.start()

    # Fit model
    task_name = None
    emissions = math.nan
    if tracker is not None:
        task_name = str(uuid4())
        tracker.start_task(task_name)
    t_start = perf_counter()
    cvclf.fit(X[train_idx], y_train)
    duration = perf_counter() - t_start
    if tracker is not None:
        emissions_data = tracker.stop_task()
        emissions = emissions_data.emissions if emissions_data else math.nan
    _ensure_fitted(cvclf)

    if tracker is not None:
        tracker.stop()

    # Optionally save model
    hdf5_path = config["hdf5_path"]
    eval_type = config["eval_type"]
    if hdf5_path is not None and config["save_model"]:
        model_save_path = _create_save_path(
            hdf5_path=hdf5_path,
            code=dataset.code,
            subject=subject,
            session="" if score_per_session else session,
            name=pipeline_name,
            grid=is_search,
            eval_type=eval_type,
        )
        _save_model_cv(model=cvclf, save_path=model_save_path, cv_index=str(cv_ind))

    scorer = _create_scorer(cvclf, scoring)

    # Build score groups: per-session or full test set
    if score_per_session:
        test_sessions = metadata.iloc[test_idx]["session"].values
        score_groups = [
            (test_idx[test_sessions == s], y_test[test_sessions == s], s)
            for s in np.unique(test_sessions)
        ]
    else:
        score_groups = [(test_idx, y_test, session)]

    results = []
    for group_idx, group_y, group_session in score_groups:
        is_error = False
        try:
            score = scorer(cvclf, X[group_idx], group_y)
        except ValueError as err:
            if error_score == "raise":
                raise err
            score = {"score": error_score}
            is_error = True

        res = {
            "time": duration,
            "dataset": dataset,
            "subject": subject,
            "session": group_session,
            "n_samples": len(train_idx),
            "n_samples_test": len(group_y),
            "n_samples_total": len(train_idx) + len(test_idx),
            "n_classes": len(np.unique(group_y)),
            "n_channels": nchan,
            "pipeline": pipeline_name,
            "is_error": is_error,
        }
        for col in additional_columns:
            if col not in res:
                if split_metadata and col in split_metadata:
                    res[col] = split_metadata[col]
                else:
                    res[col] = math.nan
        _update_result_with_scores(res, score)
        if _carbonfootprint:
            res["carbon_emission"] = 1000 * emissions
            res["codecarbon_task_name"] = task_name
        results.append(res)

    return results


class BaseEvaluation(ABC):
    """Base class that defines necessary operations for an evaluation.
    Evaluations determine what the train and test sets are and can implement
    additional data preprocessing steps for more complicated algorithms.

    Parameters
    ----------
    paradigm : :class:`~moabb.paradigms.base.BaseParadigm`
        The paradigm to use.
    datasets : list of :class:`~moabb.datasets.base.BaseDataset`
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state : int or None
        If not None, can guarantee same seed for shuffling examples.
        Defaults to ``None``.
    n_jobs : int
        Number of jobs for fitting of pipeline. Defaults to ``1``.
    overwrite : bool
        If true, overwrite the results. Defaults to ``False``.
    error_score : str or float
        Value to assign to the score if an error occurs in estimator fitting. If set to
        ``’raise’``, the error is raised. Defaults to ``"raise"``.
    suffix : str
        Suffix for the results file.
    hdf5_path : str
        Specific path for storing the results.
    additional_columns : None
        Adding information to results.
    return_epochs : bool
        Use MNE epoch to train pipelines. Defaults to ``False``.
    return_raws : bool
        Use MNE raw to train pipelines. Defaults to ``False``.
    mne_labels : bool
        If returning MNE epoch, use original dataset label if True.
        Defaults to ``False``.
    n_splits : int or None
        Number of splits for cross-validation. If None, the number of splits
        is equal to the number of subjects. Defaults to ``None``.
    cv_class : type or None
        Optional cross-validation class to override the evaluation’s default
        splitter behavior. Defaults to ``None``.
    cv_kwargs : dict or None
        Keyword arguments passed to cv_class when constructing the splitter.
        Defaults to ``None``.
    save_model : bool
        Save model after training, for each fold of cross-validation if needed.
        Defaults to ``False``.
    cache_config : :class:`~moabb.datasets.base.CacheConfig` or None
        Configuration for caching of datasets. See :class:`moabb.datasets.base.CacheConfig` for details.
        Defaults to ``None``.
    optuna : bool
        If optuna is enable it will change the GridSearch to a RandomizedGridSearch with 15 minutes of cut off time.
        This option is compatible with list of entries of type None, bool, int, float and string.
        Defaults to ``False``.
    time_out : int
        Cut off time for the optuna search expressed in seconds.
        Only used with optuna equal to True. Defaults to ``60*15`` (15 minutes).
    verbose : bool, str, int, or None
        If not None, override the default MOABB logging level used by this evaluation
        (see ``moabb.utils.verbose`` for more information on how this is handled).
        If used, it should be passed as a keyword-argument only.
        Defaults to ``None``.
    codecarbon_config : dict or None
        Allow CodeCarbon script level configurations.
        Can use combination of CodeCarbon environment variable and configuration files.
        See CodeCarbon developer documentation for more information.
        Defaults to ``dict(save_to_file=False, log_level="error")``.

    Notes
    -----
    .. versionadded:: 1.1.0
       n_splits, save_model, cache_config parameters.
    .. versionadded:: 1.1.1
       optuna, time_out parameters.
    .. versionadded:: 1.5
       verbose parameter.
    """

    search = False
    _eval_type = None
    _score_per_session = False
    _needs_all_subjects = False
    _aggregate_folds = False

    @verbose
    def __init__(
        self,
        paradigm: "BaseParadigm",
        datasets: Optional[list["BaseDataset"]] = None,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        overwrite: bool = False,
        error_score: Union[str, float] = "raise",
        suffix: str = "",
        hdf5_path: Optional[str] = None,
        additional_columns: Optional[list[str]] = None,
        return_epochs: bool = False,
        return_raws: bool = False,
        mne_labels: bool = False,
        n_splits: Optional[int] = None,
        cv_class: Optional[type] = None,
        cv_kwargs: Optional[dict] = None,
        save_model: bool = False,
        cache_config: Optional["CacheConfig"] = None,
        optuna: bool = False,
        time_out: int = 60 * 15,
        verbose: Optional[Union[bool, str, int]] = None,
        codecarbon_config: Optional[dict] = None,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.error_score = error_score
        self.hdf5_path = hdf5_path
        self.return_epochs = return_epochs
        self.return_raws = return_raws
        self.mne_labels = mne_labels
        self.n_splits = n_splits
        self.cv_class = cv_class
        self.cv_kwargs = {} if cv_kwargs is None else cv_kwargs
        self.save_model = save_model
        self.cache_config = cache_config
        self.optuna = optuna
        self.time_out = time_out
        self.verbose = verbose
        self.emissions = Emissions(codecarbon_config=codecarbon_config)

        self.additional_columns = additional_columns
        if additional_columns is None:
            self.additional_columns = []

        if self.cv_class is not None and hasattr(self.cv_class, "metadata_columns"):
            for col in self.cv_class.metadata_columns:
                if col not in self.additional_columns:
                    self.additional_columns.append(col)

        if self.optuna and not optuna_available:
            raise ImportError("Optuna is not available. Please install it first.")
        if (self.time_out != 60 * 15) and not self.optuna:
            warn(
                "time_out parameter is only used when optuna is enabled. "
                "Ignoring time_out parameter.",
                stacklevel=2,
            )
        # check paradigm
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm
        scorer = _create_scorer(None, self.paradigm.scoring)
        if not isinstance(scorer, _DictScorer):
            scoring_keys = [f"score_{key}" for key in scorer._scorers.keys()]
            self.additional_columns.extend(scoring_keys)

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
                raise (ValueError("datasets must be a list or a dataset instance"))

        for dataset in datasets:
            if not (isinstance(dataset, BaseDataset)):
                raise (ValueError("datasets must only contains dataset instance"))
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
                # Get specific reason for incompatibility
                eval_type = self.__class__.__name__
                reason = self._get_incompatibility_reason(dataset)
                log.warning(
                    f"{dataset} not compatible with {eval_type}: {reason}. "
                    "Removing this dataset from the list."
                )
                rm.append(dataset)

        [datasets.remove(r) for r in rm]
        if len(datasets) > 0:
            self.datasets = datasets
        else:
            raise Exception("""No datasets left after paradigm
            and evaluation checks""")

        self.results = Results(
            type(self),
            type(self.paradigm),
            overwrite=overwrite,
            suffix=suffix,
            hdf5_path=self.hdf5_path,
            additional_columns=self.additional_columns,
        )

    def _resolve_cv(self, default_class, default_kwargs=None):
        """Resolve the cross-validation class and kwargs for a splitter."""
        if self.cv_class is None:
            cv_class = default_class
            cv_kwargs = {} if default_kwargs is None else dict(default_kwargs)
        else:
            cv_class = self.cv_class
            cv_kwargs = dict(self.cv_kwargs)
        return cv_class, cv_kwargs

    def _load_data(
        self, dataset, run_pipes, process_pipeline, postprocess_pipeline, subjects=None
    ):
        """Load data for an evaluation, handling epoch requirements.

        Parameters
        ----------
        dataset : BaseDataset
            The dataset to load.
        run_pipes : dict
            Pipelines to run (used to check epoch requirements).
        process_pipeline : :class:`sklearn.pipeline.Pipeline`
            The processing pipeline.
        postprocess_pipeline : :class:`sklearn.pipeline.Pipeline` | None
            Optional post-processing pipeline.
        subjects : list | None
            List of subjects to load. If None, loads all subjects.

        Returns
        -------
        X : :class:`numpy.ndarray` or :class:`mne.Epochs`
            The loaded data.
        y : :class:`numpy.ndarray`
            The labels.
        metadata : DataFrame
            The metadata.
        """
        requires_epochs = any(
            _pipeline_requires_epochs(clf) for clf in run_pipes.values()
        )
        return_epochs = True if requires_epochs else self.return_epochs
        kwargs = {
            "dataset": dataset,
            "return_epochs": return_epochs,
            "return_raws": self.return_raws,
            "cache_config": self.cache_config,
            "postprocess_pipeline": postprocess_pipeline,
            "process_pipelines": None if requires_epochs else [process_pipeline],
        }
        if subjects is not None:
            kwargs["subjects"] = subjects
        return self.paradigm.get_data(**kwargs)

    @staticmethod
    def _get_nchan(X):
        """Extract number of channels from data (Epochs or ndarray)."""
        return _get_nchan(X)

    def _build_scored_result(
        self,
        dataset,
        subject,
        session,
        pipeline,
        n_samples,
        n_channels,
        duration,
        scorer,
        model,
        X_test,
        y_test,
        split_metadata=None,
        **extra,
    ):
        """Build a result dict and score it in one place."""
        metadata = {}
        if split_metadata is None:
            splitter = getattr(self, "cv", None)
            if splitter is not None and hasattr(splitter, "get_metadata"):
                split_metadata = splitter.get_metadata()
        if split_metadata:
            metadata.update(split_metadata)
        metadata.update(extra)
        res = self._build_result(
            dataset,
            subject,
            session,
            pipeline,
            n_samples,
            n_channels,
            duration,
            **metadata,
        )
        res["n_samples_test"] = len(y_test)
        res["n_classes"] = len(np.unique(y_test))
        try:
            return _score_and_update(res, scorer, model, X_test, y_test)
        except ValueError as err:
            if self.error_score == "raise":
                raise err
            res["score"] = self.error_score
            return res

    def _fit_cv(self, model, X_train, y_train, tracker=None):
        """Fit a model for a CV fold with optional CodeCarbon tracking."""
        task_name = None
        emissions = math.nan
        if tracker is not None:
            task_name = str(uuid4())
            tracker.start_task(task_name)
        t_start = perf_counter()
        model.fit(X_train, y_train)
        duration = perf_counter() - t_start
        if tracker is not None:
            emissions_data = tracker.stop_task()
            emissions = emissions_data.emissions if emissions_data else math.nan
        _ensure_fitted(model)
        return duration, emissions, task_name

    def _maybe_save_model_cv(
        self, model, dataset, subject, session, name, cv_ind, eval_type
    ):
        """Save model for a CV fold when saving is enabled."""
        if self.hdf5_path is None or not self.save_model:
            return
        model_save_path = _create_save_path(
            hdf5_path=self.hdf5_path,
            code=dataset.code,
            subject=subject,
            session=session,
            name=name,
            grid=self.search,
            eval_type=eval_type,
        )
        _save_model_cv(model=model, save_path=model_save_path, cv_index=str(cv_ind))

    @staticmethod
    def _attach_emissions(res, emissions, task_name):
        res["carbon_emission"] = 1000 * emissions
        res["codecarbon_task_name"] = task_name

    def _build_result(
        self,
        dataset,
        subject,
        session,
        pipeline,
        n_samples,
        n_channels,
        duration,
        **extra,
    ):
        """Build a result dictionary with all required columns.

        This is the single place where the evaluation result schema is defined.
        All evaluation subclasses should use this instead of constructing the
        dict manually, so the schema stays consistent when columns are added
        or evaluations are merged.

        Any ``additional_columns`` not provided via *extra* are defaulted to
        NaN so that ``Results.add()`` never fails on a missing key.
        """
        res = {
            "time": duration,
            "dataset": dataset,
            "subject": subject,
            "session": session,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "pipeline": pipeline,
            "n_samples_test": 0,
            "n_classes": 0,
        }
        for col in self.additional_columns:
            if col not in res:
                res[col] = extra.get(col, math.nan)
        return res

    def _create_splitter(self):
        """Create the splitter for this evaluation type.

        Subclasses should override this to return their specific splitter.
        Returns None if the subclass doesn't support the flattened parallel
        approach (falls back to the old process flow).
        """
        return None

    def _build_eval_config(self, param_grid):
        """Build evaluation-wide config dict shared across all fold tasks."""
        return {
            "scoring": self.paradigm.scoring,
            "error_score": self.error_score,
            "random_state": self.random_state,
            "optuna": self.optuna,
            "time_out": self.time_out,
            "n_jobs_grid": 1,
            "additional_columns": self.additional_columns,
            "save_model": self.save_model,
            "hdf5_path": self.hdf5_path,
            "eval_type": self._eval_type or self.__class__.__name__,
            "mne_labels": self.mne_labels,
            "codecarbon_config": (
                self.emissions.codecarbon_config if _carbonfootprint else None
            ),
            "score_per_session": self._score_per_session,
            "param_grid": None,  # overridden per-task below if needed
        }

    @staticmethod
    def _preview_splits(splitter, y, metadata):
        """Materialize folds up front with optional splitter metadata."""
        preview = []
        for cv_ind, (train_idx, test_idx) in enumerate(splitter.split(y, metadata)):
            split_metadata = None
            if hasattr(splitter, "get_metadata"):
                split_metadata = splitter.get_metadata()
                if split_metadata is not None:
                    split_metadata = dict(split_metadata)
            preview.append((cv_ind, train_idx, test_idx, split_metadata))
        return preview

    def _build_task_list(
        self, dataset, X, y, metadata, splitter, work_plan, pipelines, param_grid
    ):
        """Build a flat list of fold tasks for parallel execution."""
        tasks = []
        config = self._build_eval_config(param_grid)
        fold_preview = self._preview_splits(splitter, y, metadata)

        for cv_ind, train_idx, test_idx, split_meta in fold_preview:
            test_meta = metadata.iloc[test_idx]
            subject = test_meta["subject"].iloc[0]

            if subject not in work_plan:
                continue
            run_pipes = work_plan[subject]
            session = test_meta["session"].iloc[0]

            for name, clf in run_pipes.items():
                task_config = dict(config)
                if param_grid is not None and name in param_grid:
                    task_param_grid = {name: deepcopy(param_grid[name])}
                else:
                    task_param_grid = None
                task_config["param_grid"] = task_param_grid
                tasks.append(
                    {
                        "config": task_config,
                        "dataset": dataset,
                        "pipeline_name": name,
                        "pipeline": clf,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "subject": subject,
                        "session": session,
                        "cv_ind": cv_ind,
                        "split_metadata": split_meta,
                    }
                )
        return tasks

    def _evaluate_parallel_dataset(
        self,
        dataset,
        pipelines,
        param_grid,
        process_pipeline,
        postprocess_pipeline=None,
        work_plan=None,
    ):
        """Evaluate one dataset through the splitter-based parallel path.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset to evaluate.
        pipelines : dict
            Pipeline mapping passed by the caller.
        param_grid : dict | None
            Optional hyperparameter search grids.
        process_pipeline : Pipeline
            Preprocessing pipeline already built for this dataset.
        postprocess_pipeline : Pipeline | None
            Optional fixed postprocessing pipeline.
        work_plan : dict | None
            Optional mapping ``subject -> {pipeline_name: pipeline}``.
            When ``None``, the work plan is computed from cached results.

        Returns
        -------
        list of dict
            Result dictionaries matching :meth:`evaluate`.
        """
        dataset_splitter = self._create_splitter()
        if dataset_splitter is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not define a splitter-backed "
                "parallel evaluation path."
            )
        self.cv = dataset_splitter

        if work_plan is None:
            work_plan = {}
            for subject in dataset.subject_list:
                run_pipes = self.results.not_yet_computed(
                    pipelines, dataset, subject, process_pipeline
                )
                if run_pipes:
                    work_plan[subject] = run_pipes

        if not work_plan:
            return []

        subjects_to_load = (
            dataset.subject_list if self._needs_all_subjects else list(work_plan.keys())
        )
        run_pipes = {
            name: pipe
            for subject_pipelines in work_plan.values()
            for name, pipe in subject_pipelines.items()
        }
        X, y, metadata = self._load_data(
            dataset,
            run_pipes,
            process_pipeline,
            postprocess_pipeline,
            subjects=subjects_to_load,
        )

        tasks = self._build_task_list(
            dataset, X, y, metadata, dataset_splitter, work_plan, pipelines, param_grid
        )
        if not tasks:
            return []

        # X, y, metadata passed as positional args for joblib auto-mmap.
        fold_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_evaluate_fold)(X, y, metadata, **task) for task in tasks
        )
        all_results = list(chain.from_iterable(fold_results))

        if self._aggregate_folds and not hasattr(
            getattr(dataset_splitter, "cv_class", None), "get_metadata"
        ):
            all_results = self._aggregate_fold_results(all_results)

        for res in all_results:
            res.pop("n_samples_total", None)
            res.pop("is_error", None)
        return all_results

    @staticmethod
    def _aggregate_fold_results(fold_results):
        """Aggregate per-fold results into averaged results.

        Groups by (subject, session, pipeline) and averages scores/durations.

        Parameters
        ----------
        fold_results : list of dict
            Per-fold result dicts.

        Returns
        -------
        list of dict
            Aggregated result dicts.
        """
        if not fold_results:
            return []

        df = pd.DataFrame(fold_results)
        group_keys = ["subject", "session", "pipeline"]
        score_cols = [c for c in df.columns if c == "score" or c.startswith("score_")]
        agg_ops = dict.fromkeys(score_cols + ["time"], "mean")
        if "n_samples_test" in df.columns:
            agg_ops["n_samples_test"] = "mean"
        if "n_classes" in df.columns:
            agg_ops["n_classes"] = "max"

        # Error folds may lack score_* columns; fill with their "score" fallback
        for col in score_cols:
            if col != "score":
                df[col] = df[col].fillna(df["score"])

        has_carbon = "carbon_emission" in df.columns

        grouped = df.groupby(group_keys, sort=False)
        agg_df = grouped.agg(agg_ops)

        results = []
        for key, sub_df in grouped:
            template = sub_df.iloc[0].to_dict()
            template["n_samples"] = template.get("n_samples_total", template["n_samples"])
            for col in agg_ops:
                value = agg_df.loc[key, col]
                if col in {"n_samples_test", "n_classes"} and pd.notna(value):
                    value = int(value)
                template[col] = value
            if has_carbon:
                template["carbon_emission"] = sub_df["carbon_emission"].sum()
                template["codecarbon_task_name"] = ""
            template.pop("n_samples_total", None)
            template.pop("is_error", None)
            if not has_carbon:
                template.pop("codecarbon_task_name", None)
            results.append(template)
        return results

    def process(
        self,
        pipelines: dict[str, BaseEstimator],
        param_grid: Optional[dict[str, dict]] = None,
        postprocess_pipeline: Optional[BaseEstimator] = None,
    ):
        """Runs all pipelines on all datasets.

        This function will apply all provided pipelines and return a dataframe
        containing the results of the evaluation.

        Parameters
        ----------
        pipelines : dict
            A dict of pipeline instances containing the sklearn pipelines to evaluate.
        param_grid : dict of str
            The key of the dictionary must be the same as the associated pipeline.
        postprocess_pipeline: :class:`sklearn.pipeline.Pipeline` | None
            Optional pipeline to apply to the data after the preprocessing.
            This pipeline will either receive :class:`mne.io.BaseRaw`, :class:`mne.Epochs`
            or :class:`numpy.ndarray` as input, depending on the values of ``return_epochs``
            and ``return_raws``.
            This pipeline must return a :class:`numpy.ndarray`.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.

        Returns
        -------
        results: :class:`pandas.DataFrame`
            A dataframe containing the results.
        """

        # check pipelines
        if not isinstance(pipelines, dict):
            raise (ValueError("pipelines must be a dict"))

        for _, pipeline in pipelines.items():
            if not (isinstance(pipeline, BaseEstimator)):
                raise (ValueError("pipelines must only contains Pipelines instance"))

        # Try flattened parallel approach first
        if self._create_splitter() is not None:
            return self._process_parallel(pipelines, param_grid, postprocess_pipeline)

        # Fallback to old approach (dataset-level parallelism)
        warn(
            "Legacy dataset-level evaluation loop is deprecated and will be removed "
            "in a future release. Implement _create_splitter() to use the "
            "flattened parallel evaluation path.",
            FutureWarning,
            stacklevel=2,
        )
        return self._process_legacy(pipelines, param_grid, postprocess_pipeline)

    def _process_legacy(self, pipelines, param_grid, postprocess_pipeline):
        """Original process() implementation with dataset-level parallelism."""
        # Prepare dataset processing parameters
        processing_params = [
            (
                dataset,
                self.paradigm.make_process_pipelines(
                    dataset,
                    return_epochs=self.return_epochs,
                    return_raws=self.return_raws,
                    postprocess_pipeline=postprocess_pipeline,
                )[0],
            )
            for dataset in self.datasets
        ]

        # Parallel processing...
        parallel_results = Parallel(n_jobs=self.n_jobs)(
            delayed(
                lambda d, p: list(
                    self.evaluate(
                        d,
                        pipelines,
                        param_grid=param_grid,
                        process_pipeline=p,
                        postprocess_pipeline=postprocess_pipeline,
                    )
                )
            )(dataset, process_pipeline)
            for dataset, process_pipeline in processing_params
        )

        res_per_db = []
        # Process results in order
        for (_dataset, process_pipeline), results in zip(
            processing_params, parallel_results
        ):
            for res in results:
                self.push_result(res, pipelines, process_pipeline)

            res_per_db.append(
                self.results.to_dataframe(
                    pipelines=pipelines, process_pipeline=process_pipeline
                )
            )

        return pd.concat(res_per_db, ignore_index=True)

    def _process_parallel(self, pipelines, param_grid, postprocess_pipeline):
        """Flattened parallel process: all folds per dataset in parallel."""
        res_per_db = []

        for dataset in self.datasets:
            if not self.is_valid(dataset):
                continue

            process_pipeline = self.paradigm.make_process_pipelines(
                dataset,
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                postprocess_pipeline=postprocess_pipeline,
            )[0]

            work_plan, cached_df = self.results.batch_not_yet_computed_or_cached_df(
                pipelines, dataset, dataset.subject_list, process_pipeline
            )
            if not work_plan:
                if (cached_df is not None) and (not cached_df.empty):
                    res_per_db.append(cached_df)
                    continue
                # Inconsistent cache state: "all computed" but nothing readable.
                # Recompute instead of returning an empty dataset result.
                log.warning(
                    "Empty cache detected for dataset %s in %s; recomputing.",
                    dataset.code,
                    self.__class__.__name__,
                )
                work_plan = {subj: dict(pipelines) for subj in dataset.subject_list}

            all_results = self._evaluate_parallel_dataset(
                dataset=dataset,
                pipelines=pipelines,
                param_grid=param_grid,
                process_pipeline=process_pipeline,
                postprocess_pipeline=postprocess_pipeline,
                work_plan=work_plan,
            )
            if not all_results:
                res_per_db.append(
                    self.results.to_dataframe(
                        pipelines=pipelines, process_pipeline=process_pipeline
                    )
                )
                continue

            for res in all_results:
                self._log_result(res)
            self._push_results_batch(all_results, pipelines, process_pipeline)

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

    def _log_result(self, res):
        message = "{} | ".format(res["pipeline"])
        message += "{} | {} | {}".format(
            res["dataset"].code, res["subject"], res["session"]
        )
        message += ": Score %.3f" % res["score"]
        log.info(message)

    def _push_results_batch(self, results, pipelines, process_pipeline):
        grouped = defaultdict(list)
        for res in results:
            grouped[res["pipeline"]].append(res)
        self.results.add(grouped, pipelines=pipelines, process_pipeline=process_pipeline)

    def get_results(self):
        return self.results.to_dataframe()

    @abstractmethod
    def evaluate(
        self,
        dataset: "BaseDataset",
        pipelines: dict[str, BaseEstimator],
        param_grid: Optional[dict],
        process_pipeline: BaseEstimator,
        postprocess_pipeline: Optional[BaseEstimator] = None,
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
    def is_valid(self, dataset: "BaseDataset") -> bool:
        """Verify the dataset is compatible with evaluation.

        This method is called to verify dataset given in the constructor
        are compatible with the evaluation context.

        This method should return false if the dataset does not match the
        evaluation. This is for example the case if the dataset does not
        contain enough session for a cross-session eval.

        Parameters
        ----------
        dataset : :class:`~moabb.datasets.base.BaseDataset`
            The dataset to verify.
        """

    def _get_incompatibility_reason(self, dataset):
        """Get a human-readable reason why dataset is incompatible.

        This method should be overridden by subclasses to provide
        specific incompatibility reasons.

        Parameters
        ----------
        dataset : :class:`~moabb.datasets.base.BaseDataset`
            The dataset to check.

        Returns
        -------
        str
            A human-readable reason for incompatibility.

        """
        return "requirements not met"

    def _grid_search(self, param_grid, name, grid_clf, inner_cv):
        result, is_search = _grid_search_static(
            param_grid=param_grid,
            name=name,
            grid_clf=grid_clf,
            inner_cv=inner_cv,
            scoring=self.paradigm.scoring,
            n_jobs=self.n_jobs,
            optuna=self.optuna,
            time_out=self.time_out,
        )
        self.search = is_search
        return result
