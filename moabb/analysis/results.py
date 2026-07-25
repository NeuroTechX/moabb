import hashlib
import os
import os.path as osp
import re
import warnings
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from mne import get_config, set_config
from mne.datasets.utils import _get_path
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from moabb.utils import _open_lock_hdf5


try:
    from codecarbon import EmissionsTracker  # noqa

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False


def get_string_rep(obj):
    if issubclass(type(obj), BaseEstimator):
        str_repr = repr(obj.get_params())
    else:
        str_repr = repr(obj)
    if "<lambda> at " in str_repr:
        warnings.warn(
            "You are probably using a classifier with a lambda function"
            " as an attribute. Lambda functions can only be identified"
            " by memory address which MOABB does not consider. To avoid"
            " issues you can use named functions defined using the def"
            " keyword instead.",
            RuntimeWarning,
            stacklevel=2,
        )
    str_no_addresses = re.sub(
        r"0x[\w]+>", "0x__", str_repr
    )  # \w also includes _ for address such as 0x__
    return str_no_addresses.replace("\n", "").encode("utf8")


def get_digest(obj):
    """Return hash of an object repr.

    If there are memory addresses, wipes them
    """
    return hashlib.md5(get_string_rep(obj)).hexdigest()


def get_pipeline_digest(process_pipeline, clf_pipeline):
    full_pipeline = Pipeline(steps=[("process", process_pipeline), ("clf", clf_pipeline)])
    return get_digest(full_pipeline)


class Results:
    """Class to hold results from the evaluation.evaluate method.

    Appropriate test would be to ensure the result of 'evaluate' is
    consistent and can be accepted by 'results.add'

    Saves dataframe per pipeline and can query to see if particular
    subject has already been run
    """

    def __init__(
        self,
        evaluation_class,
        paradigm_class,
        suffix="",
        overwrite=False,
        hdf5_path=None,
        additional_columns=None,
    ):
        """Class that will abstract result storage."""
        from moabb.evaluations.base import BaseEvaluation
        from moabb.paradigms.base import BaseParadigm

        if not issubclass(
            evaluation_class, BaseEvaluation
        ):  # was assert, raises properly
            raise TypeError(
                f"evaluation_class must be a subclass of BaseEvaluation, "
                f"got {evaluation_class}"
            )
        if not issubclass(paradigm_class, BaseParadigm):  # was assert
            raise TypeError(
                f"paradigm_class must be a subclass of BaseParadigm, got {paradigm_class}"
            )

        if additional_columns is None:
            self.additional_columns = []
        else:
            if not all(isinstance(ac, str) for ac in additional_columns):  # was assert
                raise TypeError("all additional_columns must be strings")
            self.additional_columns = additional_columns

        if hdf5_path is None:
            if get_config("MOABB_RESULTS") is None:
                # Use MNE_DATA if configured (env var or config file),
                # otherwise fall back to ~/mne_data
                mne_data = get_config("MNE_DATA")
                if mne_data is None:
                    mne_data = osp.join(osp.expanduser("~"), "mne_data")
                set_config("MOABB_RESULTS", mne_data)
            self.mod_dir = _get_path(None, "MOABB_RESULTS", "results")
            # was previously stored in the moabb source file folder:
            # self.mod_dir = osp.dirname(osp.abspath(inspect.getsourcefile(moabb)))
        else:
            self.mod_dir = osp.abspath(hdf5_path)
        self.filepath = osp.join(
            self.mod_dir,
            "results",
            paradigm_class.__name__,
            evaluation_class.__name__,
            "results{}.hdf5".format("_" + suffix),
        )

        os.makedirs(osp.dirname(self.filepath), exist_ok=True)
        self.filepath = self.filepath

        if overwrite:
            with _open_lock_hdf5(self.filepath, "w") as f:
                f.attrs["create_time"] = np.bytes_(
                    "{:%Y-%m-%d, %H:%M}".format(datetime.now())
                )
        elif not osp.isfile(self.filepath):
            with _open_lock_hdf5(self.filepath, "w") as f:
                f.attrs["create_time"] = np.bytes_(
                    "{:%Y-%m-%d, %H:%M}".format(datetime.now())
                )

    def add(self, results, pipelines, process_pipeline):  # noqa: C901
        """Add results."""

        def to_list(res):
            if isinstance(res, dict):
                return [res]
            elif not isinstance(res, list):
                raise ValueError(
                    "Results are given as neither dict norlist but {}".format(
                        type(res).__name__
                    )
                )
            else:
                return res

        col_names = ["score", "time", "samples", "samples_test", "n_classes"]
        if _carbonfootprint:
            n_cols = 6
            col_names.append("carbon_emission")
        else:
            n_cols = 5

        with _open_lock_hdf5(self.filepath, "r+") as f:
            for name, data_dict in results.items():
                digest = get_pipeline_digest(process_pipeline, pipelines[name])
                if digest not in f.keys():
                    # create pipeline main group if nonexistent
                    f.create_group(digest)

                ppline_grp = f[digest]
                ppline_grp.attrs["name"] = name
                ppline_grp.attrs["repr"] = repr(pipelines[name])

                dlist = to_list(data_dict)
                d1 = dlist[0]  # FIXME: handle multiple session ?
                dname = d1["dataset"].code
                n_add_cols = len(self.additional_columns)
                if dname not in ppline_grp.keys():
                    # create dataset subgroup if nonexistent
                    dset = ppline_grp.create_group(dname)
                    dset.attrs["n_subj"] = len(d1["dataset"].subject_list)
                    dset.attrs["n_sessions"] = d1["dataset"].n_sessions
                    dt = h5py.special_dtype(vlen=str)

                    # Create unique CodeCarbon task name attritbute
                    if _carbonfootprint:
                        dset.create_dataset(
                            "codecarbon_task_name", (0,), dtype=dt, maxshape=(None,)
                        )

                    dset.create_dataset("id", (0, 2), dtype=dt, maxshape=(None, 2))
                    dset.create_dataset(
                        "data",
                        (0, n_cols + n_add_cols),
                        maxshape=(None, n_cols + n_add_cols),
                    )
                    dset.attrs["channels"] = d1["n_channels"]
                    dset.attrs.create(
                        "columns", col_names + self.additional_columns, dtype=dt
                    )
                dset = ppline_grp[dname]
                # Backward compat: existing dataset may have fewer columns
                n_existing = dset["data"].shape[1]
                n_new = len(dlist)
                old_len = len(dset["id"])
                new_len = old_len + n_new
                dset["id"].resize(new_len, 0)
                dset["data"].resize(new_len, 0)
                if _carbonfootprint and "codecarbon_task_name" in dset:
                    dset["codecarbon_task_name"].resize(new_len, 0)

                for i, d in enumerate(dlist):
                    row = old_len + i
                    dset["id"][row, :] = np.asarray(
                        [str(d["subject"]), str(d["session"])]
                    )
                    try:
                        add_cols = [d[ac] for ac in self.additional_columns]
                    except KeyError:
                        raise ValueError(
                            f"Additional columns: {self.additional_columns} "
                            f"were specified in the evaluation, but results"
                            f" contain only these keys: {d.keys()}."
                        ) from None
                    cols = [
                        d["score"],
                        d["time"],
                        d["n_samples"],
                        d.get("n_samples_test", np.nan),
                        d.get("n_classes", np.nan),
                    ]
                    if _carbonfootprint:
                        # Always add carbon_emission column if codecarbon is available
                        if "carbon_emission" in d:
                            if isinstance(d["carbon_emission"], tuple):
                                cols.append(*d["carbon_emission"])
                            else:
                                cols.append(d["carbon_emission"])
                        else:
                            # Add NaN if carbon_emission is not available
                            cols.append(np.nan)

                        # Save unique CodeCarbon task name (only if dataset exists)
                        if "codecarbon_task_name" in dset:
                            dset["codecarbon_task_name"][row] = str(
                                d.get("codecarbon_task_name", "")
                            )

                    all_cols = np.asarray([*cols, *add_cols])
                    dset["data"][row, :] = all_cols[:n_existing]

    @staticmethod
    def _to_dataframe_from_file(f, digests=None):
        df_list = []
        allowed = set(digests) if digests is not None else None

        for digest, p_group in f.items():
            if (allowed is not None) and (digest not in allowed):
                continue

            name = p_group.attrs["name"]
            for dname, dset in p_group.items():
                array = np.array(dset["data"])
                ids = np.array(dset["id"])
                df = pd.DataFrame(array, columns=dset.attrs["columns"])
                df["subject"] = [s.decode() for s in ids[:, 0]]
                df["session"] = [s.decode() for s in ids[:, 1]]
                df["channels"] = dset.attrs["channels"]
                df["n_sessions"] = dset.attrs["n_sessions"]
                df["dataset"] = dname
                df["pipeline"] = name
                if _carbonfootprint and "codecarbon_task_name" in dset:
                    df["codecarbon_task_name"] = np.array(
                        dset["codecarbon_task_name"]
                    ).astype(str)
                df_list.append(df)

        if not df_list:
            return pd.DataFrame()
        result = pd.concat(df_list, ignore_index=True)
        for col in ("samples_test", "n_classes"):
            if col not in result.columns:
                result[col] = np.nan
        return result

    def to_dataframe(self, pipelines=None, process_pipeline=None):
        # get the list of pipeline hash
        digests = None
        if pipelines is not None and process_pipeline is not None:
            digests = [
                get_pipeline_digest(process_pipeline, pipelines[name])
                for name in pipelines
            ]
        elif pipelines is not None or process_pipeline is not None:
            raise ValueError(
                "Either both of none of pipelines and process_pipeline must be specified."
            )

        with _open_lock_hdf5(self.filepath, "r") as f:
            return self._to_dataframe_from_file(f, digests=digests)

    def batch_not_yet_computed_or_cached_df(
        self, pipelines, dataset, subjects, process_pipeline
    ):
        """Atomically compute work_plan and, if complete, return cached dataframe.

        Returns
        -------
        work_plan : dict
            Same format as :func:`batch_not_yet_computed`.
        cached_df : pd.DataFrame | None
            Dataframe for selected pipelines when ``work_plan`` is empty.
            ``None`` when there is still work to do.
        """
        digests = {
            name: get_pipeline_digest(process_pipeline, pipeline)
            for name, pipeline in pipelines.items()
        }
        with _open_lock_hdf5(self.filepath, "r") as f:
            computed_subjects = {}
            for name, digest in digests.items():
                if digest in f.keys():
                    pipe_grp = f[digest]
                    if dataset.code in pipe_grp.keys():
                        dset = pipe_grp[dataset.code]
                        computed_subjects[name] = set(dset["id"][:, 0])
                    else:
                        computed_subjects[name] = set()
                else:
                    computed_subjects[name] = set()

            work_plan = {}
            for subject in subjects:
                subj_encoded = str(subject).encode("utf-8")
                missing = {
                    name: pipelines[name]
                    for name in pipelines
                    if subj_encoded not in computed_subjects[name]
                }
                if missing:
                    work_plan[subject] = missing

            if work_plan:
                return work_plan, None

            cached_df = self._to_dataframe_from_file(f, digests=list(digests.values()))
            # Filter to current dataset to avoid mixing rows from other datasets
            # that share the same pipeline digest.
            if cached_df is not None and not cached_df.empty:
                cached_df = cached_df[cached_df["dataset"] == dataset.code]
            return work_plan, cached_df

    def batch_not_yet_computed(self, pipelines, dataset, subjects, process_pipeline):
        """Check all subjects at once with a single HDF5 read.

        Parameters
        ----------
        pipelines : dict of pipeline instance.
            A dict containing the sklearn pipeline to evaluate.
        dataset : Dataset instance
            The dataset to check for.
        subjects : list
            List of subjects to check.
        process_pipeline : Pipeline | None
            The processing pipeline.

        Returns
        -------
        dict
            A dict mapping subject -> {pipeline_name: pipeline} for subjects
            that still need computation. Subjects with all pipelines computed
            are omitted.
        """
        work_plan, _ = self.batch_not_yet_computed_or_cached_df(
            pipelines, dataset, subjects, process_pipeline
        )
        return work_plan

    def not_yet_computed(self, pipelines, dataset, subj, process_pipeline):
        """Check if a results is missing.

        Parameters
        ----------
        pipelines : dict of pipeline instance.
            A dict containing the sklearn pipeline to evaluate.
        dataset : Dataset instance
            The dataset to check for
        subj : str
            The subject to check for
        process_pipeline : Pipeline | None
            Optional pipeline to apply to the data after the preprocessing.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.

        Returns
        -------
        dict
            A dict containing the pipelines to compute.
        """
        ret = {
            k: pipelines[k]
            for k in pipelines.keys()
            if not self._already_computed(pipelines[k], dataset, subj, process_pipeline)
        }
        return ret

    def _already_computed(
        self, pipeline, dataset, subject, process_pipeline, session=None
    ):
        """Check existing results for pipeline / dataset / subject combination.

        Parameters
        ----------
        pipeline : dict of pipeline instance.
            A dict containing the sklearn pipeline to evaluate.
        dataset : Dataset instance
            The dataset to check for
        subject : str
            The subject to check for
        process_pipeline : Pipeline | None
            Optional pipeline to apply to the data after the preprocessing.
            This pipeline must be "fixed" because it will not be trained,
            i.e. no call to ``fit`` will be made.
        session : str | None
            Not used, kept for compatibility reason.

        Returns
        -------
        bool
            True if the pipeline has already been computed for the given
            dataset and subject, False otherwise.
        """
        with h5py.File(self.filepath, "r") as f:
            # get the digest from repr
            digest = get_pipeline_digest(process_pipeline, pipeline)

            # check if digest present
            if digest not in f.keys():
                return False
            else:
                pipe_grp = f[digest]
                # if present, check for dataset code
                if dataset.code not in pipe_grp.keys():
                    return False
                else:
                    # if dataset, check for subject
                    dset = pipe_grp[dataset.code]
                    return str(subject).encode("utf-8") in dset["id"][:, 0]
