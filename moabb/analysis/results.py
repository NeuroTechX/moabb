import functools
import hashlib
import importlib.util
import os
import os.path as osp
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from mne import get_config, set_config
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from moabb.utils import _open_lock_hdf5


_carbonfootprint = importlib.util.find_spec("codecarbon") is not None


def __getattr__(name):
    if name == "h5py":
        import h5py

        globals()["h5py"] = h5py
        return h5py
    if name == "_get_path":
        from mne.datasets.utils import _get_path

        globals()["_get_path"] = _get_path
        return _get_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_ADDRESS_RE = re.compile(r"0x[\w]+>")


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
    str_no_addresses = _ADDRESS_RE.sub(
        "0x__", str_repr
    )  # \w also includes _ for address such as 0x__
    return str_no_addresses.replace("\n", "").encode("utf8")


def get_digest(obj):
    """Return hash of an object repr.

    If there are memory addresses, wipes them
    """
    return hashlib.md5(get_string_rep(obj)).hexdigest()


@functools.lru_cache(maxsize=512)
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
            from mne.datasets.utils import _get_path

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

        if not results:
            return

        col_names = ["score", "time", "samples", "samples_test", "n_classes"]
        if _carbonfootprint:
            n_cols = 6
            col_names.append("carbon_emission")
        else:
            n_cols = 5

        digests = {
            name: get_pipeline_digest(process_pipeline, pipelines[name])
            for name in results
        }
        add_cols_tuple = tuple(self.additional_columns)
        n_add_cols = len(add_cols_tuple)

        with _open_lock_hdf5(self.filepath, "r+") as f:
            for name, data_dict in results.items():
                digest = digests[name]
                if digest not in f:
                    ppline_grp = f.create_group(digest)
                    ppline_grp.attrs["name"] = name
                    ppline_grp.attrs["repr"] = repr(pipelines[name])
                else:
                    ppline_grp = f[digest]

                dlist = to_list(data_dict)
                d1 = dlist[0]  # FIXME: handle multiple session ?
                dname = d1["dataset"].code
                if dname not in ppline_grp:
                    import h5py

                    dset = ppline_grp.create_group(dname)
                    dset.attrs["n_subj"] = len(d1["dataset"].subject_list)
                    dset.attrs["n_sessions"] = d1["dataset"].n_sessions
                    dt = h5py.special_dtype(vlen=str)

                    if _carbonfootprint:
                        dset.create_dataset(
                            "codecarbon_task_name", (0,), dtype=dt, maxshape=(None,)
                        )

                    dset.create_dataset("id", (0, 2), dtype=dt, maxshape=(None, 2))
                    dset.create_dataset(
                        "data",
                        (0, n_cols + n_add_cols),
                        dtype="f4",
                        maxshape=(None, n_cols + n_add_cols),
                    )
                    dset.attrs["channels"] = d1["n_channels"]
                    dset.attrs.create(
                        "columns", col_names + self.additional_columns, dtype=dt
                    )
                else:
                    dset = ppline_grp[dname]

                id_ds = dset["id"]
                data_ds = dset["data"]
                n_existing = data_ds.shape[1]
                n_new = len(dlist)
                old_len = len(id_ds)
                new_len = old_len + n_new
                id_ds.resize(new_len, 0)
                data_ds.resize(new_len, 0)
                has_cc_task = _carbonfootprint and "codecarbon_task_name" in dset
                if has_cc_task:
                    cc_ds = dset["codecarbon_task_name"]
                    cc_ds.resize(new_len, 0)

                ids_batch = np.empty((n_new, 2), dtype=object)
                total_cols = n_cols + n_add_cols
                data_batch = np.empty((n_new, total_cols), dtype="f4")
                cc_names_batch = np.empty(n_new, dtype=object) if has_cc_task else None

                for i, d in enumerate(dlist):
                    ids_batch[i, 0] = str(d["subject"])
                    ids_batch[i, 1] = str(d["session"])
                    data_batch[i, 0] = d["score"]
                    data_batch[i, 1] = d["time"]
                    data_batch[i, 2] = d["n_samples"]
                    data_batch[i, 3] = d.get("n_samples_test", np.nan)
                    data_batch[i, 4] = d.get("n_classes", np.nan)
                    col_offset = 5
                    if _carbonfootprint:
                        ce = d.get("carbon_emission", np.nan)
                        data_batch[i, 5] = ce[0] if isinstance(ce, tuple) else ce
                        col_offset = 6
                        if has_cc_task:
                            cc_names_batch[i] = str(d.get("codecarbon_task_name", ""))
                    if n_add_cols:
                        try:
                            for j, ac in enumerate(add_cols_tuple):
                                data_batch[i, col_offset + j] = d[ac]
                        except KeyError:
                            raise ValueError(
                                f"Additional columns: {self.additional_columns} "
                                f"were specified in the evaluation, but results"
                                f" contain only these keys: {d.keys()}."
                            ) from None

                id_ds[old_len:new_len, :] = ids_batch
                data_ds[old_len:new_len, :] = data_batch[:, :n_existing]
                if has_cc_task:
                    cc_ds[old_len:new_len] = cc_names_batch

    @staticmethod
    def _to_dataframe_from_file(f, digests=None, dataset_code=None):
        df_list = []
        if digests is not None:
            digest_items = [(d, f[d]) for d in dict.fromkeys(digests) if d in f]
        else:
            digest_items = list(f.items())

        for _digest, p_group in digest_items:
            name = p_group.attrs["name"]
            if dataset_code is not None:
                if dataset_code not in p_group:
                    continue
                dset_items = [(dataset_code, p_group[dataset_code])]
            else:
                dset_items = list(p_group.items())

            for dname, dset in dset_items:
                array = dset["data"][:]
                ids = dset["id"][:]
                cols = [
                    c.decode() if isinstance(c, bytes) else str(c)
                    for c in dset.attrs["columns"]
                ]
                data_dict = {col: array[:, idx] for idx, col in enumerate(cols)}
                id0 = ids[:, 0]
                id1 = ids[:, 1]
                data_dict["subject"] = [
                    s.decode() if isinstance(s, bytes) else str(s) for s in id0
                ]
                data_dict["session"] = [
                    s.decode() if isinstance(s, bytes) else str(s) for s in id1
                ]
                data_dict["channels"] = dset.attrs["channels"]
                data_dict["n_sessions"] = dset.attrs["n_sessions"]
                data_dict["dataset"] = dname
                data_dict["pipeline"] = name
                if _carbonfootprint and "codecarbon_task_name" in dset:
                    data_dict["codecarbon_task_name"] = [
                        s.decode() if isinstance(s, bytes) else str(s)
                        for s in dset["codecarbon_task_name"][:]
                    ]
                df_list.append(pd.DataFrame(data_dict))

        if not df_list:
            return pd.DataFrame()
        result = (
            df_list[0] if len(df_list) == 1 else pd.concat(df_list, ignore_index=True)
        )
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
        dcode = dataset.code
        with _open_lock_hdf5(self.filepath, "r") as f:
            computed_subjects = {}
            for name, digest in digests.items():
                if digest in f:
                    pipe_grp = f[digest]
                    if dcode in pipe_grp:
                        dset = pipe_grp[dcode]
                        computed_subjects[name] = {
                            s.decode("utf-8") if isinstance(s, bytes) else str(s)
                            for s in dset["id"][:, 0]
                        }
                    else:
                        computed_subjects[name] = set()
                else:
                    computed_subjects[name] = set()

            work_plan = {}
            for subject in subjects:
                subj_str = str(subject)
                missing = {
                    name: pipelines[name]
                    for name in pipelines
                    if subj_str not in computed_subjects[name]
                }
                if missing:
                    work_plan[subject] = missing

            if work_plan:
                return work_plan, None

            cached_df = self._to_dataframe_from_file(
                f, digests=list(digests.values()), dataset_code=dcode
            )
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
        return self.batch_not_yet_computed(
            pipelines, dataset, [subj], process_pipeline
        ).get(subj, {})

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
        import h5py

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
