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
        "0x[\w]+>", "0x__", str_repr
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

        assert issubclass(evaluation_class, BaseEvaluation)
        assert issubclass(paradigm_class, BaseParadigm)

        if additional_columns is None:
            self.additional_columns = []
        else:
            assert all([isinstance(ac, str) for ac in additional_columns])
            self.additional_columns = additional_columns

        if hdf5_path is None:
            if get_config("MOABB_RESULTS") is None:
                set_config("MOABB_RESULTS", osp.join(osp.expanduser("~"), "mne_data"))
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

        if overwrite and osp.isfile(self.filepath):
            os.remove(self.filepath)

        if not osp.isfile(self.filepath):
            with h5py.File(self.filepath, "w") as f:
                f.attrs["create_time"] = np.string_(
                    "{:%Y-%m-%d, %H:%M}".format(datetime.now())
                )

    def add(self, results, pipelines, process_pipeline):  # noqa: C901
        """Add results."""

        def to_list(res):
            if isinstance(res, dict):
                return [res]
            elif not isinstance(res, list):
                raise ValueError(
                    "Results are given as neither dict nor"
                    "list but {}".format(type(res).__name__)
                )
            else:
                return res

        col_names = ["score", "time", "samples"]
        if _carbonfootprint:
            n_cols = 4
            col_names.append("carbon_emission")
        else:
            n_cols = 3

        with h5py.File(self.filepath, "r+") as f:
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
                    dset.create_dataset("id", (0, 2), dtype=dt, maxshape=(None, 2))
                    dset.create_dataset(
                        "data",
                        (0, n_cols + n_add_cols),
                        maxshape=(None, n_cols + n_add_cols),
                    )
                    dset.attrs["channels"] = d1["n_channels"]
                    dset.attrs.create(
                        "columns",
                        col_names + self.additional_columns,
                        dtype=dt,
                    )
                dset = ppline_grp[dname]
                for d in dlist:
                    # add id and scores to group
                    length = len(dset["id"]) + 1
                    dset["id"].resize(length, 0)
                    dset["data"].resize(length, 0)
                    dset["id"][-1, :] = np.asarray([str(d["subject"]), str(d["session"])])
                    try:
                        add_cols = [d[ac] for ac in self.additional_columns]
                    except KeyError:
                        raise ValueError(
                            f"Additional columns: {self.additional_columns} "
                            f"were specified in the evaluation, but results"
                            f" contain only these keys: {d.keys()}."
                        ) from None
                    cols = [d["score"], d["time"], d["n_samples"]]
                    if _carbonfootprint:
                        if isinstance(d["carbon_emission"], tuple):
                            cols.append(*d["carbon_emission"])
                        else:
                            cols.append(d["carbon_emission"])
                    dset["data"][-1, :] = np.asarray(
                        [
                            *cols,
                            *add_cols,
                        ]
                    )

    def to_dataframe(self, pipelines=None, process_pipeline=None):
        df_list = []

        # get the list of pipeline hash
        digests = []
        if pipelines is not None and process_pipeline is not None:
            digests = [
                get_pipeline_digest(process_pipeline, pipelines[name])
                for name in pipelines
            ]
        elif pipelines is not None or process_pipeline is not None:
            raise ValueError(
                "Either both of none of pipelines and process_pipeline must be specified."
            )

        with h5py.File(self.filepath, "r") as f:
            for digest, p_group in f.items():
                # skip if not in pipeline list
                if (pipelines is not None) and (digest not in digests):
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
                    df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    def not_yet_computed(self, pipelines, dataset, subj, process_pipeline):
        """Check if a results has already been computed."""
        ret = {
            k: pipelines[k]
            for k in pipelines.keys()
            if not self._already_computed(pipelines[k], dataset, subj, process_pipeline)
        }
        return ret

    def _already_computed(
        self, pipeline, dataset, subject, process_pipeline, session=None
    ):
        """Check if we have results for a current combination of pipeline /
        dataset / subject."""
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
