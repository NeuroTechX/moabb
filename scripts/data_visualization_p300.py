"""
===========================
Script for visualization of ALL P300 datasets
===========================

This script will download ALL P300 datasets and create
descriptive plots for every single session.

Total downloaded size will be (as of now) 120GB.


.. versionadded:: 0.4.5
"""

import warnings

# Authors: Jan Sosulski <mail@jan-sosulski.de>
#
# License: BSD (3-clause)
from pathlib import Path

import matplotlib
import mne
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from moabb.paradigms import P300


matplotlib.use("agg")
sns.set_style("whitegrid")
mne.set_log_level("WARNING")


def create_plot_overview(epo, plot_opts=None, path=None, description=""):
    # Butterflyplot
    suptitle = f"{description} ({epo_summary(epo)[1]})"
    epo_t = epo["Target"]
    epo_nt = epo["NonTarget"]
    evkd_t = epo_t.average()
    evkd_nt = epo_nt.average()
    ix_t = epo.events[:, 2] == epo.event_id["Target"]
    ix_nt = epo.events[:, 2] == epo.event_id["NonTarget"]

    fig0, ax = plt.subplots(1, 1, figsize=(10, 3), sharey="all", sharex="all")
    ax.scatter(
        epo.events[ix_t, 0],
        np.ones((np.sum(ix_t),)),
        color="r",
        marker="|",
        label="Target",
    )
    ax.scatter(
        epo.events[ix_nt, 0],
        np.zeros((np.sum(ix_nt),)),
        color="b",
        marker="|",
        label="NonTarget",
    )
    ax.legend()
    ax.set_title("Event timeline")
    fig0.suptitle(suptitle)
    fig0.tight_layout()
    fig0.savefig(path / f"event_timeline.{plot_opts['format']}", dpi=plot_opts["dpi"])

    fig1, axes = plt.subplots(2, 1, figsize=(6, 6), sharey="all", sharex="all")
    evkd_t.plot(spatial_colors=True, show=False, axes=axes[0])
    axes[0].set_title("Target response")
    evkd_nt.plot(spatial_colors=True, show=False, axes=axes[1])
    axes[1].set_title("NonTarget response")
    fig1.suptitle(suptitle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig1.tight_layout()
    fig1.savefig(
        path / f"target_nontarget_erps.{plot_opts['format']}", dpi=plot_opts["dpi"]
    )

    # topomap
    tp = plot_opts["topo"]["timepoints"]
    tmin, tmax = plot_opts["topo"]["tmin"], plot_opts["topo"]["tmax"]
    times = np.linspace(tmin, tmax, tp)
    fig2 = evkd_t.plot_topomap(times=times, colorbar=True, show=False)
    fig2.suptitle(suptitle)
    fig2.savefig(
        path / f"target_topomap_{tp}_timepoints.{plot_opts['format']}",
        dpi=plot_opts["dpi"],
    )

    # jointmap
    fig3 = evkd_t.plot_joint(show=False)
    fig3.suptitle(suptitle)
    fig3.savefig(path / f"target_erp_topo.{plot_opts['format']}", dpi=plot_opts["dpi"])

    # sensorplot
    fig4 = mne.viz.plot_compare_evokeds(
        [evkd_t.crop(0, 0.6), evkd_nt.crop(0, 0.6)], axes="topo", show=False
    )
    fig4[0].suptitle(suptitle)
    fig4[0].savefig(path / f"sensorplot.{plot_opts['format']}", dpi=plot_opts["dpi"])

    fig5, ax = plt.subplots(2, 1, figsize=(8, 6), sharex="all", sharey="all")
    t_data = epo_t.get_data() * 1e6
    nt_data = epo_nt.get_data() * 1e6
    data = epo.get_data() * 1e6
    minmax = np.max(data, axis=2) - np.min(data, axis=2)
    per_channel = np.mean(minmax, axis=0)
    worst_ch = np.argsort(per_channel)
    worst_ch = worst_ch[max(-8, -len(epo.ch_names)) :]
    minmax_t = np.max(t_data, axis=2) - np.min(t_data, axis=2)
    minmax_nt = np.max(nt_data, axis=2) - np.min(nt_data, axis=2)
    ch = epo_t.ch_names
    for i in range(minmax_nt.shape[1]):
        lab = ch[i] if i in worst_ch else None
        sns.kdeplot(minmax_t[:, i], ax=ax[0], label=lab, clip=(0, 300))
        sns.kdeplot(minmax_nt[:, i], ax=ax[1], label=lab, clip=(0, 300))
    ax[0].set_xlim(0, 200)
    ax[0].set_title("Target minmax")
    ax[1].set_title("NonTarget minmax")
    ax[1].set_xlabel("Minmax in $\\mu$V")
    ax[1].legend(title="Worst channels")
    fig5.suptitle(suptitle)
    fig5.tight_layout()
    fig5.savefig(path / f"minmax.{plot_opts['format']}", dpi=plot_opts["dpi"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig6 = epo.plot_psd(0, 20, bandwidth=1)
        fig6.suptitle(suptitle)
        fig6.tight_layout()
    fig6.savefig(path / f"spectrum.{plot_opts['format']}", dpi=plot_opts["dpi"])

    plt.close("all")


def epo_summary(epos):
    summary = dict()
    summary["mne_string"] = repr(epos)
    summary["n_channels"] = len(epos.ch_names)
    summary["n_target"] = len(epos["Target"])
    summary["n_nontarget"] = len(epos["NonTarget"])
    info_str = (
        f"Ch:{len(epos.ch_names)},T:{len(epos['Target'])},NT:{len(epos['NonTarget'])}"
    )
    return summary, info_str


if __name__ == "__main__":
    FIGURES_PATH = Path.home() / "moabb_figures" / "erps"

    # Changing this to False re-generates all plots even if they exist. Use with caution.
    cache_plots = True

    baseline = None
    highpass = 0.5
    lowpass = 16
    sampling_rate = 100

    paradigm = P300(
        resample=sampling_rate,
        fmin=highpass,
        fmax=lowpass,
        baseline=baseline,
    )

    ival = [-0.3, 1]

    plot_opts = {
        "dpi": 120,
        "topo": {
            "timepoints": 10,
            "tmin": 0,
            "tmax": 0.6,
        },
        "format": "png",
    }

    plt.ioff()
    # dsets = P300_DSETS
    dsets = paradigm.datasets
    for dset in dsets:
        dset.interval = ival
        dset_name = dset.__class__.__name__

        print(f"Processing dataset: {dset_name}")

        data_path = FIGURES_PATH / dset_name  # path of the dataset folder
        data_path.mkdir(exist_ok=True)
        all_subjects_cached = True
        for subject in dset.subject_list:
            subject_path = data_path / f"subject_{subject}"
            if cache_plots and subject_path.exists():
                continue
            all_subjects_cached = False
            print(f"  Processing subject: {subject}")

            subject_path.mkdir(parents=True, exist_ok=True)
            try:
                epos, labels, meta = paradigm.get_data(
                    dset, [subject], return_epochs=True
                )
            except Exception:  # catch all, dont stop processing pls
                print(f"Failed to get data for {dset_name}-{subject}")
                (subject_path / "processing_error").touch()
                continue

            description = f"Dset: {dset_name}, Sub: {subject}, Ses: all"

            create_plot_overview(
                epos,
                plot_opts=plot_opts,
                path=subject_path,
                description=description,
            )

            if len(meta["session"].unique()) > 1:
                for session in meta["session"].unique():
                    session_path = subject_path / f"session_{session}"
                    session_path.mkdir(parents=True, exist_ok=True)
                    ix = meta.session == session
                    description = f"Dset: {dset_name}, Sub: {subject}, Ses: {session}"
                    create_plot_overview(
                        epos[ix],
                        plot_opts=plot_opts,
                        path=session_path,
                        description=description,
                    )

        if all_subjects_cached:
            print(" No plots necessary, every subject has output folder.")

    print("All datasets processed.")
