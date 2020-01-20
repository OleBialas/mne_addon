from matplotlib import pyplot as plt
from mne.viz import plot_evoked_topo
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mne.stats import bootstrap_confidence_interval
from mne_addon.analysis import global_rms, global_field_power
from mne.epochs import Epochs, BaseEpochs
from mne.evoked import Evoked


def plot_multiple_ERP(epochs, n_row=3, n_col=2, outfile=None, title=None, plot_topo=True):
    """
    Plot ERP for the different conditions
    """
    fig, ax = plt.subplots(n_row,n_col, sharex=True, figsize=(16,9))
    if title:
        fig.suptitle(title)
    conditions = list(epochs.event_id.keys())
    if n_row*n_col!=len(conditions):
        raise ValueError("Size of the plot and number of experimental"
         "conditions dont match!")
    for col in range(n_col):
        for row, condition in zip(range(n_row), conditions[col*n_row:col*n_row+n_row]):
            epochs[condition].average().plot(axes=ax[row, col])
            ax[row, col].set_title("evoked response "+condition)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        fig.savefig(outfile)
        plt.close()
    if plot_topo:
        evokeds = [epochs[condition].average() for condition in conditions]
        plot_evoked_topo(evokeds)

def plot_fit(x, y, function, p0=None, xtitle=None, ytitle=None):
    """
    Make a scatterplot for the x/y pairs then fit the given function to the
    data and plot the fitted line
    """
    plt.scatter(x,y, color="k", marker=".", alpha=0.3)
    par, _ = curve_fit(function, x, y, p0=p0)
    par = np.round(par,2)
    label = "a/(x-c)+b with a=%s, b=%s, c=%s" %(par[0],par[1],par[2])
    plt.plot(np.unique(x),function(np.unique(x),*par), color="k",label=label)
    plt.legend()

def plot_cluster_analysis(epochs_1, epochs_2, test_stat, clusters, p_values, title):

    times = epochs_1.times
    info = epochs_1.info
    for i_c, c in enumerate(clusters):
        if p_values[i_c] <= 0.05: # found significant cluster
            fig, ax = plt.subplots(2)
            fig.suptitle(title)
            ax[0].plot(times, np.abs(epochs_1.average().data.mean(axis=0) - epochs_2.average().data.mean(axis=0)))
            channels = np.unique(np.where(c!=0)[0])
            samples = np.unique(np.where(c!=0)[1])
            for ch in channels:
                ax[1].plot(times, test_stat[ch], label=info["ch_names"][ch])
            ax[1].axvspan(times[min(samples)], times[max(samples)], color='r', alpha=0.3)
            plt.legend()
            print(i_c)

def compare_evokeds(dataset, groups, mode="gfp", title=None, subtitles=None, tmin=None, tmax=None, vline=None, baseline=None, color_coding=None, ci=None):
    """
    Compare different evoked responses
    """
    cNorm = colors.Normalize(vmin=min(color_coding.values()), vmax=max(color_coding.values()))
    cmap = cmx.ScalarMappable(norm=cNorm, cmap="cool")
    if mode == "rms":
        stat_fun=global_rms
    elif mode == "gfp":
        stat_fun=global_field_power
    else:
        raise ValueError("Mode must be either 'rms' or 'gfp'!")
    if isinstance(dataset, list) and all(isinstance(d,Evoked) for d in dataset):
        if ci is not None:
            raise  ValueError("Need Epochs to compute Confidence Interval")
        names = [evoked.comment for evoked in dataset]
    elif isinstance(dataset, Epochs) or isinstance(dataset, BaseEpochs):
        names = list(dataset.event_id.keys())
    else:
        raise ValueError("Input must be either a list of evoked responses or epoched data")
    fig, ax = plt.subplots(len(groups),sharex="all", sharey="all")
    fig.colorbar(cmap, ax=ax)
    if title is not None:
        fig.suptitle("%s mode: %s" % (title, mode))
    for i, group in enumerate(groups):
        for g in group:
            idx = np.where(np.array(names)==g)[0][0]
            if isinstance(dataset, list): # data is evoked
                data = dataset[idx]
            else: # data is epochs
                data = dataset[names[idx]]
            data.crop(tmin, tmax)
            times = data.times
            data.apply_baseline(baseline)
            if isinstance(dataset, list): # data is evoked
                raw_data = data.data
            else:
                raw_data = data._data
            data = stat_fun(raw_data)
            ax[i].plot(times, data, color=cmap.to_rgba(color_coding[names[idx]]), label=g)
            if ci is not None:
                ci_low, ci_high = bootstrap_confidence_interval(raw_data, stat_fun=stat_fun)
                ax[i].fill_between(times, ci_low, ci_high)
            if subtitles is not None:
                ax[i].set_title(subtitles[i])
            if vline is not None:
                ax[i].axvline(vline, linestyle="--", c="black")


def rational(x, a, b, c):
    return a/(x-c)+b

