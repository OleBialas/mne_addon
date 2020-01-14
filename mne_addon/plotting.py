from matplotlib import pyplot as plt
from mne.viz import plot_evoked_topo
from scipy.optimize import curve_fit
import numpy as np
from mne.viz import plot_compare_evokeds

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

def gfp(epochs, groups, tmin=None, tmax=None, vlines="auto", baseline=None, cmap="cool", color_coding=None, names=None):
    """
    compare different evoked responses by plotting the global field power (gfp)
    """
    epochs_tmp = epochs.copy()
    epochs_tmp.crop(tmin, tmax)
    epochs_tmp.apply_baseline(baseline)
    fig, ax = plt.subplots(len(groups))
    # color_coding={"t-mt":0.66, "t-mb":0.33, "t-b":0, "b-mt":0.66, "b-mb":0.33, "b-t":1.0}
    #names={"t-mt":"midtop", "t-mb":"midbottom", "t-b":"bottom", "b-mt":"midtop", "b-mb":"midbottom", "b-t":"top"}
    # groups=[["t-mt", "t-mb", "t-b"],["b-mt", "b-mb", "b-t"]]
    for i, group in enumerate(groups):
        print(group)
        evokeds = dict()
        for g in group:
            if names is not None:
                evokeds[names[g]] = epochs_tmp[g].average()
            else:
                evokeds[g] = epochs_tmp[g].average()
        if color_coding is not None:
            colors = [color_coding[g] for g in group]
        else:
            colors=None
        plot_compare_evokeds(evokeds, cmap=cmap,colors=colors, axes=ax[i], vlines=vlines, truncate_xaxis=False)
    del epochs_tmp

def square_sum(x):
    return np.sum(x ** 2, axis=0)

def rational(x, a, b, c):
    return a/(x-c)+b

