from matplotlib import pyplot as plt
from mne.viz import plot_evoked_topo
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import patches, colors, cm
from mne.stats import bootstrap_confidence_interval
from mne_addon.analysis import gfp, rms
from mne.epochs import Epochs, BaseEpochs
from mne.evoked import Evoked
from sklearn.cluster import KMeans


def visualize_auto_reject(ar):

    fig, ax = plt.subplots(2)
    # plotipyt histogram of rejection thresholds
    ax[0].set_title("Rejection Thresholds")
    ax[0].hist(1e6 * np.array(list(ar.threshes_.values())), 30,
               color='g', alpha=0.4)
    ax[0].set(xlabel='Threshold (μV)', ylabel='Number of sensors')
    # plot cross validation error:
    loss = ar.loss_['eeg'].mean(axis=-1)  # losses are stored by channel type.
    im = ax[1].matshow(loss.T * 1e6, cmap=plt.get_cmap('viridis'))
    ax[1].set_xticks(range(len(ar.consensus)))
    ax[1].set_xticklabels(['%.1f' % c for c in ar.consensus])
    ax[1].set_yticks(range(len(ar.n_interpolate)))
    ax[1].set_yticklabels(ar.n_interpolate)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax[1].add_patch(rect)
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set(xlabel=r'Consensus percentage $\kappa$',
              ylabel=r'Max sensors interpolated $\rho$',
              title='Mean cross validation error (x 1e6)')
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(_out_folder / Path("reject_epochs.pdf"), dpi=800)
    plt.close()


def plot_multiple_ERP(epochs, n_row=3, n_col=2, outfile=None, title=None,
                      plot_topo=True):
    """
    Plot ERP for the different conditions
    """
    fig, ax = plt.subplots(n_row, n_col, sharex=True, figsize=(16, 9))
    if title:
        fig.suptitle(title)
    conditions = list(epochs.event_id.keys())
    if n_row*n_col != len(conditions):
        raise ValueError("Size of the plot and number of experimental"
                         "conditions dont match!")
    for col in range(n_col):
        for row, condition in zip(range(n_row),
                                  conditions[col*n_row:col*n_row+n_row]):
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
    plt.scatter(x, y, color="k", marker=".", alpha=0.3)
    par, _ = curve_fit(function, x, y, p0=p0)
    par = numpy.round(par, 2)
    label = "a/(x-c)+b with a=%s, b=%s, c=%s" % (par[0], par[1], par[2])
    plt.plot(numpy.unique(x), function(numpy.unique(x), *par), color="k",
             label=label)
    plt.legend()


def plot_cluster_analysis(epochs_1, epochs_2, test_stat, clusters, p_values,
                          title):

    times = epochs_1.times
    info = epochs_1.info
    for i_c, c in enumerate(clusters):
        if p_values[i_c] <= 0.05:  # found significant cluster
            fig, ax = plt.subplots(2)
            fig.suptitle(title)
            ax[0].plot(times, numpy.abs(epochs_1.average().data.mean(
                axis=0) - epochs_2.average().data.mean(axis=0)))
            channels = numpy.unique(numpy.where(c != 0)[0])
            samples = numpy.unique(numpy.where(c != 0)[1])
            for ch in channels:
                ax[1].plot(times, test_stat[ch], label=info["ch_names"][ch])
            ax[1].axvspan(times[min(samples)], times[max(samples)],
                          color='r', alpha=0.3)
            plt.legend()
            print(i_c)


def compare_evokeds(dataset, groups, mode="gfp", title=None, subtitles=None,
                    vline=None, color_coding=None, ci=None):
    """
    Compare different evoked responses
    """
    cNorm = colors.Normalize(vmin=min(color_coding.values()),
                             vmax=max(color_coding.values()))
    cmap = cm.ScalarMappable(norm=cNorm, cmap="cool")
    if mode == "gfp":
        stat_fun = gfp
    elif mode == "rms":
        stat_fun = rms
    else:
        raise ValueError("Mode must be either 'rms' or 'gfp'!")
    if isinstance(dataset, list) and all(isinstance(d, Evoked) for d in dataset):
        if ci is not None:
            raise ValueError("Need Epochs to compute Confidence Interval")
        names = [evoked.comment for evoked in dataset]
    elif isinstance(dataset, Epochs) or isinstance(dataset, BaseEpochs):
        names = list(dataset.event_id.keys())
    else:
        raise ValueError(
            "Input must be either a list of evoked responses or epoched data")
    fig, ax = plt.subplots(len(groups), sharex="all", sharey="all")
    fig.colorbar(cmap, ax=ax)
    if title is not None:
        fig.suptitle("%s mode: %s" % (title, mode))
    for i, group in enumerate(groups):
        for g in group:
            idx = numpy.where(numpy.array(names) == g)[0][0]
            if isinstance(dataset, list):  # data is evoked
                data = dataset[idx]
            else:  # data is epochs
                data = dataset[names[idx]]
            times = data.times
            if isinstance(dataset, list):  # data is evoked
                raw_data = data.data
            else:
                raw_data = data._data
            data = stat_fun(raw_data)
            ax[i].plot(times, data, color=cmap.to_rgba(
                color_coding[names[idx]]), label=g)
            if ci is not None:
                ci_low, ci_high = bootstrap_confidence_interval(raw_data, ci=ci, stat_fun=stat_fun)
                ax[i].fill_between(times, ci_low, ci_high, color=cmap.to_rgba(
                    color_coding[names[idx]]), alpha=0.3)
            if subtitles is not None:
                ax[i].set_title(subtitles[i])
            if vline is not None:
                ax[i].axvline(vline, linestyle="--", c="black")


def bootstrap_comparison(list_of_epochs, stat_fun=None, color_coding=None, title=None, subtitles=None,
                         vline=None, ci=None):
    """
    Compare different evoked responses
    """
    if stat_fun is None:
        stat_fun = gfp
    cNorm = colors.Normalize(vmin=min(color_coding.values()),
                             vmax=max(color_coding.values()))
    cmap = cm.ScalarMappable(norm=cNorm, cmap="cool")

    fig, ax = plt.subplots(len(list_of_epochs), sharex="all", sharey="all")
    fig.colorbar(cmap, ax=ax)
    if title is not None:
        fig.suptitle("%s mode: %s" % (title))
    for i, epochs in enumerate(list_of_epochs):
        for event in epochs.event_id.keys():
            data = epochs[event].get_data()
            data_to_plot = stat_fun(data)
            ax[i].plot(epochs.times, data_to_plot, color=cmap.to_rgba(
                color_coding[event]), label=event)
            if ci is not None:
                ci_low, ci_high = bootstrap_confidence_interval(data, ci=ci, stat_fun=stat_fun)
                ax[i].fill_between(epochs.times, ci_low, ci_high, color=cmap.to_rgba(
                    color_coding[event]), alpha=0.3)
        if subtitles is not None:
            ax[i].set_title(subtitles[i])
        if vline is not None:
            ax[i].axvline(vline, linestyle="--", c="black")


def peak_clustering(latency, amplitude, k=3, max_k=10):
    """
    Use kmeans clustering to divide peak amplitudes of time series data into
    temporal clusters. The clustering is done for different cluster numbers.
    Each time, the squared sum of errors (SSE) is calculated to see which is
    the optimal number of clusters for the dataset. Creates a plot with two
    subplots. Top: SSE versus number of clusters, bottom: scatterplot of
    amplitudes vs latency, divided into clusters by color.
    Parameters:
        latency: array-like containing the peak latencies
        amplitude: array-like containing the peak amplitudes
        k: int, number of clusters for the clustered scatterplot
        max_k: int, number of clusters to try for the SSE plot

    """
    data = numpy.array([latency, amplitude]).T
    fig, ax = plt.subplots(2)
    fig.suptitle("Peak Amplitude Clustering")
    sse = {}
    for this_k in range(1, max_k):
        kmeans = KMeans(
            n_clusters=this_k, max_iter=1000).fit(data)
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[this_k] = kmeans.inertia_
    ax[0].plot(list(sse.keys()), list(sse.values()), c="black")
    ax[0].set_xlabel("Number of cluster")
    ax[0].set_ylabel("SSE")

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    cluster = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    ax[1].scatter(latency, amplitude, c=cluster)
    ax[1].scatter(centers[:, 0], centers[:, 1], c='black', s=100)
    ax[1].set_xlabel("Peak Latency")
    ax[1].set_ylabel("Peak Amplitude")
    plt.show()


def rational(x, a, b, c):
    return a/(x-c)+b
