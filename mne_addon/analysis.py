from mne.evoked import Evoked
from mne import combine_evoked
from mne.epochs import Epochs
from mne.channels import make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
_scaling = 10**6  # scaing factor for the data


def set_scaling_factor(scaling_factor):
    global _scaling
    _scaling = scaling_factor


def permutation_cluster_analysis(epochs, n_permutations=1000, plot=True):
    """
    Do a spatio-temporal cluster analyis to compare experimental conditions.
    """
    # get the data for each event in epochs.evet_id transpose because the cluster test requires
    # channels to be last. In this case, inference is done over items. In the same manner, we could
    # also conduct the test over, e.g., subjects.
    tfce = dict(start=.2, step=.2)
    time_unit = dict(time_unit="s")
    events = list(epochs.event_id.keys())
    if plot:
        if len(events) == 2:  # When comparing two events subtract evokeds
            evoked = combine_evoked([epochs[events[0]].average(), -epochs[events[1]].average()],
                                    weights='equal')
            title = "%s vs %s" % (events[0], events[1])
        elif len(events) > 2:  # When comparing more than two events verage them
            evoked = combine_evoked([epochs[e].average() for e in events], weights='equal')
            evoked.data /= len(events)
            title = ""
            for e in events:
                title += e+" + "
            title = title[:-2]
        evoked.plot_joint(title=title, ts_args=time_unit, topomap_args=time_unit)
        X = [epochs[e].get_data().transpose(0, 2, 1) for e in events]
        t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(X, tfce, n_permutations)
        significant_points = cluster_pv.reshape(t_obs.shape).T < .05
        selections = make_1020_channel_selections(evoked.info, midline="12z")
        fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
        axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
        evoked.plot_image(axes=axes, group_by=selections, colorbar=False, show=False,
                          mask=significant_points, show_names="all", titles=None,
                          **time_unit)
        plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=.3, label="µV")

    plt.show()


def noise_rms(epochs):
    """
    Calculate the noise that remains after averaging in the ERP following
    the procedure from Schimmel(1967): invert every other trial then average.
    Afterwards calculate the root mean square across the whole EPR.
    """
    epochs_tmp = epochs.copy()
    n_epochs = epochs._data.shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp._data[i, :, :] = -epochs_tmp._data[i, :, :]
    evoked = epochs_tmp.average().data
    rms = np.sqrt(np.mean(evoked**2))
    del epochs_tmp
    return rms


def signal_to_noise(epochs, signal_interval=(0.68, 0.72)):
    """
    Calculate the RMS in the signal interval divide it by the noise rms
    """
    signal = epochs.copy()
    signal.crop(signal_interval[0], signal_interval[1])
    n_rms = noise_rms(epochs)
    s_rms = np.sqrt(np.mean(signal.average().data**2))
    snr = s_rms/n_rms  # signal rms divided by noise rms
    return snr


def get_evoked_data(data):
    """
    Compute a global estimate of the eeg data. If mode = gfp, return the global
    field power which is the standard deviation of the squared evoked response.
    If mode = rms return the root mean square over all channels.
    If inumput data is 3-dimensional it is considered
    as epoched data and averaged over the first dimension (number
    of epochs) to obtain the evoked response.
    """
    if isinstance(data, Evoked):
        data = data._data
    elif isinstance(data, Epochs):
        data = data.data
    if data.ndim == 3:  # average over epochs
        data = np.mean(data, axis=0)
    return data


def rms(data):
    data = get_evoked_data(data)
    return np.sqrt(np.mean(data**2, axis=0))


def gfp(data):
    data = get_evoked_data(data)
    return np.std(data**2, axis=0)


def find_peaks(data, min_dist=1, thresh=0.3, degree=None):
    """
    Find peaks in the data, optionally apply a baseline before.
    min_dist: minimum distance between peaks, the biggest peak is preferred.
    thresh: float between 0.0 and 1.0, normalized detection threshold
    degree: degree of the polynomial that will estimate the data baseline.
    if None no baseline detection is done
    """
    import peakutils
    if degree is not None:
        base = peakutils.baseline(data, degree)
    else:
        base = np.zeros(len(data))
    peak_idx = peakutils.indexes(data-base, thres=thresh, min_dist=min_dist)
    return peak_idx, base


def peak_clustering(latency, amplitude, k=3, max_k=10, plot=True):
    """
    Use kmeans clustering to divide peak amplitudes of time series data into
    temporal clusters. The clustering is done for different cluster numbers.
    Each time, the squared sum of errors (SSE) is calculated to see which is
    the optimal number of clusters for the dataset. Creates a plot with two
    subplots if plot=True. Top: SSE versus number of clusters, bottom:
    scatterplot of amplitudes vs latency, divided into clusters by color.
    Parameters:
        latency: array-like containing the peak latencies
        amplitude: array-like containing the peak amplitudes
        k: int, number of clusters for the clustered scatterplot
        max_k: int, number of clusters to try for the SSE plot
    Returns:
        cluster_intervals:

    """
    data = np.array([latency, amplitude]).T
    sse = {}
    for this_k in range(1, max_k):
        kmeans = KMeans(
            n_clusters=this_k, max_iter=1000).fit(data)
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[this_k] = kmeans.inertia_
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    clusters = kmeans.predict(data)
    centers = kmeans.cluster_centers_
    if plot:
        fig, ax = plt.subplots(2)
        fig.suptitle("Peak Amplitude Clustering")
        ax[0].plot(list(sse.keys()), list(sse.values()), c="black")
        ax[0].set_xlabel("Number of cluster")
        ax[0].set_ylabel("SSE")
        ax[1].scatter(latency, amplitude, c=clusters)
        ax[1].scatter(centers[:, 0], centers[:, 1], c='black', s=100)
        ax[1].set_xlabel("Peak Latency")
        ax[1].set_ylabel("Peak Amplitude")
        plt.show()
    cluster_intervals = []
    for i in range(k):
        idx = np.where(clusters == i)[0]
        cluster_intervals.append((min(latency[idx]), max(latency[idx])))
    return cluster_intervals


"""
def wavelet_tf(data, times, sample_rate, channel="Fz", frex=None,
               n_cycles=[4, 6, 8], baseline=None, crop=None, wavelet_dur=4,
               plot=True):
    #Time-frequency analysis of epoched data using wavelets
    if not (isinstance(data, np.ndarray) and data.ndim == 2):
        raise ValueError("data must a 2-d matrix with trials x time!")
    if crop is None:
        print("WARNING! Not cropping the data will result in edge artifacts!")
    elif isinstance(crop, float):
        n_crop = int(crop*sample_rate)
    elif isinstance(crop, int):
        n_crop = crop
    else:
        raise ValueError("crop must be None, an integer or a float")
    if frex is None:
        frex = np.linspace(2, 40, 50)
    if baseline is None:
        print("WARNING! No baseline specified!")
    elif not ((isinstance(baseline, tuple) or isinstance(baseline, list)) and
              len(baseline) == 2):
        raise ValueError("Baseline must be a tuple or list of length two \n"
                         "containing start and stop of the baseline interval "
                         "in seconds!")
    else:  # get start an stop indices for baseline
        bi = (np.where(epochs.times == baseline[0])[0][0],
              np.where(epochs.times == baseline[1])[0][0])
    wavelet_time = np.linspace(-(wavelet_dur/2), wavelet_dur/2,
                               int(epochs.info["sfreq"]*wavelet_dur)+1)
    half_wave = int((len(wavelet_time)-1)/2)
    # length of the result of convolution is N+M-1
    nConv = len(wavelet_time)+len(data.flatten())-1
    # initialize output time-frequency data
    tf_data = np.zeros([len(n_cycles), len(frex), data.shape[1]])

    dataX = np.fft.fft(data.flatten(), n=nConv)
    # loop over cycles and frequencies
    for c, cycles in enumerate(n_cycles):
        for f, freq in enumerate(frex):
            # create complex morlet wavelet, compute it's FT and normalize
            s = cycles/(2*np.pi*freq)
            cmw = np.exp(2j*np.pi*freq*wavelet_time) * np.exp(
                -wavelet_time ** 2/(2*s ** 2))
            cmwX = np.fft.fft(cmw, n=nConv)
            cmwX = cmwX/max(cmwX)
            # convolve by multiplying FT of data and wavelet:
            signal = np.fft.ifft(cmwX*dataX)
            signal = signal[half_wave:-half_wave]  # trim edges
            signal = signal.reshape(data.shape)  # reshape to trials x times
            # put the average power across trials into the big matrix:
            tf_data[c, f, :] = np.mean(np.abs(signal)**2, axis=0)

    # Compute the baseline for each number of cycles and each frequency
    baseline_data = tf_data[:, :, bi[0]:bi[1]].mean(axis=2)
    tf_data = 10*np.log10(tf_data / baseline_data[:, :, np.newaxis])
    tf_data = tf_data[:, :, n_crop:-n_crop+1]
    if plot:
        x, y = np.meshgrid(epochs.times[n_crop:-n_crop+1], frex)
        fig, ax = plt.subplots(1, len(n_cycles), sharex=True, sharey=True)
        fig.suptitle("baseline from %s to %s seconds"
                     % (baseline[0], baseline[1]))
        for i, cycles in enumerate(n_cycles):
            z = tf_data[i, :, :]
            cax = ax[i].contour(x, y, z, linewidths=0.3,
                                colors="k", norm=colors.Normalize())
            cax = ax[i].contourf(x, y, z, norm=colors.Normalize(),
                                 cmap=plt.cm.jet)
            ax[i].set_title("wavelet with %s cycles" % (cycles))
        fig.colorbar(cax)
        plt.show()
"""
