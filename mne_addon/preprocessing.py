from mne import set_eeg_reference, channels, events_from_annotations
from mne.epochs import Epochs
from autoreject import AutoReject, Ransac
import numpy as np
from mne.preprocessing.ica import ICA, corrmap, read_ica
import os
from pathlib import Path
from matplotlib import pyplot as plt, patches
from mne.io import read_raw_brainvision


def auto_reject(epochs, **kwargs):
    """
    run the auto reject pipeline on epoched data. Return the instance of AutoReject and the cleaned epochs.
    for a detailed description on how the algorithm works and what the arguments are, see https://autoreject.github.
    """
    ar = AutoReject(**kwargs, verbose="tqdm")
    epochs = ar.fit_transform(epochs)
    return ar, epochs


def filtering(raw, notch=None, highpass=None, lowpass=None,
              fir_window="hamming", fir_design="firwin"):
    """
    Filter the data. Make a 2 by 2 plot with time
    series data and power spectral density before and after.
    """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("before removing power line noise")
    ax[1].set_title("after removing power line noise")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    raw.plot_psd(average=True, area_mode=None, ax=ax[0], show=False)
    if notch is not None:  # notch filter at 50 Hz and harmonics
        raw.notch_filter(freqs=notch, fir_window=fir_window,
                         fir_design=fir_design)
    if lowpass is not None:  # lowpass filter at 50 Hz
        raw.filter(h_freq=lowpass, l_freq=None, fir_window=fir_window,
                   fir_design=fir_design)
    if highpass is not None:  # lowpass filter at 50 Hz
        raw.filter(h_freq=None, l_freq=highpass, fir_window=fir_window,
                   fir_design=fir_design)
    raw.plot_psd(average=True, area_mode=None, ax=ax[1], show=False)
    fig.tight_layout()
    fig.savefig(_out_folder/Path("remove_power_line_noise.pdf"), dpi=800)
    plt.close()
    return raw


def read_brainvision(file_name, electrode_mapping=None, electrode_montage=None, preload=False):
    """ load data in the brainvision format, optionally rename electrodes and apply an electrode montage

    Arguments:
        file_name (str): Path to the raw data to load
        electrode_mapping (dict): New electrode names, the key is the channel number and the value the new name.
            for example {"1": "Fp1"} will rename the first electrode to "Fp1"
        electrode_montage (str | instance of channels.montage): Position of the electrodes. Can either be the path
            to a .bvef file or an instance of mne.channels.montage.DigMontage
        preload (bool): preload the data
    Returns:
        instance of mne.io.Raw: the raw data
        """
    raw = read_raw_brainvision(file_name, preload=preload)
    if electrode_mapping is not None:
        raw.rename_channels(electrode_mapping)
    if electrode_montage is not None:
        if isinstance(electrode_montage, (str, Path)):
            electrode_montage = channels.read_custom_montage(electrode_montage)
        raw.set_montage(electrode_montage)
    return raw


def interpolate_bads(epochs, ransac_parameters):
    ransac = Ransac(**ransac_parameters, verbose="tqdm")
    evoked = epochs.average()  # for plotting
    epochs = ransac.fit_transform(epochs)
    evoked.info["bads"] = ransac.bad_chs_
    # plot evoked response with and without interpolated bads:
    fig, ax = plt.subplots(2)
    evoked.plot(exclude=[], axes=ax[0], show=False)
    ax[0].set_title('Before RANSAC')
    evoked = epochs.average()  # for plotting
    evoked.info["bads"] = ransac.bad_chs_
    evoked.plot(exclude=[], axes=ax[1], show=False)
    ax[1].set_title('After RANSAC')
    fig.tight_layout()
    fig.savefig(_out_folder/Path("interpolate_bad_channels.pdf"), dpi=800)
    plt.close()
    return epochs


def robust_avg_ref(epochs, ransac_parameters, apply=True):
    """
    Create a robust average reference by first interpolating the bad channels
    to exclude outliers. The reference is applied as a projection. Return
    epochs with reference projection applied if apply=True
    """
    ransac = Ransac(**ransac_parameters, verbose="tqdm")
    epochs_tmp = epochs.copy()
    epochs_tmp = ransac.fit_transform(epochs)
    set_eeg_reference(epochs_tmp, ref_channels="average", projection=True)
    robust_avg_proj = epochs_tmp.info["projs"][0]
    del epochs_tmp
    epochs.info["projs"].append(robust_avg_proj)
    if apply:
        epochs.apply_proj()
    return epochs


def reject_ica(inst, reference, n_components=0.99, method="fastica",
               corr_thresh=0.9, random_state=None, plot=False):

    if isinstance(reference, str):
        reference = read_ica(reference)

    ica = ICA(n_components=n_components, method=method)
    ica.fit(inst)
    labels = list(reference.labels_.keys())
    components = list(reference.labels_.values())

    for component, label in zip(components, labels):
        corrmap([reference, ica], template=(0, component[0]),
                plot=plot, label=label, threshold=corr_thresh)

    exclude = [item for subl in list(ica.labels_.values()) for item in subl]
    ica.apply(inst, exclude=exclude)

    return inst, ica
