from mne.channels import read_custom_montage
from mne import set_eeg_reference, events_from_annotations
from mne.epochs import Epochs
from autoreject import AutoReject, Ransac
import numpy as np
from mne.preprocessing.ica import ICA, corrmap, read_ica
import os
from pathlib import Path
from matplotlib import pyplot as plt, patches
from mne.io import read_raw_brainvision


def run_pipeline(raw, parameters, out_folder):
    """
    Do all the preprocessing steps according to the parameters. Processed data,
    log files and plots are saved in out_folder.
    """
    global _out_folder
    _out_folder = out_folder
    if "filtering" in parameters:  # STEP1: filter the data
        print("removing power line noise...")
        raw = filtering(raw, **parameters["filtering"])
    if "epochs" in parameters:  # STEP2: epoch the data
        epochs = Epochs(raw, events_from_annotations(raw)[0],
                        **parameters["epochs"], preload=True)
        del raw
        # all other steps work on epoched data:
        if "rereference" in parameters:  # STEP3: re-reference the data
            print("computing robust average reference...")
            epochs = robust_avg_ref(epochs, parameters["rereference"])
        if "ica" in parameters:  # STEP4: remove blinks and sacchades
            epochs, ica = reject_ica(epochs, **parameters["ica"])
        if "interpolate" in parameters:  # STEP5: interpolate bad channels
            print("interpolating bad channels...")
            interpolate_bads(epochs, parameters["interpolate"])
        if "reject" in parameters:  # STEP6: epoch rejection / reparation
            print("repairing / rejecting bad epochs")
            epochs = reject_epochs(epochs, parameters["reject"])

        return epochs, ica
    else:
        return raw


def reject_epochs(epochs, autoreject_parameters):
    ar = AutoReject(**autoreject_parameters, verbose="tqdm")
    # for event in epochs.event_id.keys():
    #    epochs[event] = ar.fit_transform(epochs[event])
    epochs = ar.fit_transform(epochs)
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
    fig.savefig(_out_folder/Path("reject_epochs.pdf"), dpi=800)
    plt.close()
    return epochs


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


def read_brainvision(fname, apply_montage=True, preload=False):
    """Load brainvision data. If apply_montage=True, load and apply the standard
    montage for the 64-channel acticap. If add_ref=True add a reference
    channel with all zeros"""

    raw = read_raw_brainvision(fname, preload=preload)
    if apply_montage:
        mapping = {"1": "Fp1", "2": "Fp2", "3": "F7", "4": "F3", "5": "Fz",
                   "6": "F4", "7": "F8", "8": "FC5", "9": "FC1", "10": "FC2",
                   "11": "FC6", "12": "T7", "13": "C3", "14": "Cz", "15": "C4",
                   "16": "T8", "17": "TP9", "18": "CP5", "19": "CP1",
                   "20": "CP2", "21": "CP6", "22": "TP10", "23": "P7",
                   "24": "P3", "25": "Pz", "26": "P4", "27": "P8", "28": "PO9",
                   "29": "O1", "30": "Oz", "31": "O2", "32": "PO10",
                   "33": "AF7", "34": "AF3", "35": "AF4", "36": "AF8",
                   "37": "F5", "38": "F1", "39": "F2", "40": "F6", "41": "FT9",
                   "42": "FT7", "43": "FC3", "44": "FC4", "45": "FT8",
                   "46": "FT10", "47": "C5", "48": "C1", "49": "C2",
                   "50": "C6", "51": "TP7", "52": "CP3", "53": "CPz",
                   "54": "CP4", "55": "TP8", "56": "P5", "57": "P1",
                   "58": "P2", "59": "P6", "60": "PO7", "61": "PO3",
                   "62": "POz", "63": "PO4", "64": "PO8"}
        raw.rename_channels(mapping)
        montage = read_custom_montage(
            Path(os.environ["EXPDIR"])/Path("AS-96_REF.bvef"))
        raw.set_montage(montage)
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
