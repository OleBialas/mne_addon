import numpy as np
from mne.stats import permutation_cluster_test

_scaling = 10**6 #scaing factor for the data

def set_scaling_factor(scaling_factor):
    global _scaling
    _scaling = scaling_factor

def get_noise_rms(epochs):
    """
    Calculate the noise that remains after averaging in the ERP following
    the procedure from Schimmel(1967): invert every other trial then average.
    Afterwards calculate the root mean square across the whole EPR.
    """
    epochs_tmp = epochs.copy()
    n_epochs = epochs._data.shape[0]
    for i in range(n_epochs):
        if not i%2:
            epochs_tmp._data[i,:,:] = -epochs_tmp._data[i,:,:]
    evoked = epochs_tmp.average().data
    rms = np.sqrt(np.mean(evoked**2))*_scaling # noise rms in micro volts
    del epochs_tmp
    return rms

def get_snr(epochs, signal_interval=(0.68,0.72)):
    """
    Calculate the RMS in the signal interval divide it by the noise rms
    """
    signal = epochs.copy()
    signal.crop(signal_interval[0],signal_interval[1])
    noise_rms = get_noise_rms(epochs)
    signal_rms = np.sqrt(np.mean(signal.average().data**2))*_scaling
    snr = signal_rms/noise_rms
    return snr
