import numpy as np

_scaling = 10**6  # scaing factor for the data


def set_scaling_factor(scaling_factor):
    global _scaling
    _scaling = scaling_factor


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
    rms = np.sqrt(np.mean(evoked**2))*_scaling  # noise rms in micro volts
    del epochs_tmp
    return rms


def signal_to_noise(epochs, signal_interval=(0.68, 0.72)):
    """
    Calculate the RMS in the signal interval divide it by the noise rms
    """
    signal = epochs.copy()
    signal.crop(signal_interval[0], signal_interval[1])
    n_rms = noise_rms(epochs)
    s_rms = np.sqrt(np.mean(signal.average().data**2))*_scaling
    snr = s_rms/n_rms  # signal rms divided by noise rms
    return snr


def global_field_power(data):
    """
    Compute the global field power which is the standard deviation of the
    squared evoked response if input data is 3-dimensional it is considered
    as epoched data and averaged over the first dimension (number
    of epochs) to obtain the evoked response.
    """
    if data.ndim == 3:
        evoked = np.mean(data, axis=0)
    elif data.ndim == 2:
        evoked = data
    else:
        raise ValueError("Data must be either 2-dimensional (evoked responses)"
                         "or 3-dimensional (epochs)")
    return np.std(evoked**2, axis=0)


def global_rms(data):
    """
    Compute the square-root of the mean over all channels after squaring
    (= root mean square).If input data is 3-dimensional it is considered as
    epoched data and averaged over the first dimension (numberof epochs)
    to obtain the evoked response.
    """
    if data.ndim == 3:
        evoked = np.mean(data, axis=0)
    elif data.ndim == 2:
        evoked = data
    else:
        raise ValueError("Data must be either 2-dimensional (evoked responses)"
                         " or 3-dimensional (epochs)")
    return np.sqrt(np.mean(evoked**2, axis=0))
