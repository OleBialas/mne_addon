import mne
from matplotlib import pyplot as plt
import numpy as np

"""
# Use MNE data
data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info["bads"] = ["EEG 004", "EEG 008", "EEG 015", "EEG 018"]
raw.filter(1, 40, n_jobs=1, fir_design='firwin')
events = mne.read_events(event_fname)
tmin, tmax = -0.2, 0.5
reject = dict(eeg=80e-6)
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, picks=('eeg'),
                    baseline=None, reject=reject, preload=True)
epochs = epochs[["1", "2"]]  # only use auditory events
"""

con1, con2 = "b-mb", "b-mt"  # the conditions to compare
# use own data
epochs = mne.read_epochs("/home/ole/projects/elevation/data/eegl03/eegl03_mid-epo.fif")
epochs = epochs[[con1, con2]]
epochs.filter(None, 30)
tmin, tmax = 0.5, 1.1
epochs.crop(tmin, tmax)
# epochs.interpolate_bads()
# subtract mean per epoch
epochs._data -= np.expand_dims(epochs._data.mean(axis=2), axis=2)
X = np.concatenate([epochs[con1]._data, epochs[con2]._data])  # the whole data ordered by condition

n_epochs, n_channels, n_times = X.shape
# transpose so data is n_channels x n_epochs x n_times
X = np.transpose(X, (1, 0, 2))
# concatenate across epochs
X = X.reshape(n_channels, n_epochs * n_times).T  # in the MNE example this is transposed
X.shape

keep1 = 40  # number of components to keep after first rotation
keep2 = 15  # number of components to keep after second rotations

C0 = X.T @ X  # Data covariance Matrix
D, P = np.linalg.eig(C0)  # eigendecomposition of C0
idx = np.argsort(D)[::-1][0:keep1]  # sort array by descending magnitude
D = D[idx]
P = P[:, idx]

# check the result of the first PCA by plotting the components:
comp1 = X @ P
comp1 = np.reshape(comp1.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
evoked1 = mne.EvokedArray(np.mean(comp1, axis=0), mne.create_info(
    keep1, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked1.plot()

N = np.diag(np.sqrt(1. / D))  # diagonal whitening matrix

fig, ax = plt.subplots(1, 3)
ax[0].imshow(C0)
ax[1].imshow(P)
ax[2].imshow(N)
plt.show()

Z = X @ P @ N  # sphered signal

# epoch and plot the sphered data:
comp2 = np.reshape(Z.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
evoked2 = mne.EvokedArray(np.mean(comp2, axis=0), mne.create_info(
    keep1, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked2.plot()

L = np.tile(np.identity(epochs._data.shape[2]), epochs._data.shape[0])  # bias filter
Zbar = L @ Z  # filter data
C1 = Zbar.T @ Zbar  # Covariance of filtered data
Dz, Q = np.linalg.eig(C1)  # eigendecomposition of bias filtered covariance matrix
idx = np.argsort(Dz)[::-1]
Dz = Dz[idx]
Q = Q[:, idx]

# plot distribution of eigenvalues for both decompositions:
plt.semilogy(D, marker="o", label="first eigendecomposition")
plt.semilogy(Dz, marker="o", label="second eigendecomposition")
plt.show()

Ybar = Zbar @ Q  # Components from the bias filtered data
W = P @ N @ Q  # The unmixing matrix
Y = X @ W  # Components from the unfiltered data

# Y and Ybar is the same except for the scaling --> Ybar needs to be devided by
# n_epochs:

evoked3 = mne.EvokedArray(Ybar.T, mne.create_info(
    keep1, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked3.plot()

comp4 = np.reshape(Y.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
evoked4 = mne.EvokedArray(np.mean(comp4, axis=0), mne.create_info(
    keep1, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked4.plot()

# This Y is the activity that is maximally reproducible across trials. Now we want the activty
# that is distinct between conditions

X = Y[:, 0:keep2]  # only keep some components
C2 = Y.T @ Y

C0 = X.T @ X  # Data covariance Matrix
D, P = np.linalg.eig(C0)  # eigendecomposition of C0
idx = np.argsort(D)[::-1][0:keep1]  # sort array by descending magnitude
D = D[idx]
P = P[:, idx]
N = np.diag(np.sqrt(1. / D))  # diagonal whitening matrix


comp1 = X @ P
comp1 = np.reshape(comp1.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
evoked1 = mne.EvokedArray(np.mean(comp1, axis=0), mne.create_info(
    keep2, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked1.plot()

Z = X @ P @ N  # sphered signal

# epoch and plot the sphered data:
comp2 = np.reshape(Z.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
evoked2 = mne.EvokedArray(np.mean(comp2, axis=0), mne.create_info(
    keep2, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked2.plot()

# +1 identity matrix for con1, -1 for con2:
L = np.concatenate([
    np.tile(np.identity(epochs._data.shape[2]), epochs[con1]._data.shape[0]),
    np.tile(np.identity(epochs._data.shape[2]), epochs[con2]._data.shape[0])], axis=1)

Zbar = L @ Z  # filter data
C1 = Zbar.T @ Zbar  # Covariance of filtered data
Dz, Q = np.linalg.eig(C1)  # eigendecomposition of bias filtered covariance matrix
idx = np.argsort(Dz)[::-1]
Dz = Dz[idx]
Q = Q[:, idx]

plt.semilogy(D, marker="o", label="first eigendecomposition")
plt.semilogy(Dz, marker="o", label="second eigendecomposition")
plt.show()

Ybar = Zbar @ Q  # Components from the bias filtered data
W = P @ N @ Q  # The unmixing matrix
Y = X @ W  # Components from the unfiltered data

# Y and Ybar is the same except for the scaling --> Ybar needs to be devided by
# n_epochs:

evoked3 = mne.EvokedArray(Ybar.T, mne.create_info(
    keep2, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked3.plot()

comp4 = np.reshape(Y.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
evoked4 = mne.EvokedArray(np.mean(comp4, axis=0), mne.create_info(
    keep2, epochs.info["sfreq"], ch_types="eeg"), tmin=tmin)
evoked4.plot()