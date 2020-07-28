import mne
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path

epochs = mne.read_epochs(Path(os.environ["DATADIR"])/Path("eegl03/eegl03_mid-epo.fif",preload=True))
epochs = epochs["b-mb", "b-mt", "b-t"]

#Which preprocessing steps are nesseccary?
epochs.set_eeg_reference(["TP9", "TP10"])
epochs.filter(None, 30)
# epochs.apply_baseline((0.5,0.6))
epochs.crop(-0.1, 1)

keep_n1 = 30

X = epochs._data.reshape(
    epochs._data.shape[0]*epochs._data.shape[2], epochs._data.shape[1])  # The raw data
X -= X.mean(axis=0)  # center the data on 0

C0 = X.T @ X
D, P = np.linalg.eig(C0)

idx = np.argsort(D, axis=0)[::-1]  # sort by descending magnitude
keep = idx[0:keep_n1]  # only keep the first n1 components
remove = idx[keep_n1:]
D = D[keep]
P = P[:, keep]
# P[:, remove] = 0

N = np.diag(np.sqrt(1. / D))  # diagonal whitening matrix


fig, ax = plt.subplots(1, 3)
ax[0].imshow(C0)
ax[1].imshow(P)
ax[2].imshow(N)
plt.show()

plt.semilogy(D)
plt.show()

Z = X @ P @ N  # sphered signal
Z.shape

evokedZ = Z.reshape(epochs._data.shape[0], keep_n1, epochs._data.shape[2]).mean(axis=0)
for i in range(evokedZ.shape[0]):
    plt.plot(epochs.times, evokedZ[i, :])
plt.show()

L = np.tile(np.identity(epochs._data.shape[2]), epochs._data.shape[0])  # bias filter
Zbar = L @ Z  # filter data
C1 = Zbar.T @ Zbar  # Covariance of filtered data
Dz, Q = np.linalg.eig(C1)  # eigendecomposition of bias filtered covariance matrix
# sort by magnitude
idx = np.argsort(Dz, axis=0)[::-1]  # sort by descending magnitude
Dz = Dz[idx]
Q = Q[:,idx]

plt.semilogy(D)
plt.semilogy(Dz)

Ybar = Zbar @ Q
W = P @ N @ Q
Y = X @ W

Y.shape
W.shape

Y = Y.reshape(epochs._data.shape[0], keep_n1, epochs._data.shape[2])
evokedY = Y.mean(axis=0)

fig, ax = plt.subplots(3, 2)
for n in range(6):
    if n < 3:
        j = 0
        i = n
    else:
        j = 1
        i = n-3
    ax[i, j].plot(epochs.times, evokedY[n, :])
    ax[i, j].set(title="component %s" % (n))
    ax[i, j].axvline(0.6, color="red", linestyle="--")
plt.show()

# plot topomap for component
