import mne
import numpy as np
from scipy.signal import detrend


class JointDecorrelation:

    """
    Implementation of Joint Decorrelation from de Cheveigné & Parra [2014] to
    1) Find components (linear combinations of sensors that maximize either evoked
    activity or the difference between two experimental conditions and
    2) Reproject a subset of components onto the sensor space to filter the data
    """

    def __init__(self, kind="evoked"):
        if kind not in ["evoked", "difference"]:
            raise ValueError("kind must be either 'evoked' or 'difference'!")
        self.kind = kind
        self.mixing = None
        self.unmixing = None

    def fit(self, epochs, keep1=None, keep2=None, keep3=None,
            condition1=None, condition2=None, detrending=True):

        if keep1 is None:
            keep1 = epochs.info["nchan"]
        if keep2 is None:
            keep2 = keep1
        if keep3 is None and self.kind == "difference":
            keep3 = keep2
        if condition1 is None and condition2 is None and self.kind == "difference":
            if len(list(epochs.event_id.keys())) != 2:
                raise ValueError("If conditions are not specified, "
                                 "epochs must contain exactly two types of events!")
            else:
                condition1, condition2 = list(epochs.event_id.keys())
        n_epochs, n_channels, n_times = epochs._data.shape
        if self.kind == "evoked":
            X = epochs._data
        elif self.kind == "difference":  # sort by condition
            X = np.concatenate([epochs[condition1]._data, epochs[condition1]._data])
        if detrending:
            X = detrend(X, axis=-1, type="linear")
            print("detrending data...")
        X = np.transpose(epochs._data, (1, 0, 2))
        X = X.reshape(n_channels, n_epochs * n_times).T
        L = np.tile(np.identity(n_times), n_epochs)
        P, N, Q = JointDecorrelation._transform(X, L, keep1, keep2)
        self.unmixing = P @ N @ Q
        self.mixing = Q.T @ np.diag(1/np.diag(N)) @ P.T
        if self.kind == "difference":
            L = np.concatenate([np.tile(np.identity(n_times), epochs[condition1]._data.shape[0]),
                                np.tile(np.identity(n_times), epochs[condition2]._data.shape[0])
                                * -1], axis=1)
            Y = X @ self.unmixing
            P, N, Q = JointDecorrelation._transform(Y, L, keep2, keep3)
            self.unmixing = self.unmixing @ P @ N @ Q
            self.mixing = Q.T @ np.diag(1/np.diag(N)) @ P.T @ self.mixing

    def get_components(self, epochs):
        n_epochs, n_channels, n_times = epochs._data.shape
        X = np.transpose(epochs._data, (1, 0, 2))
        X = X.reshape(n_channels, n_epochs * n_times).T
        Y = X @ self.unmixing
        Y = np.reshape(Y.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
        info = mne.create_info(Y.shape[1], epochs.info["sfreq"], ch_types="eeg")
        return mne.EpochsArray(Y, info, epochs.events, epochs.tmin)

    def reproject_components(self, epochs):
        n_epochs, n_channels, n_times = epochs._data.shape
        X = np.transpose(epochs._data, (1, 0, 2))
        X = X.reshape(n_channels, n_epochs * n_times).T
        Y = X @ self.unmixing
        X = Y @ self.mixing
        X = np.reshape(X.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
        return mne.EpochsArray(X, epochs.info, epochs.events, epochs.tmin)

    def sort_epochs(epochs, condition1, condition2):

        X = np.concatenate([epochs[condition1]._data, epochs[condition1]._data])
        events = np.concatenate([epochs.events[epochs.events[:, 2] == epochs.event_id[condition1]],
                                 epochs.events[epochs.events[:, 2] == epochs.event_id[condition2]]])
        # Do we need to chage the event time  in column 0 as well?
        return X, events

    def _transform(X, L, keep1, keep2):

        C0 = X.T @ X  # Data covariance Matrix
        D, P = np.linalg.eig(C0)  # eigendecomposition of C0
        idx = np.argsort(D)[::-1][0:keep1]  # sort array by descending magnitude
        D = D[idx]
        P = P[:, idx]
        # STEP2: Normalize to "sphere" the signal
        N = np.diag(np.sqrt(1. / D))  # diagonal whitening matrix
        Z = X @ P @ N  # sphered signal
        # STEP3: apply bias filter to the sphered data
        Zbar = L @ Z  # bias filter data
        # STEP4: Do a PCA on the bias filtered data
        C1 = Zbar.T @ Zbar  # Covariance of filtered data
        Dz, Q = np.linalg.eig(C1)  # eigendecomposition of bias filtered covariance matrix
        idx = np.argsort(Dz)[::-1][0:keep2]
        Dz = Dz[idx]
        Q = Q[:, idx]
        return P, N, Q


def surrogate_data(epochs, kind="shuffle"):

    n_epochs, n_channels, n_times = epochs._data.shape
    data = np.concatenate(epochs._data, axis=0)
    data = np.transpose(epochs._data, (1, 0, 2))
    data = data.reshape(n_channels, n_epochs * n_times).T

    if kind == "shuffle":
        [np.random.shuffle(data[:, i]) for i in range(n_channels)]

    data = np.reshape(data.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
    return mne.EpochsArray(data, epochs.info, epochs.events, epochs.tmin)


def bootstrap_components(epochs, n_bootstrap=1000, keep1=40, keep2=15, ci=0.95, invert=True):
    # pre allocate data matrix for bootstrap runs
    # TODO: implement for kind=="difference"
    ci = (((1 - ci) / 2) * 100, ((1 - ((1 - ci) / 2))) * 100)
    n_epochs, n_channels, n_times = epochs._data.shape
    data = np.zeros([n_bootstrap, keep2, n_times])
    jd = JointDecorrelation(kind="evoked")
    jd.fit(epochs, keep1=keep1, keep2=keep2)
    for i in range(n_bootstrap):
        indices = np.random.choice(np.arange(n_epochs, dtype=int), n_epochs, replace=True)
        permutated_epochs = epochs.copy()
        permutated_epochs._data = permutated_epochs._data[indices, :, :]
        jd_i = JointDecorrelation(kind="evoked")
        jd_i.fit(permutated_epochs, keep1=keep1, keep2=keep2)
        components_i = jd.get_components(permutated_epochs).average().data
        if invert:  # check if dot product of vectors is negative, if so invert
            for nc in range(components_i.shape[0]):
                if(np.dot(jd.mixing[nc, :], jd_i.mixing[nc, :])) < -1e-05:
                    components_i[nc, :] = components_i[nc, :]*-1
        data[i, :, :] = components_i
        ci_low, ci_up = np.percentile(data, ci, axis=0)
    return ci_low, ci_up


if __name__ == "__main":

    from matplotlib import pyplot as plt
    data_path = mne.datasets.sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    events = mne.read_events(event_fname)
    tmin, tmax = -0.2, 0.5
    reject = dict(eeg=80e-6)
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, picks=('eeg'),
                        baseline=None, reject=reject, preload=True)
    epochs._data = detrend(epochs._data, axis=-1)
    con1, con2 = "1", "2"  # the conditions to compare
    epochs = epochs[[con1, con2]]  # only use auditory events
    jd = JointDecorrelation(kind="evoked")
    jd.fit(epochs, keep1=40, keep2=5)

    components = jd.get_components(epochs).average().data
    ci_low, ci_up = bootstrap_components(
        epochs, n_bootstrap=1000, keep1=40, keep2=15, ci=0.95, invert=True)

    plt.plot(epochs.times, components[0, :])
    plt.fill_between(epochs.times, ci_low[0, :], ci_up[0, :], alpha=.5)
