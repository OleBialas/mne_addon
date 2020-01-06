from os import environ
from os.path import join
from mne.io import read_raw_brainvision
from mne.channels import read_montage
from mne import set_eeg_reference, add_reference_channels, events_from_annotations
from mne.epochs import Epochs
from mne.filter import create_filter
from autoreject import AutoReject, Ransac
from autoreject.utils import set_matplotlib_defaults  # noqa
from mne.viz import plot_filter
import matplotlib.pyplot as plt  # noqa
import numpy as np
from mne.preprocessing.ica import ICA, corrmap, read_ica

def run_pipeline(raw, parameters):
	#initialize autoreject and
	ransac = Ransac(**parameters["ransac_params"])
	ar = AutoReject(**parameters["ar_params"])
	# Step 1: filter the data
	raw = filt_raw(raw, **parameters["filter_params"])
	# Step 2: epoch the data
	epochs = Epochs(raw, events_from_annotations(raw)[0], **parameters["epoch_params"])
	# Step 3: re-reference
	epochs = robust_average_reference(epochs, ransac)
	# Step 4: interpolate bad channels
	epochs = ransac.fit_transform(epochs)
	# Step 5: repair and reject epochs
	epochs, log = ar.fit_transform(epochs, return_log=True)
	# Step 6: ICA artifact removal
	epochs, ica = find_and_reject_ica_components(epochs, **parameters["ica_params"])
	log = dict(bad_channels = ransac.bad_chs_,n_reject=int(sum(log.bad_epochs)),bad_ica=ica.labels_)
	return epochs, log, ica

def filt_raw(raw, **filter_params):
	h = create_filter(data=raw._data[0],sfreq=raw.info["sfreq"], **filter_params)
	#if plot:
	#	plot_filter(h, raw.info["sfreq"], compensate=True)
	for data, ch in zip(raw._data, range(raw.info["nchan"])):
		print("filter channel" +str(ch))
		data_filt = np.convolve(h,data) # apply filter
		data_filt = data_filt[len(h) // 2:-len(h) // 2+1] # compensate for delay
		raw._data[ch] = data_filt
	return raw

def read_brainvision(fname, montage_path, apply_montage=True, add_ref=True, preload=False):
	"""Load brainvision data. If apply_montage=True, load and apply the standard
	montage for the 64-channel acticap. If add_ref=True add a reference
	channel with all zeros"""

	raw = read_raw_brainvision(fname, preload=preload)
	if add_ref:
		raw = add_reference_channels(raw, ref_channels="REF", copy=False)
	if apply_montage:
		mapping = {"1":"Fp1", "2":"Fp2", "3":"F7", "4":"F3", "5":"Fz", "6":"F4", "7":"F8","8":"FC5", "9":"FC1",
				   "10":"FC2", "11":"FC6", "12":"T7", "13":"C3", "14":"Cz", "15":"C4", "16":"T8", "17":"TP9", "18":"CP5", "19":"CP1",
				   "20":"CP2","21":"CP6","22":"TP10","23":"P7", "24":"P3", "25":"Pz", "26":"P4", "27":"P8", "28":"PO9", "29":"O1",
				   "30":"Oz", "31":"O2", "32":"PO10", "33":"AF7", "34":"AF3", "35":"AF4", "36":"AF8", "37":"F5", "38":"F1", "39":"F2",
				   "40":"F6", "41":"FT9","42":"FT7", "43":"FC3", "44":"FC4", "45":"FT8", "46":"FT10", "47":"C5", "48":"C1", "49":"C2",
				   "50":"C6", "51":"TP7", "52":"CP3", "53":"CPz", "54":"CP4", "55":"TP8", "56":"P5", "57":"P1", "58":"P2", "59":"P6",
				   "60":"PO7", "61":"PO3", "62":"POz", "63":"PO4", "64":"PO8"}
		raw.rename_channels(mapping)
		montage= read_montage(kind="AS-96_REF",path=montage_path)
		raw.set_montage(montage)
	return raw

def load_montage(path, plot=False): # load montage so we can do epochs_ica.set_montage(montage)
	file = glob.glob(path+"/*montage*")
	if not file:
		print("could not find electrode montage...")
		return None
	else:
		try:
			ch_pos = np.load(file[0])
		except FileNotFoundError:
			print("could not find electrode montage...")
			return None
		try:
			ch_names = json.load(open(Path(os.environ["EXPDIR"]+"cfg/eeg_channel_names.cfg")))
			ch_names = list(ch_names.values())
		except FileNotFoundError:
			print("could not find electrode names, continiuing without...")
			ch_names = []
			for i in range(1,len(ch_pos)+1):
				if i < 10:
					nr = "00"+str(i)
				if i < 100:
					nr = "0"+str(i)
				ch_names.append("EEG"+nr)
		dig_ch_pos = dict(zip(ch_names, ch_pos))
		montage = DigMontage(point_names=ch_names, dig_ch_pos=dig_ch_pos)
		if plot:
			montage.plot(show_names=True, kind="topomap")
		return montage

def robust_average_reference(epochs, ransac, apply=True):
	"""
	Create a robust average reference by first interpolating the bad channels to exclude outliers.
	The reference is applied as a projection. Return epochs with reference projection applied if apply=True
	"""
	epochs_tmp = epochs.copy()
	epochs_tmp = ransac.fit_transform(epochs)
	set_eeg_reference(epochs_tmp, ref_channels="average", projection=True)
	robust_avg_proj = epochs_tmp.info["projs"][0]
	del epochs_tmp
	epochs.info["projs"].append(robust_avg_proj)
	if apply:
		epochs.apply_proj()
	return epochs

def find_and_reject_ica_components(inst, reference_ica, n_components=0.99, method="fastica", corr_thresh=0.9, plot=False):

	if isinstance(reference_ica, str):
		reference_ica = read_ica(reference_ica)

	ica=ICA(n_components=n_components, method=method)
	ica.fit(inst)
	labels = list(reference_ica.labels_.keys())
	components = list(reference_ica.labels_.values())

	for component,label in zip(components, labels):
		corrmap([reference_ica,ica], template=(0,component[0]), plot=plot, label=label, threshold=corr_thresh)

	l = list(ica.labels_.values())
	exclude = [item for sublist in l for item in sublist] # flatten list of lists
	ica.apply(inst, exclude=exclude)

	return inst, ica
