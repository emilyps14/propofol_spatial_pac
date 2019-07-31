# Author: Emily P. Stephen <emilyps14@gmail.com>

#%%
## Imports and Settings
from spatial_pac_utils import mne_pipeline
from spatial_pac_utils import spatial_phase_amplitude_coupling as spac
from spatial_pac_utils.spatialPAC_plotting import plot_single_subject_summary
from spatial_pac_utils.utils import load_neighbors_mat
from propofol_utils import propofol_paths,get_LOC_and_ROC,load_bhvr,load_pr_resp
import os.path as op
from os import makedirs
import logging

logging.basicConfig(level=logging.INFO)

subject = 'eeganes07'
newFs = 200
cov_freqthresh = 65
blnDiagObsCov = True
amplitude_snr = 3.0
blnOverwrite = False
spacing_string = '-ico-3'
depth = None
event_filename = 'event_times_level.mat'
blnSaveInv = False

subjects_dir = op.realpath('..')

paths_dict = propofol_paths(subjects_dir,subject,spacing_string)

#%%
## Set up frequency bands
lf_band = (0.1,4)
topoplot_band = (8,16)

center_freqs = range(4,50,2)
bandwidth = 2.
bands = [(freq-bandwidth/2,freq+bandwidth/2) for freq in center_freqs]

bands.append(lf_band)
bands.append(topoplot_band)

#%%
## Run preprocessing pipeline
for blnSession in [False,True]:
    for band in bands:
        if blnSession:
            # Use event level observation covariance
            cov_folder = op.join(subjects_dir,
                                 subject, 'sourceloc_preproc')
            fpfx = mne_pipeline.get_epochs_filepfx(cov_folder,subject,
                                                   event_filename,
                                                   band[0],band[1],newFs)
            if blnDiagObsCov:
                obs_cov_filepath = fpfx + '-diag-cov.fif'
            else:
                obs_cov_filepath = fpfx + '-cov.fif'

            event_filename_in = None
        else:
            obs_cov_filepath = None
            event_filename_in = event_filename

        if band==lf_band or band==topoplot_band:
            l_trans_bandwidth = 'auto'
            h_trans_bandwidth = 0.5
        else:
            l_trans_bandwidth = 1.
            h_trans_bandwidth = 1.


        # run pipeline
        mne_pipeline.run_pipeline_epochs(
            subject,
            paths_dict,
            spacing_string=spacing_string,
            obs_cov_filepath=obs_cov_filepath,
            l_freq=band[0],
            h_freq=band[1],
            newFs=newFs,
            cov_freqthresh=cov_freqthresh,
            blnDiagObsCov=blnDiagObsCov,
            amplitude_snr=amplitude_snr,
            l_trans_bandwidth=l_trans_bandwidth,
            h_trans_bandwidth=h_trans_bandwidth,
            blnOverwrite=blnOverwrite,
            blnSaveInv=blnSaveInv,
            event_filename=event_filename_in,
            obsCov_suffix='',
            baseline=None,
            detrend=1,
            depth=depth,
            blnRunSourceLoc=not blnSession)

#%%
# Load neighbor dict
_, neighbor_dict = load_neighbors_mat(paths_dict['neighborfile'])

#%%
## Run sensor-space PAC pipeline by event

# for topoplots (8-16 Hz)
# saves figures in <subjects_dir>/<subject>/Figures sensor space/PAC_0.1-4_8-16/corr
spac.run_pipeline(subjects_dir,[subject],blnsave=True,
                  blnSourceSpace=False,blnOffscreen=False,
                  blnAcrossFrequencies=False,blnSession=False,
                  events_dir=paths_dict['events_dir'],
                  neighbor_dict=neighbor_dict)

# for non-central PCA (across frequencies)
# saves figures in <subjects_dir>/<subject>/PAC_across_frequencies_0.1-4_2.0Hz/corr
spac.run_pipeline(subjects_dir,[subject],blnsave=True,
                  blnSourceSpace=False,blnOffscreen=False,
                  blnAcrossFrequencies=True,blnSession=False,
                  events_dir=paths_dict['events_dir'],
                  neighbor_dict=neighbor_dict)

#%%
## Run sensor-space PAC pipeline over the session (for coherograms)
# saves figures in <subjects_dir>/<subject>/PAC_session_0.1-4_2.0Hz/corr
spac.run_pipeline(subjects_dir,[subject],blnsave=True,
                  blnSourceSpace=False,blnOffscreen=False,
                  blnAcrossFrequencies=True,blnSession=True,
                  events_dir=None,
                  neighbor_dict=neighbor_dict)


#%%
## Create Figure 1
savepath = op.join(subjects_dir,'Manuscript_Figures')
if not op.exists(savepath):
    makedirs(savepath)

propofol_dose, propofol_t, _, _, _ = load_bhvr(subject,subjects_dir)
key_times = get_LOC_and_ROC(subject,subjects_dir)
pr_resp_T,pr_resp_verbal,pr_resp_clicks = load_pr_resp(subject,subjects_dir)

plot_single_subject_summary(subjects_dir, subject,
                              propofol_t, propofol_dose, key_times,
                              pr_resp_T, pr_resp_verbal, pr_resp_clicks,
                              savepath)

#%%
## Run source-space PAC pipeline by event (for projection analysis)
spac.run_pipeline(subjects_dir,[subject],blnsave=True,
                  blnSourceSpace=True,blnOffscreen=False,
                  blnAcrossFrequencies=True,blnSession=False,
                  events_dir=paths_dict['events_dir'])


