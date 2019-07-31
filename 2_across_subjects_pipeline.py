# Author: Emily P. Stephen <emilyps14@gmail.com>

#%%
## Imports and Settings
from spatial_pac_utils import spatial_phase_amplitude_coupling as spac
from spatial_pac_utils.spatialPAC_plotting import plot_frequency_profiles,\
     plot_projection_surfaces, plot_lobe_projection_summary
from propofol_utils import parse_event_suffix
import numpy as np
import os.path as op
from os import makedirs


event_suffix = '_sedation'
subjects = ['eeganes02','eeganes03','eeganes04','eeganes05','eeganes07',
            'eeganes08','eeganes09','eeganes10','eeganes13','eeganes15']
subjects_dir = op.realpath('..')

swtPACmetric = 'corr'
blnPhaseHilbert = False
blnAmpHilbert = True
slow_band = (0.1,4)
bandwidth = 2.
amp_bands = [(freq-bandwidth/2,freq+bandwidth/2) for freq in range(4,50,2)]
swtBootstrapMethod = None

savepath = op.join(subjects_dir,'Manuscript_Figures')
if not op.exists(savepath):
    makedirs(savepath)

# get event remapping
eventnames, event_mapping, event_filename, load_suffix, event_savename = \
    parse_event_suffix(event_suffix)


#%%
## Compute frequency profiles for all subjects in sensor space


ch_names = [f'EEG{i:03d}' for i in range(1,65)]

# load all spac objects
pacs_sens = []
for subject in subjects:
    savename = spac.pac_savename(subject, event_filename, blnPhaseHilbert,
                                 blnAmpHilbert,slow_band,amp_bands,
                                 swtPACmetric, swtBootstrapMethod)

    pac_load_path = op.join(subjects_dir, 'input', subject)
    pac = spac.read_SensorSpacePAC(pac_load_path, savename)
    pacs_sens.append(pac)

# run computation
freq_profiles, freqs, eventnames = \
    spac.compute_sensor_space_profiles(pacs_sens, ch_names, eventnames, minobs=7,
                                       event_mapping=event_mapping)

#%%
## Make Figure 2

plot_frequency_profiles(freq_profiles,freqs,savepath=savepath)

#%%
## Perform projection analysis in source space

grade = 3

# load all source space pac objects
pacs_src = {}
for subject in subjects:
    savename = spac.pac_savename(subject, event_filename, blnPhaseHilbert,
                                 blnAmpHilbert,slow_band,amp_bands,
                                 swtPACmetric, swtBootstrapMethod)
    pac_load_path = op.join(subjects_dir, 'input', subject)
    pac = spac.read_SourceSpacePAC(pac_load_path,savename)
    pacs_src[subject] = pac

proj_all_subjects, vertices_to = \
    spac.compute_source_space_projections(pacs_src.values(), len(subjects),
                                          freq_profiles, eventnames, grade,
                                          event_mapping=event_mapping,
                                          subjects_dir=op.join(subjects_dir,'input'))

# Average the projections across subjects
proj_unstacked = np.nanmean(proj_all_subjects, axis=3) # (Nf,Nlevels,N)

#%%
## Make Figure 3

Nprojs = 1
agg_subject = 'fsaverage'
savename = spac.pac_savename(agg_subject, event_filename, blnPhaseHilbert,
                             blnAmpHilbert,slow_band,amp_bands,
                             swtPACmetric, swtBootstrapMethod)

plot_projection_surfaces(proj_unstacked, eventnames, vertices_to,
                             agg_subject,op.join(subjects_dir,'input'),
                             savename, Nprojs=1,
                             blnOffscreen=False,savepath=savepath)

#%%
## Make Figure 4

plot_lobe_projection_summary(pacs_src,freq_profiles,eventnames,stats_alpha=0.05,
                                 Nprojs=1,savepath=savepath)