# Author: Emily P. Stephen <emilyps14@gmail.com>

from spatial_pac_utils.mne_pipeline import get_epochs_filepfx, get_stc_filepfx, \
    get_times_for_epochs
from spatial_pac_utils import utils
import logging
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path as op
from os import makedirs
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mne.externals.h5io import read_hdf5, write_hdf5
import warnings

log = logging.getLogger(__name__)

class SpatialPAC(object):
    def __init__(self, subject, subjects_dir, newFs, phase_band, amp_bands,
                 blnPhaseHilbert,blnAmpHilbert, Nbootstrap, swtPACmetric,
                 swtBootstrapMethod, event_filename,
                 N, figures_output_path,
                 events=None, eventnames=None, Nlevels=None,
                 blnProcessed=False, prep=None,
                 phase_band_power=None,amp_band_power=None,
                 pac_all=None, bootstrap_samples=None):
        self.subject = subject
        self.subjects_dir = subjects_dir

        self.newFs = newFs
        self.phase_band = phase_band
        self.amp_bands = amp_bands
        self.Nfreqs = len(amp_bands)

        self.blnPhaseHilbert = blnPhaseHilbert
        self.blnAmpHilbert = blnAmpHilbert
        self.Nbootstrap = Nbootstrap
        self.swtPACmetric = swtPACmetric
        self.swtBootstrapMethod = swtBootstrapMethod

        self.event_filename = event_filename

        self.N = N
        self.figures_output_path = figures_output_path

        self.events = events
        self.eventnames = eventnames
        self.Nlevels = Nlevels

        # Set by run_computation
        self.blnProcessed = blnProcessed
        self.prep = prep
        self.phase_band_power = phase_band_power
        self.amp_band_power = amp_band_power
        self.pac_all = pac_all
        self.bootstrap_samples = bootstrap_samples

        if blnPhaseHilbert and swtPACmetric in ('corr','corr_noPhaseBand_Norm','height_ratio'):
            raise RuntimeError('Correlation and height ratio were not intended to work with Hilbert estimation of slow phase')

        if not blnPhaseHilbert and swtPACmetric=='mean_vector':
            raise RuntimeError('Mean Vector Length requires Hilbert estimation of slow phase')

    ##########################################################################
    ##### Functions to parse swtPACmetric
    def _prepare_PAC(self, amp_data, phase_band_data):
        blnCrossProduct =  self.swtBootstrapMethod == 'permtest_epochs_wi_level'

        if self.swtPACmetric in ('corr','corr_noPhaseBand_Norm'):
            data = prepare_corr(amp_data, phase_band_data,
                                blnCrossProduct=blnCrossProduct)
        elif self.swtPACmetric == 'height_ratio':
            data = prepare_height_ratio(amp_data, phase_band_data,
                                        blnCrossProduct=blnCrossProduct)
        elif self.swtPACmetric == 'mean_vector':
            data = prepare_mean_vector(amp_data,phase_band_data,
                                       blnCrossProduct=blnCrossProduct)
        else:
            raise RuntimeError('Unrecognized swtPACmetric')

        return data

    def _concatenate_data_levels(self, data_levels):
        if self.swtPACmetric in ('corr','corr_noPhaseBand_Norm'):
            data = [np.concatenate([d[0] for d in data_levels], axis=1),
                    np.concatenate([d[1] for d in data_levels], axis=1),
                    np.concatenate([d[2] for d in data_levels], axis=1)]
        elif self.swtPACmetric == 'height_ratio':
            data = [np.concatenate([d[0] for d in data_levels], axis=1),
                    np.concatenate([d[1] for d in data_levels], axis=1)]
        elif self.swtPACmetric == 'mean_vector':
            data = [np.concatenate([d[0] for d in data_levels], axis=1)]
        else:
            raise RuntimeError('Unrecognized swtPACmetric')

        return data

    def _compute_PAC(self, data, epoch_labels, epoch_labeling):
        if self.swtPACmetric in ('corr','corr_noPhaseBand_Norm'):
            if self.swtPACmetric == 'corr':
                swtNormCorr = 'AS'
            elif self.swtPACmetric == 'corr_noPhaseBand_Norm':
                swtNormCorr = 'A'
            pac_byLevel_bySensor = compute_corr(data, epoch_labels, epoch_labeling,
                                   swtNorm=swtNormCorr)
        elif self.swtPACmetric == 'height_ratio':
            pac_byLevel_bySensor = compute_height_ratio(data, epoch_labels, epoch_labeling)
        elif self.swtPACmetric == 'mean_vector':
            pac_byLevel_bySensor = compute_mean_vector(data, epoch_labels, epoch_labeling)
        else:
            raise RuntimeError('Unrecognized swtPACmetric')

        return pac_byLevel_bySensor

    def _compute_PAC_crossProduct(self, data, epoch_labeling=None):
        if self.swtPACmetric in ('corr','corr_noPhaseBand_Norm'):
            if self.swtPACmetric == 'corr':
                swtNormCorr = 'AS'
            elif self.swtPACmetric == 'corr_noPhaseBand_Norm':
                swtNormCorr = 'A'
            pac_bySensor = compute_corr_crossProduct(data, epoch_labeling,
                                                swtNorm=swtNormCorr)
        elif self.swtPACmetric == 'height_ratio':
            pac_bySensor = compute_height_ratio_crossProduct(data, epoch_labeling)
        elif self.swtPACmetric == 'mean_vector':
            pac_bySensor = compute_mean_vector_crossProduct(data, epoch_labeling)
        else:
            raise RuntimeError('Unrecognized swtPACmetric')

        return pac_bySensor


    ##########################################################################
    ##### Functions to prepare and run computation
    def _prepare_computation(self):
        ###### Run as much processing in advance as possible

        log.info('Preparing PAC computation...')

        phase_band_power = np.zeros((self.Nlevels, self.N))
        amp_band_power = np.zeros((self.Nfreqs, self.Nlevels, self.N))
        data_list = []
        epoch_labeling = []
        for j, level in enumerate(self.eventnames):
            log.info(level)

            phase_band_data_filtered, amp_band_data_filtered = self._get_data(level)

            n_epochs = phase_band_data_filtered.shape[1]

            if self.blnPhaseHilbert:
                phase_band_data_level = np.angle(hilbert(phase_band_data_filtered))
            else:
                phase_band_data_level = phase_band_data_filtered

            phase_band_power[j,:] = (np.abs(phase_band_data_filtered)**2).mean(axis=2).mean(axis=1)

            data_level = []
            for k,(amp_band,amp_data) in enumerate(zip(self.amp_bands,amp_band_data_filtered)):
                if self.blnAmpHilbert:
                    amplitude_level = np.abs(hilbert(amp_data))
                else:
                    amplitude_level = np.abs(amp_data)

                amp_band_power[k, j, :] = (amplitude_level**2).mean(axis=2).mean(axis=1)

                data_level.append(self._prepare_PAC(amplitude_level,phase_band_data_level))

            data_list.append(data_level) # outer list of len Nlevels, inner lists of len Nfreqs
            epoch_labeling.append([j] * n_epochs)

        return phase_band_power, amp_band_power, data_list, epoch_labeling

    def _run_PAC_and_bootstraps(self, data_list, epoch_labeling):
        N = data_list[0][0][0].shape[0]

        log.info('Computing PAC and Bootstraps...')
        ###### Calculate PAC and run Bootstraps
        pac_all = np.zeros((self.Nfreqs,self.Nlevels,N))
        if self.swtBootstrapMethod is None:
            bootstrap_samples = None
        else:
            bootstrap_samples = np.zeros((self.Nfreqs,self.Nlevels,N,self.Nbootstrap))

        for fi in range(self.Nfreqs):
            data_levels = [dl[fi] for dl in data_list] # the data for all levels (for this frequency)
            if self.swtBootstrapMethod is None:
                data = self._concatenate_data_levels(data_levels)
                epoch_labeling_cat = np.concatenate(epoch_labeling)
                pac_all[fi,:,:] = self._compute_PAC(data, list(range(self.Nlevels)), epoch_labeling_cat)
                bootstrap_samples = None
            else:
                if self.swtBootstrapMethod == 'permtest_levels':
                    data = self._concatenate_data_levels(data_levels)
                    epoch_labeling_cat = np.concatenate(epoch_labeling)
                    pac_all[fi,:,:] = self._compute_PAC(data, list(range(self.Nlevels)), epoch_labeling_cat)

                    for k in range(self.Nbootstrap):
                        perm = np.random.permutation(epoch_labeling)
                        bootstrap_samples[fi, :, :, k] = self._compute_PAC(data,
                                                                       list(range(self.Nlevels)),
                                                                       perm)

                elif self.swtBootstrapMethod == 'permtest_epochs_wi_level':
                    for j, data_level in enumerate(data_levels):
                        log.info(self.eventnames[j])

                        pac_all[fi,j,:] = self._compute_PAC_crossProduct(data_level)
                        n_epochs = data_level[0].shape[1]
                        for k in range(self.Nbootstrap):
                            perm = np.random.permutation(n_epochs)
                            bootstrap_samples[fi, j, :, k] = \
                                self._compute_PAC_crossProduct(data_level, perm)

        return pac_all, bootstrap_samples

    def _compute_data_list_spatial_average(self, indices_list):
        # Computes PAC across space using the indices_list
        # assumes that run_computation() has already been run
        # indices_list: list of list (*not* an iterator) will average within each sub-list
        data_list = self.prep['data_list'] # in original sources/sensors

        def _average_data_across_space(data):
            new_data = []
            for sub_data in data:
                new_data.append(np.vstack([np.nanmean(sub_data[indices,...],axis=0,keepdims=True) for indices in indices_list]))
            return new_data

        if self.swtPACmetric in ('corr', 'corr_noPhaseBand_Norm',
                                 'height_ratio', 'mean_vector'):
            # all of these metrics have sources in the axis=0 position within each element of data
            new_data_list = [[_average_data_across_space(data)
                              for data in row]
                             for row in data_list]
        else:
            raise RuntimeError('Unrecognized swtPACmetric')

        return new_data_list

    def run_computation(self, blnsave):
        phase_band_power, amp_band_power, data_list, epoch_labeling = self._prepare_computation()
        pac_all, bootstrap_samples = self._run_PAC_and_bootstraps(data_list,
                                                                  epoch_labeling)

        self.blnProcessed = True
        self.prep = {'data_list':data_list,'epoch_labeling':epoch_labeling}
        self.phase_band_power = phase_band_power
        self.amp_band_power = amp_band_power
        self.pac_all = pac_all
        self.bootstrap_samples = bootstrap_samples

        if blnsave:
            fname = self.get_savename()
            self.save(fname)

    ##########################################################################
    ##### Functions to plot, save, etc
    def plot_results(self, stats_alpha, swtMultipleComparisons, blnsave,
                     yl_power=None, yl_pac=None, amp_band_i=0):
        amp_band_power = self.amp_band_power[amp_band_i,:,:]
        pac_all = self.pac_all[amp_band_i,:,:]

        if blnsave and not op.exists(self.figures_output_path):
            makedirs(self.figures_output_path)

        ###### Compute p-values
        log.info('Computing p-values and significance...')
        reject = self.compute_rejection(stats_alpha, swtMultipleComparisons, amp_band_i=amp_band_i)

        ##### Prepare fname
        fname = self.get_savename()
        fname += '_SpatialPAC'
        if swtMultipleComparisons == 'maxstat':
            fname += '_maxstat'
        elif swtMultipleComparisons == 'FDR':
            fname += '_fdr'


        ###### Plot Results
        if yl_power is None:
            yl_power = [np.nanmin(10 * np.log10(amp_band_power)),
                        np.nanmax(10 * np.log10(amp_band_power))]
        if yl_pac is None:
            yl_pac = np.array([-1., 1.]) * np.nanmax(np.abs(pac_all))
        self._plot_and_save(reject, yl_power, yl_pac, fname, blnsave, amp_band_i)

    def compute_rejection(self, stats_alpha, swtMultipleComparisons, amp_band_i=0):
        pac_byLevel_bySensor = self.pac_all[amp_band_i,:,:]
        bootstrap_samples = self.bootstrap_samples[amp_band_i,:,:,:]
        Nbootstrap = self.Nbootstrap
        Nlevels = self.Nlevels
        swtBootstrapMethod = self.swtBootstrapMethod

        if swtBootstrapMethod is not None:
            reject = compute_rejection(pac_byLevel_bySensor,bootstrap_samples,
                                       swtBootstrapMethod,stats_alpha,
                                       swtMultipleComparisons)
        else:
            reject = None

        return reject

    def remap_events(self,event_mapping,event_filename='.xxx'):
        # event_mapping: list of 2-tuples (new_event,old_event)

        Nlevels = len(event_mapping)
        events = {m[0]:self.events[m[1]] for m in event_mapping}
        eventnames = [m[0] for m in event_mapping]
        level_inds = [np.where([x==m[1] for x in self.eventnames])[0][0] for m in event_mapping]

        if self.blnProcessed:
            phase_band_power = self.phase_band_power[level_inds, :]
            amp_band_power = self.amp_band_power[:, level_inds, :]
            data_list = [self.prep['data_list'][level_ind] for level_ind in level_inds]
            epoch_labeling = [[j]*data_level[0][0].shape[1] for j,data_level in enumerate(data_list)]
            pac_all = self.pac_all[:, level_inds, :]

            if self.swtBootstrapMethod is not None:
                bootstrap_samples = self.bootstrap_samples[:,level_inds,:,:]
            else:
                bootstrap_samples = None

            self.phase_band_power = phase_band_power
            self.amp_band_power = amp_band_power
            self.prep = {'data_list':data_list,'epoch_labeling':epoch_labeling}
            self.pac_all = pac_all
            self.bootstrap_samples = bootstrap_samples


        self.Nlevels = Nlevels
        self.events = events
        self.eventnames = eventnames
        self.event_filename = event_filename
        self.session_timeaxis = np.arange(len(eventnames))


    def compute_frequency_profiles(self,blnsave):
        pac_all = self.pac_all
        Nf = self.Nfreqs
        Nlevels = self.Nlevels
        N = self.N

        if pac_all is None:
            raise RuntimeError('run_computation() must be run before compute_frequency_profiles()')

        assert(np.all(np.equal((Nf,Nlevels,N),pac_all.shape)))

        self.frequency_profiles = compute_frequency_profiles(pac_all)

        if blnsave:
            fname = self.get_savename()
            self.save(fname)

    def change_freq_profile_signs(self,signs=None):
        self.frequency_profiles = \
            change_freq_profile_signs(self.frequency_profiles,
                                      self.get_center_freqs(),
                                      signs=signs)

    def project_pac_onto_frequency_profiles(self,U):
        # take frequency profiles from another spac (e.g. sensor space)
        # and project the pac from this spac (e.g. source space) onto them

        return project_pac_onto_frequency_profiles(self.pac_all,U)


    def plot_freq_profile_singular_values(self,blnsave):
        S = self.frequency_profiles['S']

        fig = plt.figure()
        plt.clf()
        plt.plot(S,'k.')
        plt.title('Frequency Profile Singular Values')

        if blnsave:
            fname = self.get_savename() + '_FrequencyProfile_svals'
            fig.savefig(op.join(self.figures_output_path,'{}.png'.format(fname)),
                        format='png')

    def plot_freq_profiles(self,blnsave,nprofiles=3):
        U = self.frequency_profiles['U']
        center_freqs = self.get_center_freqs()

        fig = plt.figure()
        plt.plot(center_freqs,U[:,:nprofiles])
        plt.title('First {} singular vectors (Frequency Profiles)'.format(nprofiles))
        plt.xlabel('Frequency (Hz)')
        plt.legend([str(i+1) for i in range(nprofiles)])

        if blnsave:
            fname = self.get_savename() + '_FrequencyProfiles'
            fig.savefig(op.join(self.figures_output_path,'{}.png'.format(fname)),
                        format='png')

    def plot_freq_profile_projections_byLevel(self,blnsave,bln3d=False):
        Nlevels = self.Nlevels
        V_unstacked = self.frequency_profiles['V_unstacked']
        eventnames = self.eventnames

        fig = plt.figure()
        level_colors = plt.cm.jet(np.linspace(0,1,Nlevels))
        h=[]
        if bln3d:
            ax = fig.add_subplot(111,projection='3d')
            for i,color in enumerate(level_colors):
                h.append(ax.scatter(V_unstacked[0, i, :], V_unstacked[1, i, :], V_unstacked[2, i, :],c=color))
            ax.set_zlabel('Third Component')
        else:
            ax = fig.add_subplot(111)
            for i,color in enumerate(level_colors):
                h.append(ax.scatter(V_unstacked[0, i, :], V_unstacked[1, i, :], 20, color))

        ax.set_xlabel('First Component')
        ax.set_ylabel('Second Component')
        fig.legend(h,eventnames)

        if blnsave:
            fname = self.get_savename() + '_FrequencyProfiles_levelProjections'
            fig.savefig(op.join(self.figures_output_path,'{}.png'.format(fname)),
                        format='png')

    def plot_freq_profile_projections_byElectrode(self,blnsave,bln3d=False):
        N = self.N
        V_unstacked = self.frequency_profiles['V_unstacked']

        fig = plt.figure()
        electrode_colors = plt.cm.jet(np.linspace(0,1,N))
        if bln3d:
            ax = fig.add_subplot(111,projection='3d')
            for i,color in enumerate(electrode_colors):
                ax.scatter(V_unstacked[0,:,i],V_unstacked[1,:,i],V_unstacked[3,:,i],c=color)
            ax.set_zlabel('Third Component')
        else:
            ax = fig.add_subplot(111)
            for i,color in enumerate(electrode_colors):
                ax.scatter(V_unstacked[0,:,i],V_unstacked[1,:,i],20,color)

        ax.set_xlabel('First Component')
        ax.set_ylabel('Second Component')

        if blnsave:
            fname = self.get_savename() + '_FrequencyProfiles_electrodeProjections'
            fig.savefig(op.join(self.figures_output_path,'{}.png'.format(fname)),
                        format='png')

    def get_savename(self):
        return pac_savename(self.subject, self.event_filename, self.blnPhaseHilbert, self.blnAmpHilbert,
                            self.phase_band, self.amp_bands,
                            self.swtPACmetric, self.swtBootstrapMethod)

    def get_center_freqs(self):
        return [np.mean(band) for band in self.amp_bands]

    def _get_data(self, level):
        return None, None  # Dummy

    def _plot_and_save(self, reject, yl_power, yl_diff, fname, blnsave,amp_band_i):
        pass  # Dummy -- implemented in child classes

    def save(self):
        pass  # Dummy

    def load(self):
        pass  # Dummy

    def _to_dict(self):
        return dict(
            subject=self.subject,
            subjects_dir=self.subjects_dir,
            newFs=self.newFs,
            phase_band=self.phase_band,
            amp_bands=self.amp_bands,
            blnPhaseHilbert=self.blnPhaseHilbert,
            blnAmpHilbert=self.blnAmpHilbert,
            Nbootstrap=self.Nbootstrap,
            swtPACmetric=self.swtPACmetric,
            swtBootstrapMethod=self.swtBootstrapMethod,
            event_filename=self.event_filename,
            events=self.events,
            eventnames=self.eventnames,
            Nlevels=self.Nlevels,
            Nfreqs=self.Nfreqs,
            N=self.N,
            figures_output_path=self.figures_output_path,
            blnProcessed=self.blnProcessed,
            prep=self.prep,
            phase_band_power=self.phase_band_power,
            amp_band_power=self.amp_band_power,
            pac_all=self.pac_all,
            bootstrap_samples=self.bootstrap_samples)


class SourceSpacePAC(SpatialPAC):
    def __init__(self, subject, subjects_dir, newFs, phase_band, amp_bands,
                 blnPhaseHilbert,blnAmpHilbert, Nbootstrap, swtPACmetric,
                 swtBootstrapMethod, event_filename,
                 mne_output_path, spacing_string,
                 obsCov_suffix, meeg_suffix, depth,
                 blnOffscreen, figures_output_path,
                 events=None, eventnames=None,
                 epoch_tmin=0.,epoch_tmax=30.,
                 Nlevels=None,
                 N=None, vertices=None,
                 blnProcessed=False, prep=None,
                 phase_band_power=None, amp_band_power=None,
                 pac_all=None, bootstrap_samples=None, get_data_fun=None,
                 events_dir=None):

        log.info(f'Initializing SourceSpacePAC for {subject}')

        if events is None or eventnames is None or Nlevels is None:
            if events_dir is None:
                eventfile = op.join(subjects_dir, subject, 'sourceloc_preproc',
                                    event_filename)
            else:
                eventfile = op.join(events_dir,event_filename)
            events, eventnames, _, _ = utils.load_eventfile(eventfile)
            Nlevels = len(eventnames)


        ####### Preload, set important variables
        if N is None or vertices is None:
            event_time = list(events.values())[0][0] # doesn't matter which one
            tmin = event_time + epoch_tmin
            tmax = event_time + epoch_tmax
            fpfx_phase = get_stc_filepfx(mne_output_path, subject,
                                        phase_band[0], phase_band[1], newFs,
                                        tmin, tmax, obsCov_suffix,
                                        meeg_suffix, spacing_string,
                                        depth=depth)

            stc_phase_band = mne.read_source_estimate(fpfx_phase)

            vertices = stc_phase_band.vertices
            N = len(vertices[0]) + len(vertices[1])  # number of sources

        super(SourceSpacePAC, self).__init__(subject, subjects_dir, newFs, phase_band, amp_bands,
                                             blnPhaseHilbert,blnAmpHilbert, Nbootstrap, swtPACmetric,
                                             swtBootstrapMethod, event_filename,
                                             N, figures_output_path,
                                             events=events, eventnames=eventnames, Nlevels=Nlevels,
                                             blnProcessed=blnProcessed, prep=prep,
                                             phase_band_power=phase_band_power, amp_band_power=amp_band_power,
                                             pac_all=pac_all, bootstrap_samples=bootstrap_samples)

        self.vertices = vertices
        self.mne_output_path = mne_output_path
        self.spacing_string = spacing_string
        self.obsCov_suffix = obsCov_suffix
        self.meeg_suffix = meeg_suffix
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
        self.depth = depth
        self.blnOffscreen = blnOffscreen
        self._get_data_fun = get_data_fun

    def _get_data(self, level):

        if self._get_data_fun is None:
            log.info('Loading stcs...')

            phase_band_data_filtered = self._load_stcs_for_level(level,self.phase_band)

            amp_band_data_filtered = [self._load_stcs_for_level(level,amp_band)
                                      for amp_band in self.amp_bands]
        else:
            log.info('Using custom data function...')

            phase_band_data_filtered,amp_band_data_filtered = self._get_data_fun(level)

        return phase_band_data_filtered, amp_band_data_filtered

    def _load_stcs_for_level(self, level, band):
        data = []
        for i, event_time in enumerate(self.events[level]):
            tmin = event_time + self.epoch_tmin
            tmax = event_time + self.epoch_tmax

            # Load stcs
            fpfx = get_stc_filepfx(self.mne_output_path, self.subject,
                                   band[0], band[1], self.newFs,
                                   tmin, tmax, self.obsCov_suffix,
                                   self.meeg_suffix,
                                   self.spacing_string,
                                   depth=self.depth)

            log.debug('Loading stc: {}'.format(fpfx))

            data.append(mne.read_source_estimate(fpfx).data)

        data = np.stack(data, axis=0).transpose(
            (1, 0, 2))  # N x n_epochs x Ntimes

        return data

    def _plot_and_save(self, reject, yl_power, yl_diff, fname, blnsave, amp_band_i,
                       blnMaskUnknown=True):

        log.info('Plotting Surfaces...')

        toplot = self.pac_all[amp_band_i,:,:].copy()
        toplot[np.isnan(toplot)] = 0

        def label_func(f):
            return self.eventnames[int(f)]

        if reject is None:
            ctrl_pts = [yl_diff[0], 0, yl_diff[1]]
            stc_corr = mne.SourceEstimate(toplot.T,
                                          vertices=self.vertices,
                                          tmin=0, tstep=1,
                                          subject=self.subject)
            brain = utils.plot_surf(stc_corr, {'kind': 'value',
                                                      'lims': ctrl_pts},
                                           'seismic', label_func,
                                           'semi-inflated', False,
                                           self.blnOffscreen,subjects_dir=self.subjects_dir,
                                           blnMaskUnknown=blnMaskUnknown)
        else:
            toplot[np.logical_not(reject)] = 0

            Ncm = 256
            cm = plt.get_cmap('seismic', Ncm)
            cm = cm(np.arange(cm.N)) * 255
            cm[int(np.floor(Ncm / 2 - 2)):int(np.ceil(Ncm / 2 + 2)), -1] = 0
            ctrl_pts = [yl_diff[0], 0, yl_diff[1]]

            stc_corr = mne.SourceEstimate(toplot.T,
                                          vertices=self.vertices,
                                          tmin=0, tstep=1,
                                          subject=self.subject)
            brain = utils.plot_surf(stc_corr, {'kind': 'value',
                                                      'lims': ctrl_pts},
                                           cm, label_func,
                                           'semi-inflated', False,
                                           self.blnOffscreen,subjects_dir=self.subjects_dir,
                                           blnMaskUnknown=blnMaskUnknown)

        if blnsave:
            for j,(t,level) in enumerate(zip(stc_corr.times,self.eventnames)):
                brain.set_time(t)
                utils.save_surf(brain, fname, self.figures_output_path,
                                   '_{}{}'.format(j, level))
            brain.close()

    def _to_dict(self):
        d = super(SourceSpacePAC, self)._to_dict()
        d['mne_output_path'] = self.mne_output_path
        d['spacing_string'] = self.spacing_string

        d['obsCov_suffix'] = self.obsCov_suffix
        d['meeg_suffix'] = self.meeg_suffix
        d['depth'] = self.depth

        d['epoch_tmin'] = self.epoch_tmin
        d['epoch_tmax'] = self.epoch_tmax
        d['vertices'] = self.vertices
        d['blnOffscreen'] = self.blnOffscreen

        return d

    def save(self,fname):
        log.info('Writing SourceSpacePAC to disk...')
        fname += '-SourceSpacePAC.h5'

        write_hdf5(op.join(self.figures_output_path,fname),
                   self._to_dict(),
                   title='SourceSpacePAC',
                   overwrite=True)

        log.info('[done.]')

    def plot_freq_profile_projections_surfaces(self,blnsave,nprofiles=2,
                                               freq_profiles=None,
                                               saveflag='_FrequencyProfiles_surfProjection{}',
                                               blnMaskUnknown=True):
        if freq_profiles is None:
            freq_profiles = self.frequency_profiles

        VT = freq_profiles['VT']
        V_unstacked = freq_profiles['V_unstacked']
        Nlevels = self.Nlevels
        eventnames = self.eventnames

        def label_func(f):
            return self.eventnames[int(f)]

        yl_diff = np.array([-1,1])*np.percentile(np.abs(VT[:2,:]), 100)
        for i in range(nprofiles):
            toplot = V_unstacked[i, :, :]
            toplot[np.isnan(toplot)] = 0

            ctrl_pts = [yl_diff[0], 0, yl_diff[1]]
            stc_proj = mne.SourceEstimate(toplot.T,
                                          vertices=self.vertices,
                                          tmin=0, tstep=1,
                                          subject=self.subject)
            brain = utils.plot_surf(stc_proj, {'kind': 'value',
                                                      'lims': ctrl_pts},
                                           'seismic', label_func,
                                           'semi-inflated', False,
                                           self.blnOffscreen,subjects_dir=self.subjects_dir,
                                           blnMaskUnknown=blnMaskUnknown)

            if blnsave:
                for j,(t,level) in enumerate(zip(stc_proj.times,self.eventnames)):
                    fname = self.get_savename() + saveflag.format(i)
                    brain.set_time(t)
                    utils.save_surf(brain, fname, self.figures_output_path,
                                       '_{}{}'.format(j, level))
                brain.close()

    def compute_PAC_in_ROIs(self,roi_labels):
        # Computes PAC within regions of interest
        # assumes that run_computation() has already been run
        epoch_labeling = self.prep['epoch_labeling']
        verts = self.vertices

        indices_list = [utils.find_label_mask(lbl,verts) for lbl in roi_labels]
        roi_data_list = self._compute_data_list_spatial_average(indices_list)
        roi_pac_all, roi_bootstrap_samples = self._run_PAC_and_bootstraps(roi_data_list,
                                                                          epoch_labeling)

        return roi_pac_all, roi_bootstrap_samples


class SensorSpacePAC(SpatialPAC):
    def __init__(self, subject, subjects_dir, newFs, phase_band, amp_bands,
                 blnPhaseHilbert,blnAmpHilbert, Nbootstrap,
                 swtPACmetric, swtBootstrapMethod, event_filename,
                 figures_output_path, blnSession, key_times=[],
                 events=None, eventnames=None, Nlevels=None,
                 N=None, ch_names=None, neighbor_dict=None, pos=None, session_timeaxis=None,
                 blnProcessed=False, prep=None,
                 phase_band_power=None, amp_band_power=None,
                 pac_all=None, bootstrap_samples=None, frequency_profiles=None,
                 events_dir=None):

        log.info(f'Initializing SensorSpacePAC for {subject}')

        if N is None or pos is None \
                or prep is None or ch_names is None or events is None \
                or eventnames is None or Nlevels is None:
            # Load epochs from filtered data
            amp_band_epochs,phase_band_epochs,events,eventnames,session_timeaxis,pos = \
                load_sensor_data(subject, subjects_dir, newFs, phase_band,
                                 amp_bands, event_filename, blnSession,
                                 events_dir=events_dir)
            Nlevels = len(eventnames)

            N = len(phase_band_epochs.ch_names)  # number of electrodes
            ch_names = phase_band_epochs.ch_names

        if neighbor_dict is None:
            # Load neighbor file for laplacian reference
            neighbor_mat_file = op.join(subjects_dir,'channelneighbors.mat')
            _, neighbor_dict = utils.load_neighbors_mat(neighbor_mat_file)

        if blnSession:
            event_filename = None

        super(SensorSpacePAC, self).__init__(subject, subjects_dir, newFs, phase_band, amp_bands,
                                             blnPhaseHilbert,blnAmpHilbert, Nbootstrap, swtPACmetric,
                                             swtBootstrapMethod, event_filename,
                                             N, figures_output_path,
                                             events=events, eventnames=eventnames, Nlevels=Nlevels,
                                             blnProcessed=blnProcessed, prep=prep,
                                             phase_band_power=phase_band_power, amp_band_power=amp_band_power,
                                             pac_all=pac_all, bootstrap_samples=bootstrap_samples)


        self.pos = pos
        self.neighbor_dict = neighbor_dict
        self.ch_names = ch_names
        self.session_timeaxis = session_timeaxis
        self.key_times = key_times
        self.blnSession = blnSession

        if prep is None:
            # Don't need these unless prepare_computations hasn't been run yet
            self.phase_band_epochs = phase_band_epochs
            self.amp_band_epochs = amp_band_epochs
        else:
            self.phase_band_epochs = None
            self.amp_band_epochs = None

        self.frequency_profiles = frequency_profiles


    def _get_data(self, level):
        # Laplacian reference
        phase_band_data_filtered = np.transpose(
            utils.get_laplacian_referenced_data_epochs(self.phase_band_epochs[level],
                                                       self.neighbor_dict),
            (1, 0, 2))  # N x n_epochs x Ntimes
        amp_band_data_filtered = [np.transpose(
            utils.get_laplacian_referenced_data_epochs(
                abe[level], self.neighbor_dict),
            (1, 0, 2)) for abe in self.amp_band_epochs]  # list of (N x n_epochs x Ntimes)

        return phase_band_data_filtered, amp_band_data_filtered


    def plot_sensor_names(self,blnsave,mask=None):
        pos = self.pos
        ch_names = [x[-2:] for x in self.ch_names]

        fig = plt.figure(1)
        plt.clf()
        im, cn = mne.viz.plot_topomap(np.zeros(len(pos)),pos,names=ch_names,mask=mask,
                                      show_names=True,head_pos={'center':[0.5,0.5],'scale':[1,1]})
        if blnsave:
            fig.savefig(op.join(self.figures_output_path, 'sensor_names.png'),
                        format='png')

    def remap_events(self,event_mapping,event_filename='.xxx'):
        super(SensorSpacePAC,self).remap_events(event_mapping,event_filename)

        if self.amp_band_epochs is not None:
            self.amp_band_epochs = [abe[self.eventnames] for abe in self.amp_band_epochs]

        if self.phase_band_epochs is not None:
            self.phase_band_epochs = self.phase_band_epochs[self.eventnames]

    def _plot_and_save(self, reject, yl_power, yl_diff, fname, blnsave,amp_band_i):
        Nlevels = self.Nlevels
        eventnames = self.eventnames
        swtBootstrapMethod = self.swtBootstrapMethod
        swtPACmetric = self.swtPACmetric
        pos = self.pos
        amp_band = self.amp_bands[amp_band_i]

        log.info('Plotting topoplots...')

        fig = plt.figure(2)
        plt.gcf().set_size_inches(19, 9.4)
        plt.clf()
        for j, level in enumerate(eventnames):
            # Power
            ax = plt.subplot2grid((2, Nlevels + 1), (0, j), rowspan=1,
                                  colspan=1)
            toplot = 10*np.log10(self.amp_band_power[amp_band_i,j,:])
            valid_inds = np.logical_not(np.isnan(toplot))
            im, cn = mne.viz.plot_topomap(toplot[valid_inds],
                                          pos[valid_inds,:], cmap='jet', vmin=yl_power[0],
                                          vmax=yl_power[1],head_pos={'center':[0.5,0.5],'scale':[1,1]})
            if j == Nlevels - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
            if j == 0:
                ax.set_ylabel('{}-{} Hz Power (dB)'.format(amp_band[0], amp_band[1]))
            ax.set_title(level)

            # PAC
            ax = plt.subplot2grid((2, Nlevels + 1), (1, j), rowspan=1,
                                  colspan=1)
            toplot = self.pac_all[amp_band_i,j,:]
            valid_inds = np.logical_not(np.isnan(toplot))
            if swtBootstrapMethod is not None:
                sigmask = reject[j,:]
                im, cn = mne.viz.plot_topomap(toplot[valid_inds], pos[valid_inds,:],
                                              cmap='seismic', vmin=yl_diff[0],
                                              vmax=yl_diff[1], contours=[0],
                                              mask=sigmask[valid_inds],head_pos={'center':[0.5,0.5],'scale':[1,1]})
            else:
                im, cn = mne.viz.plot_topomap(toplot[valid_inds], pos[valid_inds,:],
                                              cmap='seismic', vmin=yl_diff[0],
                                              vmax=yl_diff[1], contours=[0],head_pos={'center':[0.5,0.5],'scale':[1,1]})

            if j == Nlevels - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
            if j == 0:
                if swtPACmetric == 'corr':
                    ax.set_ylabel('Corr(high amplitude, low voltage)')
                elif swtPACmetric == 'corr_noPhaseBand_Norm':
                    ax.set_ylabel('AS\'/sqrt(AA\')')
                elif swtPACmetric == 'height_ratio':
                    ax.set_ylabel('(up-down)/max(up,down)')
                elif swtPACmetric == 'mean_vector':
                    ax.set_ylabel('real(mean vector)')

        if blnsave:
            fig.savefig(op.join(self.figures_output_path, fname + '.png'),
                        format='png')

    def plot_PAC_across_sensors(self, blnsave, amp_band_i=0):

        if self.blnSession:
            time_limits = self.session_timeaxis[[0,-1]]/60
            blnLabelLevels = False
        else:
            time_limits = [-0.5,self.Nlevels+0.5]
            blnLabelLevels = True

        toplot = self.pac_all[amp_band_i,:,:]

        fig = plt.figure(3,figsize=[10,2])
        fig.clf()
        plt.imshow(toplot.T, aspect='auto', interpolation='none',
                   origin='lower', cmap='seismic',
                   clim=np.array([-1,1])*np.max(np.abs(toplot)),
                   extent=[time_limits[0],time_limits[1],0,self.N])
        plt.xlabel('Time (min)')
        plt.ylabel('Electrode Index')
        plt.colorbar()

        if blnLabelLevels:
            plt.xticks(list(range(self.Nlevels)),self.eventnames,rotation='vertical')

        phase_band = self.phase_band
        amp_band = self.amp_bands[amp_band_i]
        plt.title('Correlation of {}-{} Hz Signal with {}-{} Hz Amplitude'.format(phase_band[0],phase_band[1],amp_band[0],amp_band[1]))

        for time in self.key_times:
            plt.axvline(time/60,color='k',linestyle='-',linewidth=2)

        plt.xlim([time_limits[0],time_limits[-1]])

        if blnsave:
            fname = self.get_savename() + '_AcrossSensors_{}-{}Hz'.format(amp_band[0],amp_band[1])
            fig.savefig(op.join(self.figures_output_path, fname + '.png'),
                        format='png')

    def plot_PAC_across_frequencies(self,blnsave,sensor_i):
        if self.blnSession:
            time_limits = self.session_timeaxis[[0,-1]]/60
            blnLabelLevels = False
        else:
            time_limits = [-0.5,self.Nlevels+0.5]
            blnLabelLevels = True

        toplot = self.pac_all[:,:,sensor_i]

        fig = plt.figure(4,figsize=[10,2])
        fig.clf()
        plt.imshow(toplot,aspect='auto',interpolation='none',
                   origin='lower',cmap='seismic',
                   clim=np.array([-1,1])*np.max(np.abs(toplot)),
                   extent=[time_limits[0],time_limits[1],
                           self.amp_bands[0][0],self.amp_bands[-1][1]])
        plt.xlabel('Time (min)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()

        if blnLabelLevels:
            plt.xticks(list(range(self.Nlevels)),self.eventnames,rotation='vertical')

        plt.title('Channel {}'.format(self.ch_names[sensor_i]))

        for time in self.key_times:
            plt.axvline(time/60,color='k',linestyle='-',linewidth=2)

        plt.xlim([time_limits[0],time_limits[-1]])

        if blnsave:
            fname = self.get_savename() + '_AcrossFrequencies_{}'.format(self.ch_names[sensor_i])
            fig.savefig(op.join(self.figures_output_path, fname + '.png'),
                        format='png')

    def plot_PAC_traces(self, blnsave,
                        ch_groups=[['EEG002','EEG006','EEG037'],['EEG028','EEG058','EEG057','EEG027','EEG022','EEG052']],
                        group_names=['Frontal','Posterior'],
                        save_suffix='-Front_Post_traces',
                        ax = None,
                        amp_band_i=0):
        # ch_groups is list of list of channel names (each group will be averaged for a trace)
        # group_names is the names for the groups (for the legend)
        if self.blnSession:
            timeaxis = self.session_timeaxis/60
            blnLabelLevels = False
        else:
            timeaxis = self.session_timeaxis
            blnLabelLevels = True

        T = len(timeaxis)
        toplot = np.zeros((T,len(ch_groups)))
        for i,esel in enumerate(ch_groups):
            electrode_inds = [j for j in range(self.N) if self.ch_names[j] in esel]
            toplot[:,i] = self.pac_all[np.ix_([amp_band_i], np.arange(T), electrode_inds)].mean(axis=2)[0,:]

        if ax is None:
            fig = plt.figure(3,figsize=[10,2])
            fig.clf()
            ax = plt.subplot(111)
            blnLabels = True

        ax.plot(timeaxis,toplot)
        ax.axhline(y=0,color='k')
        ax.legend(group_names,loc='upper left',fontsize=10)

        if blnLabels:
            ax.set_xlabel('Time (min)')
            if self.swtPACmetric == 'corr':
                ax.set_ylabel('Correlation')
            elif self.swtPACmetric == 'corr_noPhaseBand_Norm':
                ax.set_ylabel('AS\'/sqrt(AA\')')
            elif self.swtPACmetric == 'height_ratio':
                ax.set_ylabel('(up-down)/max(up,down)')
            elif self.swtPACmetric == 'mean_vector':
                ax.set_ylabel('real(Mean Vector)')

            phase_band = self.phase_band
            amp_band = self.amp_bands[amp_band_i]
            ax.set_title('Correlation of {}-{} Hz Signal with {}-{} Hz Amplitude'.format(phase_band[0],phase_band[1],amp_band[0],amp_band[1]))

        if blnLabelLevels:
            ax.set_xticks(timeaxis)
            ax.set_xticklabels(self.eventnames,rotation='vertical')

        for time in self.key_times:
            plt.axvline(time/60,color='k',linestyle='-',linewidth=2)

        if blnsave:
            fname = self.get_savename() + save_suffix
            fig.savefig(op.join(self.figures_output_path, fname + '.png'),
                        format='png')

    def plot_level_phase_summaries(self, fname, blnsave, amp_band_i=0):

        if self.swtPACmetric not in ('mean_vector'):
            raise RuntimeError('Phase summaries only work with complex PAC metrics (e.g. mean vector)')

        Nlevels = self.Nlevels
        eventnames = self.eventnames
        pos = self.pos
        ch_names = self.ch_names

        if self.swtBootstrapMethod == 'permtest_epochs_wi_level':
            # cross product prepared data: take diagonal of second two axes
            data_list = [d[amp_band_i][0][:,list(range(d[amp_band_i][0].shape[1])),list(range(d[amp_band_i][0].shape[1]))] for d in self.prep['data_list']]
        else:
            data_list = [d[amp_band_i][0] for d in self.prep['data_list']]

        assert(all([np.allclose(np.mean(d,1).real,p,atol=0)
                    for d,p in zip(data_list,self.pac_all[amp_band_i,:,:])]))

        log.info('Plotting polar plots...')
        layout = mne.channels.Layout(box=(0, 1, 0, 1), pos=pos, names=ch_names, kind='EEG', ids=1 + np.arange(self.N))
        info = mne.create_info(ch_names,self.newFs)

        for j, level in enumerate(eventnames):

            toplot = data_list[j]

            fig = plt.figure(j)
            fig.set_size_inches(17.5, 9.4)
            fig.clf()
            lim = np.max(np.abs(toplot))
            def pac_callback(ax,ch_idx,blnLabels=True):
                p = ax.get_position()
                ax.remove()
                ax = plt.axes(p,projection='polar')
                ax.grid(False)
                ch_data = toplot[ch_idx,:]
                ch_mean = np.mean(ch_data)
                if ch_mean.real<0:
                    color='b'
                else:
                    color='r'
                ax.scatter(np.angle(ch_data),np.abs(ch_data),marker='.',c='k')
                ax.plot([0,np.angle(ch_mean)],[0,np.abs(ch_mean)],color=color,linewidth=2)
                ax.viewLim.y1 = lim
                if blnLabels:
                    # ax.set_title(ch_names[ch_idx])
                    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2])
                    ax.set_rticks([lim])
                else:
                    ax.set_xticks([])
                    ax.set_rticks([])
            for ax,idx in mne.viz.iter_topography(info,layout=layout,fig_facecolor='white',axis_facecolor='white',axis_spinecolor='white',on_pick=pac_callback,fig=fig):#,layout_scale=1./2):
                pac_callback(ax,idx,blnLabels=idx==self.N-1)

            if blnsave:
                fig.savefig(op.join(self.figures_output_path,'{}_{}.png'.format(fname,level)),
                            format='png')


    def plot_total_phase_summary(self, fname, blnsave, amp_band_i=0):

        if self.swtPACmetric not in ('mean_vector'):
            raise RuntimeError('Phase summaries only work with complex PAC metrics (e.g. mean vector)')

        if self.swtBootstrapMethod == 'permtest_epochs_wi_level':
            # cross product prepared data: take diagonal of second two axes
            data_list = [d[amp_band_i][0][:,list(range(d[amp_band_i][0].shape[1])),list(range(d[amp_band_i][0].shape[1]))] for d in self.prep['data_list']]
        else:
            data_list = [d[amp_band_i][0] for d in self.prep['data_list']]

        assert(all([np.allclose(np.mean(d,1).real,p,atol=0)
                    for d,p in zip(data_list,self.pac_all[amp_band_i,:,:])]))

        log.info('Plotting summary polar plot...')
        data_means = np.hstack([np.mean(d,1) for d in data_list])

        fig = plt.figure(facecolor='white')
        fig.clf()
        fig.set_size_inches(10,5)
        ax = fig.add_subplot(121,projection='polar')
        for mean_vector in np.nditer(data_means):
            ax.plot([0,np.angle(mean_vector)],[0,np.abs(mean_vector)],color='k')
        ax.set_title('Mean Vectors for all Epochs')

        nbins = 16
        bins = np.linspace(-np.pi,np.pi,nbins+1,endpoint=True)
        bins -= (bins[1]-bins[0])/2
        dist = np.zeros((nbins,3))
        phases = np.angle(data_means)
        lengths = np.abs(data_means)
        for i in range(nbins):
            inds_in_phase_bin = np.logical_and(np.greater(phases,bins[i]),
                                               np.less_equal(phases,bins[i+1]))
            dist[i,:] = np.percentile(lengths[inds_in_phase_bin],[2.5,50,97.5])

        ax = fig.add_subplot(122,projection='polar')
        paxis = (bins[:-1]+bins[1:])/2
        paxis = np.append(paxis,paxis[0])
        dist = np.append(dist,dist[np.newaxis,0,:],axis=0)
        ax.plot(paxis,dist[:,0],color='gray',linestyle=':')
        ax.plot(paxis,dist[:,1],color='k',linestyle='-')
        ax.plot(paxis,dist[:,2],color='gray',linestyle='--')
        ax.legend(['2.5%','50%','97.5%'],loc='upper left')
        ax.text(0,ax.get_rmax()/2,'Peakmax',ha='center',va='center',color='r')
        ax.text(np.pi,ax.get_rmax()/2,'Troughmax',ha='center',va='center',color='b')
        ax.set_title('Distribution of Mean Vectors')

        if blnsave:
            fig.savefig(op.join(self.figures_output_path,'{}.png'.format(fname)),
                        format='png')

    def plot_freq_profile_projections_topos(self,blnsave,nprofiles=2):
        S = self.frequency_profiles['S']
        VT = self.frequency_profiles['VT']
        V_unstacked = self.frequency_profiles['V_unstacked']
        Nlevels = self.Nlevels
        eventnames = self.eventnames
        pos = self.pos

        fig = plt.figure()
        g_kwargs = {'left': 0.05, 'right': 0.98, 'bottom': 0.02, 'top': 0.98, 'hspace':0.1, 'wspace': 0.01}
        gs = gridspec.GridSpec(nprofiles+1, Nlevels, height_ratios=np.concatenate((3*np.ones(nprofiles),[0.75])), **g_kwargs)
        yl_diff = np.array([-1,1])*np.percentile(np.abs(VT[:2,:]), 100)

        for i in range(nprofiles):
            axes = [fig.add_subplot(gs[i, j]) for j in range(Nlevels)]
            im = utils.plot_topos(axes, V_unstacked[i, :, :], yl_diff, pos)
            axes[0].set_ylabel('Component {}'.format(i+1))
            if i==0:
                for ax,name in zip(axes,eventnames):
                    ax.set_title(name)

        cax = fig.add_subplot(gs[nprofiles, 1:Nlevels-1])
        utils.add_colorbar(fig,cax,im)

        if blnsave:
            fname = self.get_savename() + '_FrequencyProfiles_{}topoProjections'.format(nprofiles)
            fig.savefig(op.join(self.figures_output_path,'{}.png'.format(fname)),
                        format='png')

    def compute_PAC_across_electrode_sets(self,sel_list):
        # Computes PAC within sets of electrodes
        # assumes that run_computation() has already been run
        # sel_list: list of list of electrode names (will average within each sub-list)
        epoch_labeling = self.prep['epoch_labeling']
        N = self.N
        ch_names = self.ch_names

        indices_list = [[i for i in range(N) if ch_names[i] in sel] for sel in sel_list]
        for inds,sel in zip(indices_list,sel_list):
            if len(inds)!=len(sel):
                raise RuntimeError(f'Missing electrode name. Selection ({sel}) Indices ({inds})')
        roi_data_list = self._compute_data_list_spatial_average(indices_list)
        roi_pac_all, roi_bootstrap_samples = self._run_PAC_and_bootstraps(roi_data_list,
                                                                          epoch_labeling)

        return roi_pac_all, roi_bootstrap_samples

    def _to_dict(self):
        d = super(SensorSpacePAC, self)._to_dict()
        d['neighbor_dict'] = self.neighbor_dict
        d['pos'] = self.pos
        d['ch_names'] = self.ch_names
        d['session_timeaxis'] = self.session_timeaxis
        d['key_times'] = self.key_times
        d['blnSession'] = self.blnSession
        d['frequency_profiles'] = self.frequency_profiles

        return d

    def save(self,fname):
        log.info('Writing SensorSpacePAC to disk...')
        fname += '-SensorSpacePAC.h5'

        write_hdf5(op.join(self.figures_output_path,fname),
                   self._to_dict(),
                   title='SensorSpacePAC',
                   overwrite=True)

        log.info('[done.]')

def pac_savename(subject,event_filename,blnPhaseHilbert,blnAmpHilbert,phase_band,amp_bands,swtPACmetric,swtBootstrapMethod):
    if event_filename is None:
        fname = '{}_session_pac'.format(subject)
    else:
        fname = '{}_{}_pac'.format(subject, event_filename[:-4])

    fname += '_phase{}-{}'.format(phase_band[0],phase_band[1])
    if blnPhaseHilbert:
        fname += 'Hilbert'

    if len(amp_bands)==1:
        fname += '_amp{}-{}'.format(amp_bands[0][0],amp_bands[0][1])
    else:
        fname += '_amp'
    if blnAmpHilbert:
        fname += 'Hilbert'

    fname += '_{}'.format(swtPACmetric)

    if swtBootstrapMethod == 'permtest_levels':
        fname += '_permtest_levels'
    elif swtBootstrapMethod == 'permtest_epochs_wi_level':
        fname += '_permtest_epochs'

    return fname

def compute_rejection(pac_byLevel_bySensor, bootstrap_samples, swtBootstrapMethod,
                      stats_alpha, swtMultipleComparisons):
    # pac_byLevel_bySensor: (Nlevels,N)
    # bootstrap_samples: (Nlevels,N,Nbootstrap)
    (Nlevels,N,Nbootstrap) = bootstrap_samples.shape

    if swtMultipleComparisons == 'FDR':
        bootstrap_p = np.nansum(
            np.greater(pac_byLevel_bySensor[:, :, np.newaxis],
                       bootstrap_samples), axis=2,
            dtype=np.float) / Nbootstrap
        bootstrap_p[bootstrap_p == 0] = 1. / Nbootstrap
        bootstrap_p[bootstrap_p == 1] = 1 - 1. / Nbootstrap

        reject_mc_low, pval_mc_low = mne.stats.fdr_correction(
            bootstrap_p, alpha=stats_alpha / 2, method='indep')
        reject_mc_high, pval_mc_high = mne.stats.fdr_correction(
            1 - bootstrap_p, alpha=stats_alpha / 2, method='indep')
        reject = np.logical_or(reject_mc_low, reject_mc_high)
    elif swtMultipleComparisons == 'maxstat':
        if swtBootstrapMethod == 'permtest_levels':
            max_stat_dist = np.abs(bootstrap_samples).nanmax(axis=1).nanmax(
                axis=0)
            critical_threshold = np.percentile(max_stat_dist,
                                               (1 - stats_alpha) * 100,
                                               interpolation='lower') * np.ones(
                Nlevels)
        elif swtBootstrapMethod == 'permtest_epochs_wi_level':
            critical_threshold = np.zeros(Nlevels)
            for j in range(Nlevels):
                max_stat_dist_level = np.abs(
                    bootstrap_samples[j, :, :]).nanmax(axis=0)
                critical_threshold[j] = np.nanpercentile(
                    max_stat_dist_level, (1 - stats_alpha) * 100,
                    interpolation='lower')
        reject = np.greater_equal(np.abs(pac_byLevel_bySensor),
                                  critical_threshold[:, np.newaxis])

    return reject


def compute_frequency_profiles(pac_all):
    # Assumes frequency is in the first dimension

    original_shape = pac_all.shape
    Nf = original_shape[0]

    pac_stacked = np.reshape(pac_all,(Nf,np.prod(original_shape[1:])))

    # take out all channels/levels with NaNs (make sure we don't eliminate any real data)
    # NaNs are expected in the following cases:
        # Sensor space subject average: channels/levels will too few observations
        # Source space computed with no unknown or corpus callosum (MNE estimate is 0 in those locations, yielding NaN for the PAC)
    nan_columns = np.any(np.isnan(pac_stacked),axis=0)
    assert(np.all(np.isnan(pac_stacked[:,nan_columns])))

    # Compute SVD
    U,S,VT = np.linalg.svd(pac_stacked[:,np.logical_not(nan_columns)],full_matrices=False)

    # put NaNs back in, so that the channels line up with ch_names
    VT_full = np.zeros((Nf,len(nan_columns)))
    VT_full[:,nan_columns] = np.nan
    VT_full[:,np.logical_not(nan_columns)] = VT

    V_unstacked = np.reshape(VT_full, original_shape)

    return dict(U=U,S=S,VT=VT,V_unstacked=V_unstacked)


def change_freq_profile_signs(frequency_profiles,faxis,signs=None):
        U = frequency_profiles['U']
        VT = frequency_profiles['VT']
        V_unstacked = frequency_profiles['V_unstacked']

        if signs is None:
            alpha_beta_mask = np.logical_and(np.greater_equal(faxis,8),np.less_equal(faxis,20))

            signs = np.ones(3)
            signs[0] = np.sign(np.mean(U[:,0]))
            signs[1] = np.sign(np.mean(U[alpha_beta_mask,1]))
            signs[2] = np.sign(np.mean(U[alpha_beta_mask,2]))

        for i,s in enumerate(signs):
            U[:,i] *= s
            VT[i,:] *= s
            V_unstacked[i,:,:] *= s

        frequency_profiles['U'] = U
        frequency_profiles['VT'] = VT
        frequency_profiles['V_unstacked'] = V_unstacked

        return frequency_profiles


def project_pac_onto_frequency_profiles(pac_all,U):
    # pac_all: Nf x Nlevels x N
    Nf,Nlevels,N = pac_all.shape
    assert(all(np.equal(U.shape,(Nf,Nf))))

    pac_stacked = np.reshape(pac_all,(Nf,N*Nlevels))

    SVT_proj = U.T.dot(pac_stacked)
    S_proj = np.nansum(SVT_proj**2,axis=1)**(1/2)
    VT_full_proj = SVT_proj / S_proj[:, np.newaxis]

    V_proj_unstacked = np.reshape(VT_full_proj, (Nf, Nlevels, N))

    nan_columns = np.any(np.isnan(VT_full_proj),axis=0)
    VT_proj = VT_full_proj[:,np.logical_not(nan_columns)]

    freq_profiles_proj = dict(U=U,S=S_proj,VT=VT_proj,V_unstacked=V_proj_unstacked)
    return freq_profiles_proj


####### computation functions
def prepare_corr(amp_data, phase_band_data, blnCrossProduct=False):
    # input: alpha amplitude and slow data, each (N x n_epochs x Ntimes)
    # output: dot products for each channel and epoch

    amp_data -= np.mean(amp_data, axis=2)[:, :, np.newaxis]

    if blnCrossProduct:
        # N x n_epochs x n_epochs
        (N, n_epochs, _) = amp_data.shape
        amp_phase = np.zeros((N, n_epochs, n_epochs))
        for i in range(N):
            amp_phase[i, :, :] = np.dot(amp_data[i, :, :],
                                         phase_band_data[i, :, :].T)
    else:
        # N x n_epochs
        amp_phase = np.sum(np.multiply(amp_data, phase_band_data), axis=2)

    amp_amp = np.sum(np.multiply(amp_data, amp_data), axis=2)
    phase_phase = np.sum(np.multiply(phase_band_data, phase_band_data), axis=2)

    # can take sum across epochs, as long as they are the same length (time)
    # corrcoef = amp_phase(.)/np.sqrt(amp_amp(.)*phase_phase(.))

    return amp_phase, amp_amp, phase_phase


def compute_corr(data, epoch_labels=[1], epoch_labeling=None, swtNorm='AS'):
    amp_phase, amp_amp, phase_phase = data

    N, n_epochs = amp_phase.shape

    if epoch_labeling is None:
        epoch_labeling = np.ones(n_epochs)

    corr = np.zeros((len(epoch_labels), N))
    for i, lab in enumerate(epoch_labels):
        inds = epoch_labeling == lab
        a_s = np.sum(amp_phase[:, inds], axis=1)
        a_a = np.sum(amp_amp[:, inds], axis=1)
        s_s = np.sum(phase_phase[:, inds], axis=1)

        if swtNorm == 'A':  # test
            corr[i, :] = np.divide(a_s, np.sqrt(a_a))
        else:
            corr[i, :] = np.divide(a_s, np.sqrt(np.multiply(a_a, s_s)))

    return corr


def compute_corr_crossProduct(data, epoch_labeling=None, swtNorm='AS'):
    amp_phase, amp_amp, phase_phase = data
    [N, n_epochs, _] = amp_phase.shape
    if epoch_labeling is None:
        epoch_labeling = list(range(n_epochs))

    a_s = np.zeros(N)
    a_a = np.zeros(N)
    s_s = np.zeros(N)
    for i, epoch in enumerate(epoch_labeling):
        a_s += amp_phase[:, i, epoch]
        a_a += amp_amp[:, i]
        s_s += phase_phase[:, epoch]

    if swtNorm == 'A':  # test
        corr = np.divide(a_s, np.sqrt(a_a))
    else:
        corr = np.divide(a_s, np.sqrt(np.multiply(a_a, s_s)))

    return corr


def prepare_height_ratio(amp_data, phase_band_data, blnCrossProduct=False):
    # input: alpha amplitude and slow data, each (N x n_epochs x Ntimes)
    # output: alpha up and down for each channel and epoch

    (N, n_epochs, _) = amp_data.shape

    if blnCrossProduct:
        # N x n_epochs x n_epochs
        amp_up = np.zeros((N, n_epochs, n_epochs))
        amp_down = np.zeros((N, n_epochs, n_epochs))
        for i in range(N):
            for k in range(n_epochs):
                up_inds = phase_band_data[i, k, :] > 0
                down_inds = phase_band_data[i, k, :] < 0
                if not(any(up_inds)) or not(any(down_inds)):
                    amp_mean = np.mean(amp_data[i, :, :], axis=1)
                    amp_up[i, :, k] = amp_mean
                    amp_down[i, :, k] = amp_mean
                else:
                    for j in range(n_epochs):
                        amp_up[i, j, k] = np.mean(amp_data[i, j, up_inds])
                        amp_down[i, j, k] = np.mean(amp_data[i, j, down_inds])
    else:
        # N x n_epochs
        amp_up = np.zeros((N, n_epochs))
        amp_down = np.zeros((N, n_epochs))
        for i in range(N):
            for j in range(n_epochs):
                up_inds = phase_band_data[i, j, :] > 0
                down_inds = phase_band_data[i, j, :] < 0
                if not(any(up_inds)) or not(any(down_inds)):
                    amp_mean = np.mean(amp_data[i, j, :])
                    amp_up[i, j] = amp_mean
                    amp_down[i, j] = amp_mean
                else:
                    amp_up[i, j] = np.mean(amp_data[i, j, up_inds])
                    amp_down[i, j] = np.mean(amp_data[i, j, down_inds])

    return amp_up, amp_down


def compute_height_ratio(data, epoch_labels=[1], epoch_labeling=None):
    amp_up, amp_down = data

    N, n_epochs = amp_up.shape

    if epoch_labeling is None:
        epoch_labeling = np.ones(n_epochs)

    height_ratio = np.zeros((len(epoch_labels), N))
    for i, lab in enumerate(epoch_labels):
        inds = epoch_labeling == lab
        a_u = np.mean(amp_up[:, inds], axis=1)
        a_d = np.mean(amp_down[:, inds], axis=1)

        height_ratio[i,:] = np.divide(a_u-a_d, np.maximum(a_u,a_d))

    return height_ratio

def compute_height_ratio_crossProduct(data, epoch_labeling=None):
    amp_up, amp_down = data
    [N, n_epochs, _] = amp_up.shape

    if epoch_labeling is None:
        epoch_labeling = list(range(n_epochs))

    a_u = np.zeros(N)
    a_d = np.zeros(N)
    for i, epoch in enumerate(epoch_labeling):
        a_u += amp_up[:, i, epoch]
        a_d += amp_down[:, i, epoch]

    a_u /= len(epoch_labeling)
    a_d /= len(epoch_labeling)
    height_ratio = np.divide(a_u-a_d, np.maximum(a_u,a_d))

    return height_ratio

def prepare_mean_vector(amp_data, phase_band_data, blnCrossProduct=False):
    # input: alpha amplitude and slow data, each (N x n_epochs x Ntimes)
    # output: mean vector (complex) for each channel and epoch

    if blnCrossProduct:
        # N x n_epochs x n_epochs
        (N, n_epochs, Ntimes) = amp_data.shape
        amp_phase = np.zeros((N, n_epochs, n_epochs),dtype=np.complex64)
        for i in range(N):
            amp_phase[i, :, :] = 1./Ntimes*np.dot(amp_data[i, :, :],
                                        np.exp(1j*phase_band_data[i, :, :].T))
    else:
        # N x n_epochs
        amp_phase = np.mean(np.multiply(amp_data,np.exp(1j*phase_band_data)), axis=2)

    # can take mean across epochs, as long as they are the same length (time)
    # mvl = np.abs(amp_phase(.))

    return [amp_phase]

def compute_mean_vector(data, epoch_labels=[1], epoch_labeling=None, swtTransform='real'):
    amp_phase = data[0]

    N, n_epochs = amp_phase.shape

    if epoch_labeling is None:
        epoch_labeling = np.ones(n_epochs)

    mean_vector = np.zeros((len(epoch_labels), N), dtype=np.complex64)
    for i, lab in enumerate(epoch_labels):
        inds = epoch_labeling == lab
        mean_vector[i,:] = np.mean(amp_phase[:, inds], axis=1)

    if swtTransform=='real':
        mean_vector = mean_vector.real
    elif swtTransform=='length':
        mean_vector = np.abs(mean_vector)

    return mean_vector

def compute_mean_vector_crossProduct(data, epoch_labeling=None, swtTransform='real'):
    amp_phase = data[0]
    [N, n_epochs, _] = amp_phase.shape

    if epoch_labeling is None:
        epoch_labeling = list(range(n_epochs))

    mean_vector = np.zeros(N,dtype=np.complex64)
    for i, epoch in enumerate(epoch_labeling):
        mean_vector += amp_phase[:, i, epoch]
    mean_vector /= len(epoch_labeling)

    if swtTransform=='real':
        mean_vector = mean_vector.real
    elif swtTransform=='length':
        mean_vector = np.abs(mean_vector)

    return mean_vector


###### Reading objects
def read_SourceSpacePAC(outputpath,fname):
    log.info('Reading SourceSpacePAC from disk...')

    if fname[-18:]!='-SourceSpacePAC.h5':
        fname += '-SourceSpacePAC.h5'

    d = read_hdf5(op.join(outputpath,fname), title='SourceSpacePAC')

    log.info('Constructing the object...')
    pac = SourceSpacePAC(d['subject'], d['subjects_dir'], d['newFs'], d['phase_band'], d['amp_bands'],
                         d['blnPhaseHilbert'], d['blnAmpHilbert'], d['Nbootstrap'], d['swtPACmetric'],
                         d['swtBootstrapMethod'], d['event_filename'],
                         d['mne_output_path'], d['spacing_string'],
                         d['obsCov_suffix'], d['meeg_suffix'],
                         d.get('depth',None),
                         d['blnOffscreen'], d['figures_output_path'],
                         events=d['events'], eventnames=d['eventnames'],
                         epoch_tmin=d.get('epoch_tmin',0.),epoch_tmax=d.get('epoch_tmax',30.),
                         Nlevels=d['Nlevels'],
                         N=d['N'], vertices=d['vertices'],
                         blnProcessed=d['blnProcessed'], prep=d['prep'],
                         phase_band_power=d['phase_band_power'],amp_band_power=d['amp_band_power'],
                         pac_all=d['pac_all'], bootstrap_samples=d['bootstrap_samples'])

    log.info('[done.]')
    return pac

def read_SensorSpacePAC(outputpath,fname):
    log.info('Reading SensorSpacePAC from disk...')

    if fname[-18:]!='-SensorSpacePAC.h5':
        fname += '-SensorSpacePAC.h5'

    d = read_hdf5(op.join(outputpath,fname), title='SensorSpacePAC')

    log.info('Constructing the object...')
    pac = SensorSpacePAC(d['subject'], d['subjects_dir'], d['newFs'], d['phase_band'], d['amp_bands'],
                         d['blnPhaseHilbert'], d['blnAmpHilbert'], d['Nbootstrap'], d['swtPACmetric'],
                         d['swtBootstrapMethod'], d['event_filename'],
                         d['figures_output_path'], d['blnSession'], key_times=d.get('key_times',[]),
                         events=d['events'], eventnames=d['eventnames'], Nlevels=d['Nlevels'],
                         N=d['N'], ch_names=d['ch_names'], neighbor_dict=d['neighbor_dict'],
                         pos=d['pos'], session_timeaxis=d.get('session_timeaxis',None), blnProcessed=d['blnProcessed'], prep=d['prep'],
                         phase_band_power=d['phase_band_power'],amp_band_power=d['amp_band_power'],
                         pac_all=d['pac_all'], bootstrap_samples=d['bootstrap_samples'],
                         frequency_profiles=d.get('frequency_profiles',None))

    log.info('[done.]')
    return pac


###### Frequency profiles and projections
def compute_sensor_space_profiles(pacs, ch_names, eventnames, minobs=7,
                                  event_mapping=None):
    """

    :type pacs: list of SensorSpacePAC
    """

    # collect pac_all and bootstraps for all subjects, align axes
    Nsubjects = len(pacs)
    Nlevels = len(eventnames)
    N = len(ch_names)
    Nfreqs = pacs[0].Nfreqs

    assert(Nfreqs>1)

    pac_all_subjects = np.empty((Nfreqs,Nlevels,N,Nsubjects))
    for i,pac in enumerate(pacs):
        if event_mapping is not None:
            log.info(f'Remapping Events for {pac.subject}...')
            pac.remap_events(event_mapping[pac.subject])

        log.info(f'Collecting data from {pac.subject}...')
        for j,level in enumerate(eventnames):
            if level in pac.eventnames:
                pac_j = np.argwhere([str==level for str in pac.eventnames])[0][0]
                for k,ch in enumerate(ch_names):
                    if ch in pac.ch_names:
                        pac_k = np.argwhere([pac_ch==ch for pac_ch in pac.ch_names])[0][0]

                        pac_all_subjects[:,j,k,i] = pac.pac_all[:,pac_j,pac_k]
                    else:
                        pac_all_subjects[:,j,k,i] = np.nan
            else:
                pac_all_subjects[:,j,:,i] = np.nan

    # Eliminate channels with too few observations
    if minobs>0:
        log.info(f'Requiring at least {minobs} observations for each channel and level...')
        nobs = np.sum(np.logical_not(np.isnan(pac_all_subjects[0,:,:,:])),axis=2)
        elim = np.less(nobs,minobs)

        for i in range(Nlevels):
            for j in range(N):
                if elim[i,j]:
                    pac_all_subjects[:,i,j,:] = np.nan

    # Take out entirely empty levels
    empty_levels = [np.all(np.isnan(pac_all_subjects[:,j,:,:])) for j in range(Nlevels)]
    keep_levels = np.logical_not(empty_levels)
    pac_all_subjects = pac_all_subjects[:,keep_levels,:,:]
    eventnames = [name for i,name in enumerate(eventnames) if keep_levels[i]]

    # Compute frequency profiles on the *unaveraged* subject pac_all data
    freq_profiles = compute_frequency_profiles(pac_all_subjects)
    # average the projections across subjects:
    freq_profiles['V_unstacked'] = np.nanmean(freq_profiles['V_unstacked'],axis=3)

    freqs = pacs[0].get_center_freqs()

    freq_profiles = change_freq_profile_signs(freq_profiles,freqs)

    return freq_profiles, freqs, eventnames


def compute_source_space_projections(pacs,Nsubjects,freq_profiles,
                                        eventnames, grade, event_mapping=None,
                                        subjects_dir=None):
    """
    :type pacs: list or generator of SourceSpacePAC
    """
    subject_to = 'fsaverage'
    vertices_to = mne.grade_to_vertices(subject_to, grade=grade,subjects_dir=subjects_dir)

    Nlevels = len(eventnames)
    N = len(vertices_to[0]) + len(vertices_to[1])
    Nf = freq_profiles['U'].shape[1]

    proj_all_subjects = np.empty((Nf,Nlevels,N,Nsubjects))
    for i,pac in enumerate(pacs):
        # Remap events if necessary
        if event_mapping is not None:
            pac.remap_events(event_mapping[pac.subject])

        # Project sensor space modes in source space
        freq_profiles_proj = pac.project_pac_onto_frequency_profiles(freq_profiles['U'])


        # Morph results to the fsaverage brain
        vertices_from = pac.vertices
        morph_mat = mne.compute_morph_matrix(pac.subject, subject_to,
                                             vertices_from, vertices_to,
                                             subjects_dir=subjects_dir)

        def morph_array(data):
            stc_from = mne.SourceEstimate(data, vertices=vertices_from,
                                           tmin=0, tstep=1,
                                           subject=pac.subject)
            stc_to = mne.morph_data_precomputed(pac.subject, subject_to,
                                                  stc_from, vertices_to, morph_mat)
            return stc_to.data

        subject_proj = np.multiply(freq_profiles_proj['S'][:,np.newaxis,np.newaxis],freq_profiles_proj['V_unstacked'])

        for j,level in enumerate(eventnames):
            if level in pac.eventnames:
                pac_j = np.argwhere([str==level for str in pac.eventnames])[0][0]
                for k in range(Nf):
                    proj_all_subjects[k,j,:,i] = morph_array(subject_proj[k,pac_j,:])
            else:
                proj_all_subjects[:,j,:,i] = np.nan

    return proj_all_subjects, vertices_to


###### ROI Summary
def prepare_roi_summary(pacs,eventnames,swtBootstrapMethod,
                        stats_alpha,swtMultipleComparisons,parc='aparc',
                        ordered_rois=None,lobes=None,
                        amp_band_i=0,project_onto_U=None):

    if project_onto_U is not None:
        # If I want to use bootstrap in the projected results, I need to implement it
        assert(swtBootstrapMethod is None)

    subjects = []
    roi_data_all = {}
    reject = {}
    roi_bootstrap_samples = {}
    roi_labels = {}
    for subject,levels_pac in list(pacs.items()):
        log.info(f'Computing PAC in ROIs for {subject}')
        subject = levels_pac.subject
        subjects_dir = levels_pac.subjects_dir

        # load labels
        roi_labels[subject] = mne.read_labels_from_annot(subject,parc,'both',subjects_dir=subjects_dir)

        pac_all, bootstrap_samples = levels_pac.compute_PAC_in_ROIs(roi_labels[subject])
        if project_onto_U is not None:
            freq_profiles_proj = project_pac_onto_frequency_profiles(pac_all,project_onto_U)
            subject_proj = np.multiply(freq_profiles_proj['S'][:,np.newaxis,np.newaxis],freq_profiles_proj['V_unstacked'])
            roi_data_all[subject] = subject_proj[amp_band_i,:,:]
        else:
            roi_data_all[subject] = pac_all[amp_band_i,:,:]
            if swtBootstrapMethod is not None:
                roi_bootstrap_samples[subject] = bootstrap_samples[amp_band_i,:,:,:]
                reject[subject] = compute_rejection(roi_data_all[subject],
                                                     roi_bootstrap_samples[subject],
                                                     swtBootstrapMethod,
                                                     stats_alpha,
                                                     swtMultipleComparisons)

        subjects.append(subject)

    if parc=='aparc' and ordered_rois is None:
        ordered_rois = [
            # Frontal
            'superiorfrontal-lh',
            'superiorfrontal-rh',
            'rostralmiddlefrontal-lh',
            'rostralmiddlefrontal-rh',
            'caudalmiddlefrontal-lh',
            'caudalmiddlefrontal-rh',
            'parsopercularis-lh',
            'parsopercularis-rh',
            'parstriangularis-lh',
            'parstriangularis-rh',
            'parsorbitalis-lh',
            'parsorbitalis-rh',
            'lateralorbitofrontal-lh',
            'lateralorbitofrontal-rh',
            'medialorbitofrontal-lh',
            'medialorbitofrontal-rh',
            'precentral-lh',
            'precentral-rh',
            'paracentral-lh',
            'paracentral-rh',
            'frontalpole-lh',
            'frontalpole-rh',
            'rostralanteriorcingulate-lh',
            'rostralanteriorcingulate-rh',
            'caudalanteriorcingulate-lh',
            'caudalanteriorcingulate-rh',
            # Parietal
            'superiorparietal-lh',
            'superiorparietal-rh',
            'inferiorparietal-lh',
            'inferiorparietal-rh',
            'supramarginal-lh',
            'supramarginal-rh',
            'postcentral-lh',
            'postcentral-rh',
            'precuneus-lh',
            'precuneus-rh',
            'isthmuscingulate-lh',
            'isthmuscingulate-rh',
            'posteriorcingulate-lh',
            'posteriorcingulate-rh',
            # Temporal
            'superiortemporal-lh',
            'superiortemporal-rh',
            'middletemporal-lh',
            'middletemporal-rh',
            'inferiortemporal-lh',
            'inferiortemporal-rh',
            'bankssts-lh',
            'bankssts-rh',
            'fusiform-lh',
            'fusiform-rh',
            'transversetemporal-lh',
            'transversetemporal-rh',
            'entorhinal-lh',
            'entorhinal-rh',
            'temporalpole-lh',
            'temporalpole-rh',
            'parahippocampal-lh',
            'parahippocampal-rh',
            # Occipital
            'lateraloccipital-lh',
            'lateraloccipital-rh',
            'lingual-lh',
            'lingual-rh',
            'cuneus-lh',
            'cuneus-rh',
            'pericalcarine-lh',
            'pericalcarine-rh',
            # Other
            'corpuscallosum-lh',
            'corpuscallosum-rh',
            'insula-lh',
            'insula-rh',
            'unknown-lh',
            'unknown-rh']

        lobes = ['Frontal']*26 + ['Parietal']*14 + ['Temporal']*18 + ['Occipital']*8 + ['Other']*6


    # Collect all ROIs
    all_rois = []
    for subject in subjects:
        all_rois = np.union1d(all_rois,[r.name for r in roi_labels[subject]])

    # assert(all(all_rois[:-1]<all_rois[1:]))
    # assert(all([roi in ordered_rois for roi in all_rois]))
    if not all([roi in all_rois for roi in ordered_rois]):
        warnings.warn("Some ROIs are not represented in the data: {}".format([roi for roi in ordered_rois if not roi in all_rois]),RuntimeWarning)
        new_ordered_rois = []
        new_lobes = []
        for roi,lobe in zip(ordered_rois,lobes):
            if roi in all_rois:
                new_ordered_rois.append(roi)
                new_lobes.append(lobe)
        ordered_rois = new_ordered_rois
        lobes = new_lobes

    Nrois = len(all_rois)
    if swtBootstrapMethod is not None:
        Nbootstrap = roi_bootstrap_samples[subjects[0]].shape[2]
    Nsubjects = len(subjects)
    Nlevels = len(eventnames)

    # Put the data from all subjects into one array
    roi_data_all_collected = np.nan*np.zeros((Nsubjects,Nlevels,Nrois))
    if swtBootstrapMethod is not None:
        reject_collected = False*np.zeros((Nsubjects,Nlevels,Nrois))
        roi_bootstrap_samples_collected = np.nan*np.zeros((Nsubjects,Nlevels,Nrois,Nbootstrap))
    for i,subject in enumerate(subjects):
        subj_rois = np.array([r.name for r in roi_labels[subject]])
        assert(all(subj_rois[:-1]<subj_rois[1:]))
        roi_inds = np.searchsorted(all_rois,subj_rois)
        level_inds = np.array([np.argwhere([str==level for str in eventnames])[0][0] for level in pacs[subject].eventnames])

        # level_inds = np.array([j for j,name in enumerate(eventnames) if name in pacs[subject].eventnames])
        roi_data_all_collected[i,level_inds[:,np.newaxis],roi_inds[np.newaxis,:]] = roi_data_all[subject]
        if swtBootstrapMethod is not None:
            reject_collected[i,level_inds[:,np.newaxis],roi_inds[np.newaxis,:]] = reject[subject]
            roi_bootstrap_samples_collected[i,level_inds[:,np.newaxis],roi_inds[np.newaxis,:],:] = roi_bootstrap_samples[subject]

    # Compute the averages
    roi_data_all_average = np.nanmean(roi_data_all_collected,axis=0)
    if swtBootstrapMethod is not None:
        roi_bootstrap_samples_average = np.nanmean(roi_bootstrap_samples_collected,axis=0)

        reject_average = compute_rejection(roi_data_all_average,
                                             roi_bootstrap_samples_average,
                                             swtBootstrapMethod,
                                             stats_alpha,
                                             swtMultipleComparisons)

    # reorder rois based on lobe
    order = np.searchsorted(all_rois,ordered_rois)
    assert(all(all_rois[order]==ordered_rois))

    lobe_names = np.unique(lobes)
    # lobe_bounds = [(0,25),(26,39),(40,57),(58,65),(66,71)]
    lobe_bounds = []
    for lobe_name in lobe_names:
        lobe_inds = np.arange(len(lobes))[np.array([lobe_name==name for name in lobes])]
        lobe_bounds.append((lobe_inds[0],lobe_inds[-1]))
    lobe_order = np.argsort([bounds[0] for bounds in lobe_bounds])
    lobe_names = lobe_names[lobe_order]
    lobe_bounds = [lobe_bounds[i] for i in lobe_order]


    if swtBootstrapMethod is not None:
        return ordered_rois,lobe_names,lobe_bounds,roi_data_all_collected[:,:,order],reject_collected[:,:,order],roi_data_all_average[:,order],reject_average[:,order]
    else:
        return ordered_rois,lobe_names,lobe_bounds,roi_data_all_collected[:,:,order],None,roi_data_all_average[:,order],None

###### Session PAC
def load_sensor_data(subject,subjects_dir,newFs,
                         phase_band,amp_bands,event_filename,blnSession,
                     events_dir=None):

    mne_output_path = op.join(subjects_dir, subject, 'sourceloc_preproc')

    phase_band_epochs_fpfx = get_epochs_filepfx(
        mne_output_path, subject,
        event_filename, phase_band[0], phase_band[1], newFs)
    phase_band_epochs = mne.read_epochs(phase_band_epochs_fpfx + '-epo.fif', proj='delayed', preload=True)

    picks = mne.pick_types(phase_band_epochs.info, meg=False, eeg=True, exclude='bads')
    pos = mne.find_layout(phase_band_epochs.info,exclude=[]).pos[picks]

    phase_band_epochs.pick_types(meg=False, eeg=True)

    amp_band_epochs = []
    for amp_band in amp_bands:
        amp_band_epochs_fpfx = get_epochs_filepfx(
            mne_output_path, subject,
            event_filename, amp_band[0], amp_band[1], newFs)
        abe = mne.epochs.read_epochs(amp_band_epochs_fpfx + '-epo.fif', proj='delayed', preload=True)
        abe.pick_types(meg=False, eeg=True)
        amp_band_epochs.append(abe)

    if blnSession:
        # Redefine events so they won't be averaged
        log.info('Redefining Events...')
        event_times = get_times_for_epochs(mne_output_path,subject,phase_band[0], phase_band[1],newFs,event_filename)
        if phase_band_epochs.tmin==0.:
            session_timeaxis = event_times + phase_band_epochs.tmax/2
        else:
            session_timeaxis = event_times

        new_events = {'{}-{}s'.format(t+phase_band_epochs.tmin,t+phase_band_epochs.tmax):t for t in event_times}
        new_eventnames = ['{}-{}s'.format(t+phase_band_epochs.tmin,t+phase_band_epochs.tmax)  for t in event_times]

        Nevents = len(phase_band_epochs)
        mne_event_id = {new_eventnames[i]:i for i in range(Nevents)}

        for abe in amp_band_epochs:
            abe.events[:,2] = list(range(Nevents))
            abe.event_id = mne_event_id
        phase_band_epochs.events[:,2] = list(range(Nevents))
        phase_band_epochs.event_id = mne_event_id

    else:
        if events_dir is None:
            eventfile = op.join(subjects_dir, subject, 'sourceloc_preproc',
                                event_filename)
        else:
            eventfile = op.join(events_dir,event_filename)
        new_events, new_eventnames, _, _ = utils.load_eventfile(eventfile)
        session_timeaxis = np.arange(len(new_eventnames))

    return amp_band_epochs,phase_band_epochs,new_events,new_eventnames,session_timeaxis,pos


## Subject pipeline
def run_pipeline(subjects_dir,subjects,blnsave,
                 blnSourceSpace,blnOffscreen,blnAcrossFrequencies,blnSession,
                 swtPACmetric='corr',events_dir=None,neighbor_dict=None):
    """Run spatial PAC pipeline

    Parameters
    ----------
    subjects_dir : basestring
        The subjects directory.
    subject : basestring
        The subject ID.
    blnsave: bool
        Whether to save the figures.
    blnSourceSpace : bool
        Whether to run the analysis in source space.
    blnOffscreen : bool
        Whether to plot the figures offscreen (only for source surfaces)
    blnAcrossFrequencies : bool
        Whether to compute the analysis across frequencies
        True -> range(4,50,2) with 2 Hz bandwidth
        False -> [8,16] (alpha/beta)
    blnSession : bool
        Whether to compute the analysis for the entire session, i.e. for a
        comodulogram.
        False -> use events from event_times_level.mat
    swtPACmetric : basestring
        Which PAC metric to use (see Tort et al 2010)
        Options:
            'corr' -> Correlation (default)
            'corr_noPhaseBand_Norm' -> Correlation, but
                don't include phase band variance in the denominator
            'height_ratio' -> (amp_up-amp_down)/max(amp_up,amp_down)
                where amp_up is the amplitude when the slow is positive
                and amp_down is the amplitude when the slow is negative
            'mean_vector' -> mean(amp*exp(i*phase)) see Tort et al 2010

    """
    newFs = 200
    slow_band = (0.1,4)

    blnPhaseHilbert = swtPACmetric=='mean_vector'

    if blnSession:
        event_filename = None
        event_suffix = ''

        if blnSourceSpace:
            raise RuntimeError('Not Implemented')
    else:
        event_filename = 'event_times_level.mat'
        event_suffix = '_byLevel'

    if blnAcrossFrequencies:
        center_freqs = range(4,50,2)
        bandwidth = 2.
        amp_bands = [(freq-bandwidth/2,freq+bandwidth/2) for freq in center_freqs]
        blnAmpHilbert = True

        Nbootstrap = 0
        swtBootstrapMethod = None
        MultipleComparisons = []  # None,'Bonferroni','FDR','maxstat'
        stats_alpha = 0.05

        if blnSession:
            PAC_foldername = 'PAC_session_{}-{}_{}Hz'.format(slow_band[0],slow_band[1],bandwidth)
        else:
            PAC_foldername = 'PAC_across_frequencies_{}-{}_{}Hz'.format(slow_band[0],slow_band[1],bandwidth)
    else:
        amp_bands = [[8, 16]]
        blnAmpHilbert = True

        Nbootstrap = 1000
        stats_alpha = 0.05
        MultipleComparisons = ['FDR']  # None,'Bonferroni','FDR','maxstat'
        swtBootstrapMethod = 'permtest_epochs_wi_level'  # 'null', 'classic', 'permtest_levels', None, 'permtest_epochs_wi_level'

        if blnSession:
            PAC_foldername = 'PAC_session_{}-{}_{}-{}'.format(slow_band[0],slow_band[1],amp_bands[0][0],amp_bands[0][1])
        else:
            PAC_foldername = 'PAC_{}-{}_{}-{}'.format(slow_band[0],slow_band[1],amp_bands[0][0],amp_bands[0][1])


    if blnSourceSpace:
        spacing_string = '-ico-3'
        depth = None

        if depth is None:
            depth_suffix = ''
        else:
            depth_suffix = '-depth{}'.format(depth)

        obsCov_suffix = ''
        meeg_suffix = '-eeg'

        figure_foldername = op.join('Figures sourceloc','power{}{}'.format(depth_suffix,event_suffix))
    else:
        figure_foldername = 'Figures sensor space'

    for subject in subjects:
        figures_output_path = op.join(subjects_dir, subject,
                                      figure_foldername,
                                      PAC_foldername,
                                      swtPACmetric)
        if blnsave and not op.exists(figures_output_path):
            makedirs(figures_output_path)

        logfilename = pac_savename(subject, event_filename, blnPhaseHilbert, blnAmpHilbert,slow_band,amp_bands,
                                        swtPACmetric, swtBootstrapMethod)+'_{}.log'
        lh = utils.initialize_log_file(figures_output_path,logfilename)

        if blnSourceSpace:
            mne_output_path = op.join(subjects_dir, subject, 'sourceloc_preproc')

            pac = SourceSpacePAC(subject, subjects_dir, newFs, slow_band,
                                      amp_bands, blnPhaseHilbert, blnAmpHilbert, Nbootstrap, swtPACmetric,
                                      swtBootstrapMethod, event_filename,
                                      mne_output_path, spacing_string,
                                      obsCov_suffix, meeg_suffix,
                                      depth, blnOffscreen, figures_output_path,
                                      events_dir=events_dir)

        else:
            pac = SensorSpacePAC(subject, subjects_dir, newFs, slow_band,
                                      amp_bands, blnPhaseHilbert, blnAmpHilbert, Nbootstrap, swtPACmetric,
                                      swtBootstrapMethod, event_filename,
                                      figures_output_path, blnSession,
                                      events_dir=events_dir,
                                      neighbor_dict=neighbor_dict)

        pac.run_computation(blnsave)
        if blnAcrossFrequencies and not blnSourceSpace:
            pac.compute_frequency_profiles(blnsave)
        utils.reset_logging_to_file(lh)

    for subject in subjects:
        plt.close('all')
        figures_output_path = op.join(subjects_dir, subject,
                                      figure_foldername,
                                      PAC_foldername,
                                      swtPACmetric)
        savename = pac_savename(subject, event_filename, blnPhaseHilbert, blnAmpHilbert,slow_band,amp_bands,
                                     swtPACmetric, swtBootstrapMethod)

        if blnSourceSpace:
            pac = read_SourceSpacePAC(figures_output_path,savename)
        else:
            pac = read_SensorSpacePAC(figures_output_path,savename)


        if not blnAcrossFrequencies:
            for swtMultipleComparisons in MultipleComparisons:
                pac.plot_results(stats_alpha,swtMultipleComparisons,blnsave)

            if swtPACmetric=='mean_vector':
                pac.plot_total_phase_summary('phase_summary',blnsave)

        if blnSession :
            if blnAcrossFrequencies:
                Nfreqs = len(amp_bands)

                for i in range(Nfreqs):
                    pac.plot_PAC_across_sensors(blnsave, amp_band_i=i)

                for i in range(pac.N):
                    pac.plot_PAC_across_frequencies(blnsave,i)
            else:
                pac.plot_PAC_across_sensors(blnsave)
                pac.plot_PAC_traces(blnsave)

        if blnAcrossFrequencies and not blnSourceSpace:
            pac.change_freq_profile_signs()
            pac.plot_freq_profile_singular_values(blnsave)
            pac.plot_freq_profiles(blnsave)
            pac.plot_freq_profile_projections_byLevel(blnsave)
            if not blnSession:
                pac.plot_freq_profile_projections_topos(blnsave)
