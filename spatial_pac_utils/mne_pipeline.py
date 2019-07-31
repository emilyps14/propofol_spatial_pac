# Author: Emily P. Stephen <emilyps14@gmail.com>

import os.path as op
import matplotlib.pyplot as plt
import mne
import numpy as np
import copy as cp
import warnings
import logging
from scipy.signal import detrend
from scipy import linalg,sparse
from math import sqrt
from copy import deepcopy
from os import makedirs
from spatial_pac_utils import utils


log = logging.getLogger(__name__)

class MNE_Pipeline(object):
    def __init__(self, mne_output_path, filebasestr, srcfile, spacing_string, transfile, bemfile, l_freq=None,
                 h_freq=200, newFs=400, cov_freqthresh=65, blnDiagObsCov=True, amplitude_snr=3.0, l_trans_bandwidth=None, h_trans_bandwidth=None,
                 blnOverwrite=False,
                 depth=None):
        
        log.info('Initializing MNE Pipeline...')
        self.mne_outputpath = mne_output_path
        self.filebasestr = filebasestr
        self.srcfile = srcfile
        self.spacing_string = spacing_string
        self.transfile = transfile
        self.bemfile = bemfile

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.newFs = newFs
        self.cov_freqthresh = cov_freqthresh
        self.blnDiagObsCov = blnDiagObsCov
        self.amplitude_snr = amplitude_snr
        self.blnOverwrite = blnOverwrite
        self.depth = depth

        log.info('[Done]')

    def process_raw(self, raw, blnCov=False):
        """ Preprocess data as Raw structures.
        (1) if no highpass, linear detrend
        (2) filter
        (3) downsample
        (4) save processed chunk

        :rtype : BaseRaw, string
        """

        if self.l_trans_bandwidth is None:
            self.l_trans_bandwidth = 'auto'
        if self.h_trans_bandwidth is None:
            self.h_trans_bandwidth = 0.5

        if blnCov:
            lf = self.cov_freqthresh
            hf = self.newFs / 2
            ltrans = 'auto'
            htrans = 0.5
        else:
            lf = self.l_freq
            hf = self.h_freq
            ltrans = self.l_trans_bandwidth
            htrans = self.h_trans_bandwidth

        blnOverwrite = self.blnOverwrite
        filebasestr = self.filebasestr
        fs = self.newFs

        dsPfx = get_rawchunk_filepfx(self.mne_outputpath,self.filebasestr,
                                     lf,hf,fs,
                                     0.,None)

        if not blnOverwrite and op.isfile(dsPfx + '-raw.fif'):
            log.info('File Exists, loading: ' + dsPfx + '-raw.fif')
            rawchunk = mne.io.Raw(dsPfx + '-raw.fif', preload=True, verbose=None)
        else:
            rawchunk = raw.copy()
            rawchunk.load_data()

            if lf is None:
                # linear detrend
                rawchunk._data = detrend(rawchunk._data,axis=1,type='linear')
                rawchunk.info['detrend'] = 'linear'
            else:
                rawchunk.info['detrend'] = None

            # band pass filter
            rawchunk.filter(lf, hf, filter_length='auto',fir_design='firwin',
                            l_trans_bandwidth=ltrans, h_trans_bandwidth=htrans,
                            method='fir', iir_params=None,
                            skip_by_annotation=['EDGE'], verbose=None)

            # downsample
            rawchunk.resample(fs,npad='auto')

            rawchunk.save(dsPfx + '-raw.fif', verbose=None, overwrite=blnOverwrite)

        rawchunk.info['bads'] = raw.info['bads'] # in case it was loaded from a file with old bads

        log.info('[Done]')
        return rawchunk, dsPfx


    def compute_ObsNoiseCov(self, hpchunk, fpfx):
        lf = self.l_freq
        hf = self.h_freq
        fs = self.newFs
        threshf = self.cov_freqthresh
        blnDiag = self.blnDiagObsCov
        blnOverwrite = self.blnOverwrite

        if blnDiag:
            covpath = fpfx + '-diag-cov.fif'
        else:
            covpath = fpfx + '-cov.fif'

        if not blnOverwrite and op.isfile(covpath):
            log.info('Loading observation noise cov...')
            cov = mne.cov.read_cov(covpath)
        else:
            log.info('Computing observation noise cov based on high frequency content')
            # C0 = (Y_hp*Y_hp')/(size(Y_hp,2)-1);
            if isinstance(hpchunk,mne.epochs.BaseEpochs):
                cov = mne.cov.compute_covariance(hpchunk, method='empirical')
            elif isinstance(hpchunk,mne.io.base.BaseRaw):
                cov = mne.cov.compute_raw_covariance(hpchunk,method='empirical')
            else:
                raise RuntimeError('High-pass data must be Raw or Epochs, was type ' + str(hpchunk.__class__))

            # % Scale noise to account for power below Fpass_hp, and b/w [Fpass1_bp Fpass2_bp]
            # Nyq_freq = Fs/2;
            # C0 = (Fpass2_bp-Fpass1_bp)/(Nyq_freq-Fpass_hp)*C0;
            nyq_freq = fs / 2
            lf = 0 if lf is None else lf
            cov['data'] *= (hf - lf) / (nyq_freq - threshf)

            # % Regularize noise covariance matrix (diagonal loading)
            # Eke = 0.1;
            # C0 = C0 + Eke*mean(diag(C0))*eye(size(C0,1));
            cov = mne.cov.regularize(cov,hpchunk.info,proj=False)

            if blnDiag:
                # Throw away off-diagonal terms
                cov = cov.as_diag()

            mne.cov.write_cov(covpath, cov)

        log.info('[Done]')
        return cov


    def get_forward_sol(self, info,meg=False,eeg=True,force_fixed=True):
        blnOverwrite = self.blnOverwrite
        srcfile = self.srcfile
        transfile = self.transfile
        bemfile = self.bemfile

        meeg_sfx = get_meeg_suffix(bln_eeg=eeg,bln_meg=meg)
        fwdfile = get_fwdfile(self.mne_outputpath, self.filebasestr,
                              self.spacing_string, meeg_sfx)

        #
        # Compute forward solution
        #
        log.info('Reading the source space')
        src = mne.read_source_spaces(srcfile, patch_stats=True, verbose=None)

        if not blnOverwrite and op.isfile(fwdfile):
            log.info('Loading the forward solution')
            fwd = mne.read_forward_solution(fwdfile, include=[], exclude=[], verbose=None)
        else:
            log.info('Making the forward solution')
            fwd = mne.make_forward_solution(info, transfile, src, bemfile,
                                            meg=meg, eeg=eeg, mindist=5.0, ignore_ref=False)
            mne.write_forward_solution(fwdfile,fwd,overwrite=blnOverwrite)

        fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=force_fixed,
                                           copy=False, use_cps=True, verbose=None)

        log.info('[Done]')
        return src, fwd

    def make_inverse_operator(self, info, src, fwd, obs_cov, fpfx, obsCov_suffix='', meeg_suffix='', blnsave=True):
        blnOverwrite = self.blnOverwrite

        fpfx_inv = _get_stc_filepfx_helper(fpfx,obsCov_suffix,meeg_suffix,self.spacing_string,self.depth)

        if not blnOverwrite and op.isfile(fpfx_inv + '-inv.fif'):
            log.info('Loading the inverse operator ' + fpfx_inv)
            inv = mne.minimum_norm.read_inverse_operator(fpfx_inv + '-inv.fif')
        else:
            log.info('Computing the inverse operator ' + fpfx_inv)
            inv = mne.minimum_norm.make_inverse_operator(info,fwd,obs_cov,loose=0.0,depth=self.depth,fixed=True,use_cps=True)
            if blnsave:
                log.info('Saving the inverse operator ' + fpfx_inv)
                mne.minimum_norm.write_inverse_operator(fpfx_inv + '-inv.fif',inv)

        return inv, fpfx_inv


    def _check_stc_file(self,stc_fpfx,ftype):
        if ftype=='stc':
            blnFileExists = op.isfile(stc_fpfx + '-lh.stc') and op.isfile(stc_fpfx + '-rh.stc')
            loadPath = stc_fpfx + '-lh.stc'
        elif ftype=='w':
            blnFileExists = op.isfile(stc_fpfx + '-lh.w') and op.isfile(stc_fpfx + '-rh.w')
            loadPath = stc_fpfx + '-lh.w'
        elif ftype=='h5':
            blnFileExists = op.isfile(stc_fpfx + '-stc.h5')
            loadPath = stc_fpfx + '-stc.h5'

        return blnFileExists,loadPath


    def process_inverse_sol_epochs(self, epochs, inv, src, stc_filepfx_list, ftype='stc'):
        Nepochs = len(epochs)
        assert(len(stc_filepfx_list)==Nepochs)
        step = 20
        for istart in range(0, Nepochs, step):  # step through in sets of 20 epochs
            iend = min(istart + step, Nepochs)
            epochs_sel = epochs[istart:iend]
            stc_pfx_sel = stc_filepfx_list[istart:iend]

            # check if the last one has already been run
            blnFilesExist = all([self._check_stc_file(pfx,ftype)[0] for pfx in stc_pfx_sel])
            if not self.blnOverwrite and blnFilesExist:
                log.info('Skipping stcs ' + str(istart) + ' until ' + str(iend))
            else:
                log.info('Running stcs ' + str(istart) + ' until ' + str(iend))
                stcs_sel = self.get_inverse_sol_epochs_generator(epochs_sel,inv,src)
                for stc, stc_pfx in zip(stcs_sel,
                                         stc_pfx_sel):  # need to use itertools.izip to return a generator before Python 3
                    log.info('Saving source estimates ' + stc_pfx)
                    stc.save(stc_pfx, ftype=ftype)


    def get_inverse_sol_epochs_generator(self,epochs,inv,src,prepared=False):
        power_snr = self.amplitude_snr ** 2
        lambda2 = 1.0 / power_snr

        stcs = mne.minimum_norm.inverse.apply_inverse_epochs(epochs, inv, lambda2, method='MNE',
                                                             nave=1, pick_ori=None,
                                                             return_generator=True,
                                                             prepared=prepared, verbose=True)
        for stc in stcs:
            yield stc

    def prepare_MNE_epochs(self, epochs, fpfx_epochs, cov, blnSaveInv,
                           obsCov_suffix, bln_eeg=True, bln_meg=False):

        # Just in case -- this should already have been run in process_chunk
        # In that case, these lines will have no effect.
        epochs.pick_types(eeg=bln_eeg,meg=bln_meg,exclude='bads')
        if bln_eeg:
            with warnings.catch_warnings():
                # Don't raise RuntimeWarning if the eeg reference already exists
                warnings.simplefilter("ignore")
                mne.io.set_eeg_reference(epochs,ref_channels='average',copy=False,projection=True)
                epochs.apply_proj()

        meeg_sfx = get_meeg_suffix(bln_meg,bln_eeg)

        #
        # Compute forward solution
        #
        src, fwd = self.get_forward_sol(epochs.info,meg=bln_meg,eeg=bln_eeg)

        #
        # Compute inverse operator and obtain source estimates
        #
        inv, fpfx_inv = self.make_inverse_operator(epochs.info, src, fwd, cov, fpfx_epochs,
                                                   obsCov_suffix=obsCov_suffix,
                                                   meeg_suffix=meeg_sfx, blnsave=blnSaveInv)

        return inv, src


def get_meeg_suffix(bln_meg,bln_eeg):
    sfx = ''
    if bln_eeg:
        sfx += '-eeg'
    if bln_meg:
        sfx += '-meg'
    return sfx


def get_fwdfile(mne_outputpath, filebasestr, spacing_string, meeg_suffix):
    fwd_fname = filebasestr + meeg_suffix + spacing_string + '-fwd.fif'
    fwdfile = op.join(mne_outputpath, fwd_fname)

    return fwdfile


def get_stc_filepfx(mne_outputpath,filebasestr,l_freq,h_freq,newFs,tmin,tmax,obsCov_suffix,meeg_suffix,spacing_string,depth=None):
    fpfx = get_rawchunk_filepfx(mne_outputpath,filebasestr,l_freq,h_freq,newFs,tmin,tmax)
    return _get_stc_filepfx_helper(fpfx,obsCov_suffix,meeg_suffix,spacing_string,depth)

def _get_stc_filepfx_helper(chunk_fpfx,obsCov_suffix,meeg_suffix,spacing_string,dept):
    fpfx = chunk_fpfx
    fpfx += obsCov_suffix
    fpfx += meeg_suffix
    fpfx += spacing_string

    return fpfx


def get_rawchunk_filepfx(mne_outputpath,filebasestr,l_freq,h_freq,newFs,tmin,tmax):
    if tmax is None:
        chunkstr = str(int(tmin)) + '-' + str(None)
    else:
        chunkstr = str(int(tmin)) + '-' + str(int(tmax))
    fpfx = op.join(mne_outputpath,filebasestr
            + '-crop-' + chunkstr
            + '-filt-' + str(l_freq) + '-' + str(int(h_freq))
            + '-ds' + str(newFs))
    return fpfx


def get_epochs_filepfx(mne_outputpath,filebasestr,event_filename,l_freq,h_freq,newFs):
    if event_filename is None:
        event_filename = ''
    else:
        event_filename = '-' + event_filename[:-4]

    fpfx = op.join(mne_outputpath,filebasestr + event_filename
                    + '-filt-' + str(l_freq) + '-' + str(h_freq)
                    + '-ds' + str(newFs))
    return fpfx


def get_events_from_epochs(mne_outputpath,eegbasestr,l_freq,h_freq,newFs,event_filename):

    fpfx_epochs = get_epochs_filepfx(mne_outputpath,eegbasestr,event_filename,l_freq,h_freq,newFs)
    epochs = mne.read_epochs(fpfx_epochs + '-epo.fif', preload=False, verbose=False)

    fpfx_raw = get_rawchunk_filepfx(mne_outputpath,eegbasestr,l_freq,h_freq,newFs,0,None)
    rawproc = mne.io.Raw(fpfx_raw + '-raw.fif', preload=False, verbose=False)

    eventnames = list(epochs.event_id.keys())
    events = {}
    for event_name in eventnames:
        events[event_name] = rawproc.times[epochs[event_name].events[:,0]-rawproc.first_samp]

    return events,eventnames


def get_times_for_epochs(mne_outputpath,eegbasestr,l_freq,h_freq,newFs,event_filename):
    fpfx_epochs = get_epochs_filepfx(mne_outputpath,eegbasestr,event_filename,l_freq,h_freq,newFs)
    epochs = mne.read_epochs(fpfx_epochs + '-epo.fif', preload=False, verbose=False)

    fpfx_raw = get_rawchunk_filepfx(mne_outputpath,eegbasestr,l_freq,h_freq,newFs,0,None)
    rawproc = mne.io.Raw(fpfx_raw + '-raw.fif', preload=False, verbose=False)

    event_times = rawproc.times[epochs.events[:,0]-rawproc.first_samp]

    return event_times


def check_bads(info_bads,bads):
    info_bads += bads
    info_bads = list(set(info_bads))  # remove duplicates
    return info_bads


def run_pipeline_epochs(subject, paths_dict, obs_cov_filepath=None, epoch_tmin=0., epoch_tmax=30., l_freq=None, h_freq=100, newFs=200,
                        cov_freqthresh=65, blnDiagObsCov=True, amplitude_snr=3.0, l_trans_bandwidth=None, h_trans_bandwidth=None, blnOverwrite=False,
                        blnSaveInv=True, spacing_string='-ico-5', out_folder='',
                        event_filename=None, obsCov_suffix='',baseline=None,detrend=1,blnRunSourceLoc=True,
                        depth=None, filebasestr=None):
    """Run preprocessing pipeline and source localization (optional).

    Parameters
    ----------
    subject : basestring
        The subject ID.
    paths_dict : dict
        Dictionary of paths, including:
            subjects_dir: subjects directory
            srcfile: source file (*-src.fif)
            bemfile: BEM file (*-bem-sol.fif)
            transfile: trans file (COR*.fif)
            rawfile: raw data file (*-raw.fif)
            badchannels_eeg: EEG bad channels (*.mat)
            badchannels_meg: MEG bad channesl (*.mat)
            badchannels_addPfx (optional): Whether to add 'EEG' and 'MEG'
                prefixes to the bad channels when loading (default: False)
    obs_cov_filepath : basestring | None
        Path to the observation noise covariance matrix. If None, the matrix
        will be generated from scratch based on high frequency power
    epoch_tmin : float
        The start time of each epoch in seconds, relative to the event time.
    epoch_tmax : float
        The end time of each epoch in seconds, relative to the event time.
    l_freq : float | None
        The low frequency cutoff of the bandpass filter, in Hz (if None, will
        linearly detrend the session).
    h_freq : float | None
        The high frequency cutoff of the bandpass filter, in Hz (if None, the
        data will only be high-passed).
    newFs : float
        The data will be downsampled to this frequency.
    cov_freqthresh : float
        The frequency cutoff for the observation noise covariance estimation.
    blnDiagObsCov : bool
        Whether the observation noise covariance is diagonal.
    amplitude_snr : float
        The amplitude SNR of the data
    l_trans_bandwidth : float | None
        The lower transition band for the filter. If None, will be 'auto'.
    h_trans_bandwidth : float | None
        The upper transition band for the filter. If None, will be 'auto'.
    blnOverwrite : bool
        Whether to overwrite existing preprocessed data
    blnSaveInv : bool
        Whether to save the inverse model.
    spacing_string : basestring
        The spacing string for the source space (e.g. '-ico-5')
    out_folder : basestring
        Path to the folder to save the preprocessed data.
    event_filename : basestring
        The event file name '*.mat'. If None, will use fixed length
        events.
    obsCov_suffix : basestring
        Suffix to indicate the type of observation noise covariance in file
        names.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction in Epochs generation
        (see mne.BaseEpochs).
    detrend : int | None
        If 0 or 1, the data channels (MEG and EEG) will be detrended when
        loaded. 0 is a constant (DC) detrend, 1 is a linear detrend. None
        is no detrending (see mne.BaseEpochs).
    blnRunSourceLoc : bool
        Whether to run the source localization.
    depth : float | None
        Depth parameter for source localization
    filebasestr : basestring
        The base string for saving the results, if different from subject.
    """

    # Files
    subjects_dir = paths_dict['subjects_dir']
    srcfile = paths_dict['srcfile']
    bemfile = paths_dict['bemfile']
    transfile = paths_dict['transfile']
    rawfile = paths_dict['rawfile']
    events_dir = paths_dict.get('events_dir',None)
    badchannels_eeg = paths_dict['badchannels_eeg']
    badchannels_meg = paths_dict['badchannels_meg']
    badchannels_addPfx = paths_dict.get('badchannels_addPfx',False)

    if out_folder=='':
        out_folder = subjects_dir

    mne_output_path = op.join(out_folder, subject, 'sourceloc_preproc')

    if filebasestr is None:
        filebasestr = subject

    fpfx = get_epochs_filepfx(mne_output_path,filebasestr,event_filename,l_freq,h_freq,newFs)

    blnHPObsCov = obs_cov_filepath is None
    if blnHPObsCov:
        if blnDiagObsCov:
            obs_cov_filepath = fpfx + '-diag-cov.fif'
        else:
            obs_cov_filepath = fpfx + '-cov.fif'

    mne.set_log_level('INFO')
    mne.set_config('SUBJECTS_DIR', subjects_dir, set_env=True)

    if not op.exists(mne_output_path):
        makedirs(mne_output_path)

    # set up logging output file to go with processed data
    lh = utils.initialize_log_file(mne_output_path,'mne_pipeline_epochs_{}.log')

    mp = MNE_Pipeline(mne_output_path, filebasestr, srcfile, spacing_string, transfile, bemfile,
                      l_freq=l_freq, h_freq=h_freq, newFs=newFs, cov_freqthresh=cov_freqthresh, blnDiagObsCov=blnDiagObsCov,
                      amplitude_snr=amplitude_snr, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, blnOverwrite=blnOverwrite,
                      depth=depth)

    bads,bln_eeg,bln_meg = utils.collect_bad_channels(badchannels_eeg,badchannels_meg,subject,
                                                      blnAddPfx=badchannels_addPfx)
    # Split into epochs
    if not blnOverwrite and op.isfile(fpfx + '-epo.fif'):
        log.info('Loading events and epochs ' + fpfx  + '-epo.fif')
        # events = mne.read_events(fpfx + '.eve')
        epochs = mne.read_epochs(fpfx + '-epo.fif', proj='delayed')
        epochs.info['bads'] = check_bads(epochs.info['bads'],bads)

        ev,_ = get_events_from_epochs(mne_output_path,filebasestr,l_freq,h_freq,newFs,event_filename)
        event_onsets = np.concatenate(list(ev.values()))
        event_onsets.sort()
    else:
        log.info('Creating events and epochs ' + fpfx  + '-epo.fif')

        raw_pfx = get_rawchunk_filepfx(mne_output_path,filebasestr,l_freq,h_freq,newFs,0,None)
        if op.exists(raw_pfx + '-raw.fif'):
            log.info('File Exists, loading: ' + raw_pfx + '-raw.fif')
            rawproc = mne.io.Raw(raw_pfx + '-raw.fif', preload=True, verbose=None)
            rawproc.info['bads'] = check_bads(rawproc.info['bads'],bads)
        else:
            #
            # Load and process raw data
            #
            raw = mne.io.Raw(rawfile, verbose=None, preload=False)
            raw.info['bads'] = check_bads(raw.info['bads'],bads)

            # Preprocess Raw
            rawproc,_ = mp.process_raw(raw, blnCov=False)
            hpproc,_ = mp.process_raw(raw, blnCov=True)

        # create events and epochs
        if event_filename is None:
            event_id = {'fixed-length':1}
            mne_events = mne.make_fixed_length_events(rawproc, event_id['fixed-length'], duration=epoch_tmin+epoch_tmax, start=-epoch_tmin)
        else:
            if events_dir is None:
                eventfile = op.join(subjects_dir, subject, 'sourceloc_preproc', event_filename)
            else:
                eventfile = op.join(events_dir,event_filename)

            events,eventnames,_,_ = utils.load_eventfile(eventfile)
            mne_events,event_id = utils.make_mne_events_from_events(events,eventnames,rawproc)

        epochs = mne.Epochs(rawproc, mne_events, event_id=event_id, tmin=epoch_tmin, tmax=epoch_tmax,
                            baseline=baseline, detrend=detrend, preload=True, proj='delayed')

        mne.write_events(fpfx + '.eve', mne_events)
        epochs.save(fpfx + '-epo.fif')

        event_onsets = rawproc.times[mne_events[:,0]-rawproc.first_samp]

    if not blnHPObsCov or (not blnOverwrite and op.isfile(obs_cov_filepath)):
        log.info('Loading observation noise covariance ' + obs_cov_filepath)
        cov = mne.cov.read_cov(obs_cov_filepath)
    else:
        log.info('Computing observation noise covariance ' + obs_cov_filepath)
        if baseline is None:
            cov_baseline = None
        else:
            cov_baseline = (None,None)

        hpproc_pfx = get_rawchunk_filepfx(mne_output_path,filebasestr,cov_freqthresh,int(newFs/2),newFs,tmin=0,tmax=None)
        hpproc = mne.io.Raw(hpproc_pfx + '-raw.fif', preload=True, verbose=None)
        hpproc.info['bads'] = check_bads(hpproc.info['bads'],bads)


        # Add CAR *and apply*
        hpproc.pick_types(eeg=True,meg=True,exclude='bads')
        mne.io.set_eeg_reference(hpproc,ref_channels='average',copy=False,projection=True)
        hpproc.apply_proj()

        hp_epochs = mne.Epochs(hpproc, epochs.events,event_id=epochs.event_id, tmin=epochs.tmin, tmax=epochs.tmax,
                        baseline=cov_baseline, detrend=detrend, proj='delayed')
        cov = mp.compute_ObsNoiseCov(hp_epochs, fpfx)

    if blnRunSourceLoc:
        stc_filepfx_list = [
            get_stc_filepfx(mne_output_path, filebasestr, l_freq, h_freq, newFs, event_time + epoch_tmin, event_time + epoch_tmax,
                            obsCov_suffix,
                            get_meeg_suffix(bln_meg, bln_eeg), spacing_string,
                            depth=depth) for event_time in event_onsets]

        inv,src = mp.prepare_MNE_epochs(epochs, fpfx, cov, blnSaveInv,
                              obsCov_suffix, bln_eeg=bln_eeg, bln_meg=bln_meg)

        mp.process_inverse_sol_epochs(epochs,inv,src,stc_filepfx_list)

    log.info('[done.]')
    utils.reset_logging_to_file(lh)


