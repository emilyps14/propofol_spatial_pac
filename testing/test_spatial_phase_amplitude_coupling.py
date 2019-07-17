from spatial_pac_utils import mne_pipeline
from spatial_pac_utils import spatial_phase_amplitude_coupling as spac
import numpy as np
import pytest
import mne
import os.path as op
from mne.datasets import testing
from mne.io import read_raw_fif
from mne import make_fixed_length_events, Epochs
from mne.utils import _TempDir
import warnings

mne.set_log_level(verbose='WARNING')
warnings.simplefilter('ignore',RuntimeWarning,lineno=185,append=True) # laplacian with no neighbors
warnings.simplefilter('ignore',RuntimeWarning,lineno=52,append=True) # average reference projection
warnings.simplefilter('ignore',RuntimeWarning,lineno=63,append=True) # short filter
warnings.simplefilter('ignore',FutureWarning,lineno=1593,append=True) # scipy multidim indexing


data_path = testing.data_path(download=False)
fname_raw = op.join(data_path,'MEG','sample','sample_audvis_trunc_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample','sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_src = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-4-src.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem', 'sample-1280-1280-1280-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-trans.fif')

params_acrossFrequencies=dict(
    slow_band = [0.1, 4],
    blnPhaseHilbert = False,
    swtPACmetric = 'corr',  # 'corr','corr_noPhaseBand_Norm','height_ratio', 'mean_vector'

    center_freqs = list(range(4,20,2)),
    bandwidth = 2.,
    amp_bands = [(freq-1.,freq+1.) for freq in range(4,20,2)],
    blnAmpHilbert = True,

    Nbootstrap = 0,
    swtBootstrapMethod = None,
    MultipleComparisons = [],  # None,'Bonferroni','FDR','maxstat'
    stats_alpha = 0.05,
)

epoch_tmin = -1.
epoch_tmax = 8.
winlen = 10.

@pytest.fixture(scope='module')
def raw_and_events():
    raw = read_raw_fif(fname_raw,preload=True)
    events = make_fixed_length_events(raw,duration=winlen,start=-epoch_tmin)
    raw.pick_types(meg=False,eeg=True,exclude='bads')
    mne.io.set_eeg_reference(raw,ref_channels='average',copy=False,projection=True)
    raw.apply_proj()
    return raw,events

@pytest.fixture(scope='module')
def sensor_space_slow_epochs(raw_and_events):
    raw,events = raw_and_events
    raw = raw.copy()
    raw.filter(0.1, 4, filter_length='auto',fir_design='firwin',
                l_trans_bandwidth='auto', h_trans_bandwidth=0.5,
                method='fft', iir_params=None,
                skip_by_annotation=['EDGE'], verbose=None)
    eeg_epo = Epochs(raw,events,None,epoch_tmin,epoch_tmax,baseline=None,reject=None,proj='delayed',preload=True)
    return eeg_epo

@pytest.fixture(scope='module')
def sensor_space_hp_epochs(raw_and_events):
    raw,events = raw_and_events
    raw = raw.copy()
    raw.filter(65, None, filter_length='auto',fir_design='firwin',
                l_trans_bandwidth='auto', h_trans_bandwidth=0.5,
                method='fft', iir_params=None,
                skip_by_annotation=['EDGE'], verbose=None)

    eeg_hp_epo = Epochs(raw,events,None,epoch_tmin,epoch_tmax,baseline=None,reject=None,proj='delayed',preload=True)
    return eeg_hp_epo

@pytest.fixture(scope='module')
def sensor_space_alpha_epochs(raw_and_events):
    raw,events = raw_and_events
    raw = raw.copy()
    raw.filter(8, 15, filter_length='auto',fir_design='firwin',
                l_trans_bandwidth='auto', h_trans_bandwidth=0.5,
                method='fft', iir_params=None,
                skip_by_annotation=['EDGE'], verbose=None)

    eeg_alpha_epo = Epochs(raw,events,None,epoch_tmin,epoch_tmax,baseline=None,reject=None,proj='delayed',preload=True)
    return eeg_alpha_epo

@pytest.fixture(scope='module')
def sensor_space_epochs_acrossFreqs(raw_and_events):
    raw,events = raw_and_events

    eeg_epos = []
    for band in params_acrossFrequencies['amp_bands']:
        rawband = raw.copy()
        rawband.filter(band[0], band[1], filter_length='auto',fir_design='firwin',
                    l_trans_bandwidth=1, h_trans_bandwidth=1,
                    method='fft', iir_params=None,
                    skip_by_annotation=['EDGE'], verbose=None)
        eeg_epo = Epochs(rawband,events,None,epoch_tmin,epoch_tmax,baseline=None,reject=None,proj='delayed',preload=True)
        eeg_epos.append(eeg_epo)
    return eeg_epos

@pytest.fixture(scope='module')
def eeg_src_and_fwd():
    src = mne.read_source_spaces(fname_src,patch_stats=True)
    fwd = mne.read_forward_solution(fname_fwd)

    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=False,
                                       copy=False, use_cps=True, verbose=None)

    return src,fwd


@pytest.fixture(scope='module')
def source_space_inputs(sensor_space_slow_epochs, sensor_space_hp_epochs, eeg_src_and_fwd):
    tempdir = _TempDir()
    src,fwd = eeg_src_and_fwd
    fpfx = 'eeg'

    newFs = sensor_space_slow_epochs.info['sfreq']
    l_freq,h_freq = params_acrossFrequencies['slow_band']

    mp = mne_pipeline.MNE_Pipeline(tempdir, 'test-fft', '', '', '', '', l_freq=l_freq, h_freq=h_freq, newFs=newFs, cov_freqthresh=65,
                                   blnDiagObsCov=False, blnOverwrite=True)

    cov = mp.compute_ObsNoiseCov(sensor_space_hp_epochs, op.join(tempdir,'eeg_epochs'))

    inv,fpfx_inv = mp.make_inverse_operator(sensor_space_slow_epochs.info, src, fwd, cov, fpfx,
                                            obsCov_suffix='', blnsave=False)

    return inv,src,mp

def get_src_space_data(epochs,level,inv,src,mp):
    data = [stc.data for stc in mp.get_inverse_sol_epochs_generator(epochs[level], inv, src, prepared=False)]
    return np.stack(data, axis=0).transpose((1, 0, 2))  # N x n_epochs x Ntimes

@pytest.fixture(scope='module')
def sensor_space_pac_acrossFrequencies(sensor_space_slow_epochs,sensor_space_epochs_acrossFreqs):
    tempdir = _TempDir()
    paf = params_acrossFrequencies

    pos = mne.find_layout(sensor_space_slow_epochs.info,exclude=[]).pos
    ch_names = sensor_space_slow_epochs.ch_names
    events = {'1':sensor_space_slow_epochs.events[:,0]/sensor_space_slow_epochs.info['sfreq']}
    eventnames = ['1']
    neighbor_dict = {name:[] for name in ch_names}

    pac = spac.SensorSpacePAC('test-sens-acrossFreqs', tempdir, None, paf['slow_band'],
                              paf['amp_bands'], paf['blnPhaseHilbert'], paf['blnAmpHilbert'], paf['Nbootstrap'], paf['swtPACmetric'],
                              paf['swtBootstrapMethod'], 'events.xxx',
                              tempdir, False,N=len(ch_names),pos=pos,prep=[],
                              ch_names=ch_names,events=events,
                              eventnames=eventnames,Nlevels=len(eventnames),
                              neighbor_dict=neighbor_dict)

    pac.phase_band_epochs = sensor_space_slow_epochs
    pac.amp_band_epochs = sensor_space_epochs_acrossFreqs

    pac.run_computation(False)
    return pac

@pytest.fixture(scope='module')
def source_space_pac_acrossFrequencies(sensor_space_slow_epochs,sensor_space_epochs_acrossFreqs,source_space_inputs):
    tempdir = _TempDir()
    paf = params_acrossFrequencies
    inv,src,mp = source_space_inputs

    def get_data_fun(level):
        phase_band_data = get_src_space_data(sensor_space_slow_epochs,level,inv,src,mp)
        amp_band_data = [get_src_space_data(epochs,level,inv,src,mp) for epochs in sensor_space_epochs_acrossFreqs]
        return phase_band_data, amp_band_data

    stc = next(mp.get_inverse_sol_epochs_generator(sensor_space_slow_epochs[0], inv, src, prepared=False))
    vertices = stc.vertices
    N = len(vertices[0]) + len(vertices[1])
    events = {'1':sensor_space_slow_epochs.events[:,0]/sensor_space_slow_epochs.info['sfreq']}
    eventnames = ['1']

    pac = spac.SourceSpacePAC('test-src-acrossFreqs', tempdir, None, paf['slow_band'],
                              paf['amp_bands'], paf['blnPhaseHilbert'], paf['blnAmpHilbert'], paf['Nbootstrap'], paf['swtPACmetric'],
                              paf['swtBootstrapMethod'], 'events.xxx',
                              tempdir, '', '',
                              '', '',
                              False, tempdir,N=N,vertices=vertices,
                              events=events,eventnames=eventnames,
                              epoch_tmin=epoch_tmin,epoch_tmax=epoch_tmax,
                              Nlevels=len(eventnames),
                              get_data_fun=get_data_fun)

    pac.run_computation(False)
    return pac


def test_spac_pipeline():
    #TODO
    return


def test_corr(sensor_space_slow_epochs,sensor_space_alpha_epochs):
    slow_data = sensor_space_slow_epochs._data.transpose((1,0,2)) # N x Nepochs x Ntimes
    alpha_data = sensor_space_alpha_epochs._data.transpose((1,0,2)) # N x Nepochs x Ntimes

    slow_data -= slow_data.mean(axis=2)[:,:,np.newaxis] # I don't do this step in the pipeline, because I don't want to mess with the zero crossings
    alpha_data -= alpha_data.mean(axis=2)[:,:,np.newaxis]

    data1 = spac.prepare_corr(alpha_data,slow_data)
    corr1 = spac.compute_corr(data1)

    data2 = spac.prepare_corr(alpha_data,slow_data,blnCrossProduct=True)
    corr2 = spac.compute_corr_crossProduct(data2)

    N = len(sensor_space_alpha_epochs.ch_names)
    corr3 = np.zeros(N)
    for i in range(N):
        corr3[i] = np.corrcoef(slow_data[i,:,:].flatten(),alpha_data[i,:,:].flatten())[0,1]

    assert(np.allclose(corr1,corr2,atol=0))
    assert(np.allclose(corr2,corr3,atol=0))


def test_mean_vector():
    #TODO
    return


def test_height_ratio():
    #TODO
    return


def test_frequency_profiles(sensor_space_pac_acrossFrequencies,source_space_pac_acrossFrequencies):
    sensor_space_pac_acrossFrequencies.compute_frequency_profiles(False)
    frequency_profile_helper(sensor_space_pac_acrossFrequencies.pac_all,
                             sensor_space_pac_acrossFrequencies.frequency_profiles)

    source_space_pac_acrossFrequencies.compute_frequency_profiles(False)
    frequency_profile_helper(source_space_pac_acrossFrequencies.pac_all,
                             source_space_pac_acrossFrequencies.frequency_profiles)

    freq_profiles_proj = source_space_pac_acrossFrequencies.project_pac_onto_frequency_profiles(sensor_space_pac_acrossFrequencies.frequency_profiles['U'])
    frequency_profile_helper(source_space_pac_acrossFrequencies.pac_all,
                             freq_profiles_proj)

    # Projecting onto your own frequency profiles should return the same S, VT, and V_unstacked
    freq_profiles_self = source_space_pac_acrossFrequencies.frequency_profiles
    freq_profiles_self_proj = source_space_pac_acrossFrequencies.project_pac_onto_frequency_profiles(freq_profiles_self['U'])
    assert(np.allclose(freq_profiles_self['U'],freq_profiles_self_proj['U'],atol=0,rtol=0))
    assert(np.allclose(freq_profiles_self['S'],freq_profiles_self_proj['S'],atol=0))
    assert(np.allclose(freq_profiles_self['VT'],freq_profiles_self_proj['VT'],atol=0))
    assert(np.allclose(freq_profiles_self['V_unstacked'],freq_profiles_self_proj['V_unstacked'],atol=0))

def frequency_profile_helper(P,freq_profiles):
    (Nf,Nlevels,N) = P.shape
    P_stacked = np.reshape(P,(Nf,N*Nlevels))

    U = freq_profiles['U']
    S = freq_profiles['S']
    VT = freq_profiles['VT']
    V_unstacked = freq_profiles['V_unstacked']
    assert(np.allclose(P_stacked, U.dot(np.diag(S)).dot(VT), atol=0))
    assert(np.allclose(U.T.dot(P_stacked), np.diag(S).dot(VT), atol=0))
    assert(np.allclose(np.nansum((VT ** 2), axis=1), np.ones(Nf), atol=0))
    assert(np.allclose(np.nansum(U.T.dot(P_stacked)**2,axis=1)**(1./2),S,atol=0))
    assert(np.all(np.equal(np.isnan(P),np.isnan(V_unstacked))))


