
import numpy as np
import logging
import mne
import warnings
import pandas as pd
from mne.externals.h5io import read_hdf5, write_hdf5
from src_loc_modules import utils


log = logging.getLogger(__name__)

class FreqDomain_BaseEpochs(mne.epochs.BaseEpochs):
    """Epochs corresponding to frequency domain data
    Don't use this constructor: use convert_epochs_td_to_fd or read_freqdomain_epochs

    Parameters
    ----------
    fd_epochs_data : np.ndarray (n_frequencies, n_channels, n_epochs * n_windows * n_tapers)
        frequency domain data
    info : instance of Info
        Info dictionary.
    fd_info : Dictionary
        movingwin : (window,winstep) The length of the moving window and step size in seconds (set them equal to have no overlap)
        NW : The time-half-bandwidth product
        Fs : Original sample rate
        f_axis : Frequency Axis, length = nfrequencies
        e_axis : Original epochs axis, length = n_epochs
        t_axis : Time axis within epochs (i.e. from the spectrogram) length = n_windows
        fft_data_shape : Shape of the windowed FFT, np.ndarray (n_frequencies, n_channels, n_epochs, n_windows, n_tapers)
    fd_events : None | array of int, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
        If None (default), all event values are set to 1 and event time-samples
        are set to range(n_epochs).
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to access associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    """

    def __init__(self, fd_epochs_data, info, fd_info, fd_events, event_id):
        f_axis = fd_info['f_axis']
        data = np.asanyarray(fd_epochs_data,dtype=np.complex128)
        info = info.copy()  # do not modify original info

        if data.ndim != 3:
            raise ValueError('Data must be a 3D array of shape (n_frequencies, '
                             'n_channels, n_samples)')
        if len(info['ch_names']) != data.shape[1]:
            raise ValueError('Info and data must have same number of '
                             'channels.')
        if data.shape[0] != len(fd_events):
            raise ValueError('The number of epochs and the number of events'
                             'must match')

        super(FreqDomain_BaseEpochs, self).__init__(info, data, fd_events, event_id, 0,
                                          data.shape[2]-1, baseline=None, reject=None,
                                          flat=None, reject_tmin=None,
                                          reject_tmax=None, decim=1,
                                          proj=False, on_missing='error')

        self._bad_dropped = True
        self.fd_info = fd_info
        self._inverse_event_id = {v:k for k,v in list(event_id.items())}


    def save(self, fname):

        log.info('Writing FreqDomain_Epochs to disk...')
        if fname[-4:]=='.fif':
            fname = fname[:-4] + '.h5'

        if fname[-6:-3]!='epo':
            warnings.warn('FreqDomain_Epochs filename should end in -epo.h5',RuntimeWarning)

        write_hdf5(fname,
                   dict(data=self.get_data(),
                        info=self.info, fd_info=self.fd_info,
                        events=self.events,event_id=self.event_id),
                   title='FreqDomain_Epochs',
                   overwrite=True)

        log.info('[done.]')


    @property
    def sample_axis_shape(self):
        # n_epochs, n_windows, n_tapers
        return self.fd_info['fft_data_shape'][2:]


    def _pick_drop_channels(self, idx):
        self.fd_info['fft_data_shape'][1] = len(idx)
        super(FreqDomain_BaseEpochs,self)._pick_drop_channels(idx)


    def convert_sample_index_td_to_fd(self,epoch_index,window_index,taper_index):
        return np.ravel_multi_index((epoch_index,window_index,taper_index), self.sample_axis_shape, order='C')


    def convert_sample_index_fd_to_td(self,sample_index):
        (epoch_i,win_i,taper_i) = np.unravel_index(sample_index, self.sample_axis_shape, order='C')
        return (epoch_i,win_i,taper_i)


    def convert_sample_index_to_time(self,sample_index):
        (epoch_i,win_i,taper_i) = self.convert_sample_index_fd_to_td(sample_index)
        time = self.fd_info['td_events'][epoch_i,0]/self.fd_info['Fs'] + self.fd_info['t_axis'][win_i]
        return time,taper_i


    def convert_time_to_sample_indices(self,times):
        all_times = self.convert_sample_index_to_time(self.times.astype(int))[0]
        return np.nonzero(np.in1d(all_times,times))[0]


    def get_frequency_for_epoch(self,epoch_index):
        return self._inverse_event_id[self.events[epoch_index,2]]


    def get_windowed_fft(self):
        """Convert a frequency domain Epochs object into a windowed fft

        Parameters
        ----------

        Returns
        -------
        windowed_fft : np.ndarray (n_frequencies, n_channels, n_epochs, n_windows, n_tapers)
            frequency domain data
        """
        return self.get_data().reshape(self.fd_info['fft_data_shape'],order='C')


    def get_spectrograms(self):
        """Convert a frequency domain Epochs object into an array of spectrgrams

        Parameters
        ----------

        Returns
        -------
        spectrogram : np.ndarray (n_channels, n_frequencies, n_times)
            frequency domain data
        t_axis :
        f_axis :
        """
        windowed_fft = self.get_windowed_fft() # (n_frequencies, n_channels, n_epochs, n_windows, n_tapers)
        (n_frequencies, n_channels, n_epochs, n_windows, n_tapers) = windowed_fft.shape

        spectrogram = np.real_if_close(np.mean(windowed_fft*windowed_fft.conj(),-1))*2/self.fd_info['Fs'] # (n_frequencies, n_channels, n_epochs, n_windows)
        spectrogram = spectrogram.reshape([n_frequencies,n_channels,n_epochs*n_windows],order='C') # (n_frequencies, n_channels, n_times)
        spectrogram = np.transpose(spectrogram,[1,0,2]) # (n_channels, n_frequencies, n_times)

        t_axis = np.unique(self.convert_sample_index_to_time(self.times.astype(int))[0])
        f_axis = self.fd_info['f_axis']

        return spectrogram,t_axis,f_axis


    def crop_by_sample_indices(self,sample_indices=None,inplace=False):
        # returns a new freq_domain_epochs object with only the samples indicated by indices
        if inplace:
            out = self
        else:
            out = self.copy()

        out.times = out.times[sample_indices]
        out._raw_times = out._raw_times[sample_indices]
        out._data = out._data[:, :, sample_indices]
        return out


    def __repr__(self):
        """Build string representation."""
        s = 'n_events (frequencies) : %s ' % len(self.events)
        s += '(all good)' if self._bad_dropped else '(good & bad)'
        s += ', n_samples (epochs,windows,tapers) : %s (%s,%s,%s)' % (len(self.times),self.sample_axis_shape[0],self.sample_axis_shape[1],self.sample_axis_shape[2])
        s += ', baseline : %s' % str(self.baseline)
        s += ', ~%s' % (mne.utils.sizeof_fmt(self._size),)
        s += ', data%s loaded' % ('' if self.preload else ' not')
        # class_name = self.__class__.__name__
        class_name = 'FreqDomain_Epochs'
        return '<%s  |  %s>' % (class_name, s)


    def to_data_frame(self, picks=None, index=None, scalings=None, copy=True):
        """Export data in tabular structure as a pandas DataFrame.

         Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.
        index : tuple of str | None
            Column(s) to be used as index for the data. Valid string options
            are 'epoch', 'window', 'taper', 'sample_index', 'frequency' and 'condition'. If None, all six info
            columns will be included in the table as categorial data. If empty, will use the pandas index
            (0 to number of records)
        scalings : dict | None
            Scaling to be applied to the channels picked. If None, defaults to
            ``scalings=dict(eeg=1e6, grad=1e13, mag=1e15, misc=1.0)``.
        copy : bool
            If true, data will be copied. Else data may be modified in place.

        Returns
        -------
        df : instance of pandas.core.DataFrame
            A dataframe suitable for usage with other
            statistical/plotting/analysis packages. Column/Index values will
            depend on the object type being converted, but should be
            human-readable.
        """

        default_index = ['condition','epoch','window','taper','frequency']

        if index is not None:
            mne.io.base._check_pandas_index_arguments(index, default_index)
        else:
            index = default_index

        # use the existing functionality to convert to dataframe
        td_index = ['condition','epoch','time']
        if scalings is None:
            scalings = dict(eeg=1.0,grad=1.0,mag=1.0,misc=1.0)
        df = super(FreqDomain_BaseEpochs,self).to_data_frame(picks=picks,index=td_index,scaling_time=1,
                                                             scalings=scalings,copy=copy)
        channel_columns = df.columns.values

        # Correct column names
        df = df.reset_index()
        del df['epoch']
        df.rename(columns={'condition':'frequency','time':'fd_sample_i'},inplace=True)
        df['frequency'] = df['frequency'].astype(float)

        # add columns for epoch, window, taper, time, and condition
        df_index = self.get_sample_index()

        df = pd.merge(df,df_index,how='left',on='fd_sample_i',sort=False,copy=False)

        # reorder columns
        new_columns = ['epoch','window','taper','frequency','condition','fd_sample_i','time'] + list(channel_columns)
        df = df[new_columns]
        df.columns = df.columns.astype(str)

        # set index
        if index:
            df.set_index(index,inplace=True)

        return df


    def get_sample_index(self):
        # dataframe with the (epoch, window, taper, time, condition) for each sample (fd_sample_i)
        event_ids = self.fd_info['td_event_id']
        td_events = self.fd_info['td_events']
        sorted_levels = sorted(list(event_ids.keys()),key=event_ids.get)
        inverse_level_map = {v:k for k,v in list(event_ids.items())}

        fd_sample_i = self.times.astype(int)
        (td_epoch_i,win_i,taper_i) = self.convert_sample_index_fd_to_td(fd_sample_i)
        df_index = pd.DataFrame(np.c_[fd_sample_i,td_epoch_i,win_i,taper_i],
                                columns=['fd_sample_i','epoch','window','taper'],
                                dtype=int)
        condition = [str(inverse_level_map[td_events[i,2]]) for i in td_epoch_i]
        df_index.insert(4,'condition',pd.Categorical(condition,categories=[str(s) for s in sorted_levels],ordered=True))
        time = self.convert_sample_index_to_time(fd_sample_i)[0]
        df_index.insert(5,'time',time)

        return df_index


class FreqDomain_EpochsH5(FreqDomain_BaseEpochs):
    def __init__(self,fname,preload=True,verbose=None):
        log.info('Reading FreqDomain_Epochs from disk...')
        if fname[-4:]=='.fif':
            fname = fname[:-4] + '.h5'

        read_dict = read_hdf5(fname, title='FreqDomain_Epochs')

        log.info('   Constructing the object...')
        super(FreqDomain_EpochsH5, self).__init__(read_dict['data'], mne.Info(read_dict['info']), read_dict['fd_info'], read_dict['events'], read_dict['event_id'])

        log.info('[done.]')


class FreqDomain_EpochsFFT(FreqDomain_BaseEpochs):
    """Epochs corresponding to frequency domain data

    Parameters
    ----------
    windowed_fft : np.ndarray (n_frequencies, n_channels, n_epochs, n_windows, n_tapers)
        frequency domain data
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    events : None | array of int, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
        If None (default), all event values are set to 1 and event time-samples
        are set to range(n_epochs).
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to access associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    movingwin : (window,winstep) The length of the moving window and step size in seconds
    NW : The time-half-bandwidth product
    f_axis : ndarray, (n_frequencies,)
        The frequency axis for the windowed fft (must be equally spaced frequencies)
    t_axis : ndarray, (n_windows,)
        The time axis for the windowed fft
    """

    def __init__(self, windowed_fft, info, events, event_id, movingwin, NW, f_axis, t_axis):

        fft_data_shape = windowed_fft.shape
        fd_epochs_data_shape = (fft_data_shape[0],fft_data_shape[1],np.prod(fft_data_shape[2:]))
        fd_epochs_data = windowed_fft.reshape(fd_epochs_data_shape, order='C')

        # Events will be the frequency axis
        event_times = list(range(len(f_axis)))
        event_ids = list(range(len(f_axis)))
        fd_events = np.c_[event_times,
                            np.zeros(len(f_axis)),
                            event_ids].astype(int)
        fd_event_id = {str(f):i for f,i in zip(f_axis,event_ids)}

        Fs = info['sfreq']
        info = info.copy()  # do not modify original info
        info['sfreq'] = 1

        fd_info = dict(movingwin=movingwin,NW=NW,Fs=Fs,
                       td_events=events.copy(),td_event_id=event_id.copy(),
                       f_axis=f_axis.copy(),t_axis=t_axis.copy(),
                       fft_data_shape=list(fft_data_shape))

        super(FreqDomain_EpochsFFT, self).__init__(fd_epochs_data, info, fd_info, fd_events, fd_event_id)


def read_freqdomain_epochs(fname, preload=True, verbose=None):
    return FreqDomain_EpochsH5(fname,preload=preload,verbose=verbose)

def convert_epochs_td_to_fd(td_epochs, movingwin, NW, K=None, NFFT=None, proj=False, f_targets=None):
    """Convert a time domain Epochs object into a frequency domain Epochs object

    Parameters
    ----------
    td_epochs : mne.epochs.BaseEpochs
        time domain epochs
    movingwin : (window,winstep) The length of the moving window and step size in seconds (set them equal to have no overlap)
        Default is (1,1), nonoverlapping 1s windows
    NW : The time-half-bandwidth product
        Default is 2, which leads to a frequency resolution of 4Hz for 1s windows
    K : The number of tapers to use
        Default is None, which uses all tapers with eigenvalues > 0.9
    NFFT : The number of FFT bins to compute (default: the number of samples in each window)
    proj : Boolean
        Whether to add and apply a common average reference to the data before the fft
    f_targets: np.ndarray or None
        the frequencies to keep (default: all, both positive and negative)

    Returns
    -------
    fd_epochs : FreqDomain_BaseEpochs
        frequency domain epochs
    """
    log.info('Converting time domain epochs to frequency domain...')
    if proj:
        # set and apply CAR
        mne.io.set_eeg_reference(td_epochs,ref_channels='average',copy=False,projection=True)
        td_epochs.apply_proj()

    td_data = td_epochs.get_data()
    events = td_epochs.events.copy()
    event_id = td_epochs.event_id
    info = td_epochs.info
    times = td_epochs.times

    Fs = info['sfreq']

    fd_data = []
    for i,epoch in enumerate(td_data):
        log.debug('Computing windowed fft for epoch #' + str(i))
        fdi, f_axis, t_axis, _ = utils.compute_windowed_fft(epoch, Fs, times, movingwin, NW, K=K, NFFT=NFFT, f_targets=f_targets) # n_channels x n_tapers x n_freqs x n_windows
        fd_data.append(fdi[np.newaxis,:,:,:,:])
    fd_data = np.vstack(fd_data) # n_epochs x n_channels x n_tapers x n_freqs x n_windows
    fd_data = np.transpose(fd_data,[3,1,0,4,2])# (n_freqs, n_channels, n_epochs, n_windows, n_tapers)
    t_axis += td_epochs.tmin
    return FreqDomain_EpochsFFT(fd_data, info, events, event_id, movingwin, NW, f_axis, t_axis)


