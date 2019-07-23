import os.path as op
import mne
import numpy as np
from scipy.io import loadmat
import logging
import warnings
import subprocess
from time import strftime
from nibabel import freesurfer as fs

log = logging.getLogger(__name__)


def load_bad_channels(badchannels, subject, strpfx, fill_length,
                      blnAddPfx=False):
    if type(badchannels) == str:
        # load bad channels
        badchannelsmat = loadmat(badchannels)
        bc = badchannelsmat['badchannelmat']
        subnum = int(''.join([s for s in subject if s.isdigit()]))
        bads = bc[bc[:, 0] == subnum, 1]  # find bad channels for this subject
    elif type(badchannels) == list:
        bads = badchannels
    else:
        raise RuntimeError('badchannels should be a path string or a list')

    if blnAddPfx:
        bads = [strpfx + str(i).zfill(fill_length) for i in
                bads]  # map to channel name
    return bads


def collect_bad_channels(badchannels_eeg, badchannels_meg, subject,
                         blnAddPfx=False):
    # Add bad channels to raw
    bads = []
    if badchannels_eeg is not None:
        bln_eeg = True
        bads += load_bad_channels(badchannels_eeg, subject, 'EEG', 3,
                                  blnAddPfx=blnAddPfx)
    else:
        bln_eeg = False

    if badchannels_meg is not None:
        bln_meg = True
        bads += load_bad_channels(badchannels_meg, subject, 'MEG', 4,
                                  blnAddPfx=blnAddPfx)
    else:
        bln_meg = False

    return bads, bln_eeg, bln_meg


def initialize_log_file(output_path, fpattern):
    # Set new file handler
    fname = op.join(output_path, fpattern.format(strftime('%Y%m%d-%H%M%S')))
    lh = logging.FileHandler(fname, mode='a')
    lh.setFormatter(
        logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
    logging.root.addHandler(lh)
    logging.getLogger('mne').addHandler(lh)

    log.info('Git Hash: ' + subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                                             stdout=subprocess.PIPE,
                                             universal_newlines=True).communicate()[0])
    log.info('MNE Version: ' + mne.__version__)
    return lh


def reset_logging_to_file(lh=None):
    if lh is None:
        # remove any previous file handlers
        logging.root.handlers = [h for h in logging.root.handlers if
                                 not isinstance(h, logging.FileHandler)]
        logging.getLogger('mne').handlers = [h for h in
                                             logging.getLogger('mne').handlers
                                             if not isinstance(h,
                                                               logging.FileHandler)]
    else:
        logging.root.removeHandler(lh)
        logging.getLogger('mne').removeHandler(lh)


def load_eventfile(eventfile):
    eventinfo = loadmat(eventfile, squeeze_me=False)
    try:
        events = {e['name'][0]: e['times'].ravel() for e in
                  eventinfo['events'].ravel()}
        eventnames = [e['name'][0] for e in eventinfo['events'].ravel()]
    except:
        events = {e['name'][0][0][0]: e['times'][0][0].ravel() for e in
                  eventinfo['events'].ravel()}
        eventnames = [e['name'][0][0][0] for e in eventinfo['events'].ravel()]

    cov_tmin = eventinfo['cov_win'][0][0]
    cov_tmax = eventinfo['cov_win'][0][1]

    return events, eventnames, cov_tmin, cov_tmax


def make_mne_events_from_events(events, eventnames, raw):
    event_id = {name: i + 1 for i, name in enumerate(eventnames)}
    event_list = []
    for event_type in eventnames:
        evs = events[event_type]
        if len(evs) > 0:
            ts = raw.time_as_index(events[event_type]) + raw.first_samp
            n_events = len(ts)
            id = event_id[event_type]
            event_list.append(np.c_[ts, np.zeros(n_events, dtype=int),
                                    id * np.ones(n_events, dtype=int)])
        else:
            del (event_id[event_type])
    mne_events = np.concatenate(event_list, axis=0)

    # sort by time
    mne_events = mne_events[mne_events[:, 0].argsort()]

    return mne_events, event_id


def find_label_mask(lbl, vertices):
    # lbl: mne.Label
    # vertices: tuple of lists, (left hem vertices, right hem vertices)
    # returns: indices in vertices that correspond to the label

    if lbl.hemi=='lh':
        hi = 0
    else:
        hi = 1

    src_sel = np.intersect1d(vertices[hi], lbl.vertices)
    src_sel = np.searchsorted(vertices[hi], src_sel)

    if hi==1:
        src_sel += len(vertices[0])

    return src_sel


def get_laplacian_referenced_data_epochs(epochs, neighbor_dict):
    # %LAPLACIAN FUNCTION Takes the average of the neighbor electrodes and
    # %subtracts it from the single electrode
    # based on purdongp/code/matlab/eeg_analysis/laplacian.m

    data = epochs.get_data() # n_epochs x n_channels x time
    ch_names = epochs.ch_names

    assert(all([ch in list(neighbor_dict.keys()) for ch in ch_names]))

    # %Pre-allocate laplacdata the size of eegdata
    # laplacdata=eegdata;
    laplacdata = np.zeros(data.shape)

    # %Find the number of channels
    # numchans=length(eegdata(:,1));
    numchans = data.shape[1]
    assert(len(ch_names)==numchans)

    # %Loop through each channel and compute laplacian
    # for channel=1:numchans
    for ep_i,ep in enumerate(data):
        log.debug('Epoch #' + str(ep_i))
        for channel_i,(channel_name,mainchannel) in enumerate(zip(ch_names,ep)):
            # mainchannel=eegdata(channel,:);
            #
            # %Check if any neighbors at all
            # if isempty(neighbors)
            if len(neighbor_dict)==0:
                # laplacdata(channel,:)=mainchannel;
                laplacdata[ep_i,channel_i,:] = mainchannel
            else:
                # %Get the laplacian of the current channel
                # if ~isempty(neighbors{channel})
                neighbor_indices = [i for i,n in enumerate(ch_names) if n in neighbor_dict[channel_name]]
                if ep_i==0:
                    log.debug('  Neighbors for {}: {}'.format(channel_name,[ch_names[i] for i in neighbor_indices]))
                    log.debug('      (Original {}: {})'.format(channel_name,neighbor_dict[channel_name]))

                if len(neighbor_indices)<3:
                    warnings.warn('Channel {} only has {} neighbors ({})'\
                                  '\n  (Original: {})'.format(channel_name,
                                                              len(neighbor_indices),
                                                              [ch_names[i] for i in neighbor_indices],
                                                              neighbor_dict[channel_name]),
                                  RuntimeWarning)
                if len(neighbor_indices)>0:
                    # %Set nans in neighbors to be zero
                    # neighbordata=nanmean(eegdata(neighbors{channel},:),1);
                    # neighbordata(isnan(neighbordata))=0;
                    neighbordata = np.nanmean(ep[neighbor_indices,:],axis=0)
                    neighbordata[np.isnan(neighbordata)] = 0
                    #
                    # %Compute the laplacian
                    # laplacdata(channel,:)=mainchannel-neighbordata;
                    laplacdata[ep_i,channel_i,:] = mainchannel - neighbordata
                else:
                    # %Don't change if there are no neighbors for that channel
                    # laplacdata(channel,:)=mainchannel;
                    laplacdata[ep_i,channel_i,:] = mainchannel

    return laplacdata


def load_neighbors_mat(neighbor_mat_file):
    from scipy.io import loadmat

    neigh = loadmat(neighbor_mat_file,variable_names='channelneighbors', squeeze_me=True)
    neighbor_mat = np.array(neigh['channelneighbors'])
    N,M = neighbor_mat.shape
    assert(N==M)

    def ch_name(i):
        return 'EEG' + str(int(i)).zfill(3)

    neighbor_dict = {}
    for i,row in enumerate(neighbor_mat):
        neighbor_dict[ch_name(i+1)] = [ch_name(j+1) for j,n in enumerate(row) if n>0]

    return neighbor_mat,neighbor_dict


def plot_topos(axes,toplot,clims,pos,reject=None,contours=[0],cmap='seismic',
               head_pos={'center':[0.5,0.5],'scale':[1,1]}):
    # axes: Nlevels list of axes for the topos
    # toplot: Nlevels x Nchannels array of intensity values
    # reject: Nlevels x Nchannels array of significance bools (or None, will not highlight any electrodes)

    for j, (row, axis) in enumerate(zip(toplot,axes)):
        valid_inds = np.logical_not(np.isnan(row))
        if reject is None:
            im, cn = mne.viz.plot_topomap(row[valid_inds], pos[valid_inds,:],
                                          cmap=cmap, vmin=clims[0],
                                          vmax=clims[1], contours=contours,axes=axis,
                                          head_pos=head_pos)
        else:
            sigmask = reject[j,:]
            mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                            linewidth=0, markersize=2)
            im, cn = mne.viz.plot_topomap(row[valid_inds], pos[valid_inds,:],
                                          cmap=cmap, vmin=clims[0],
                                          vmax=clims[1], contours=contours,
                                          mask=sigmask[valid_inds],
                                          mask_params=mask_params,axes=axis,
                                          head_pos=head_pos)

    return im


def plot_surf(stc,clim,colormap,time_label,surf,transparent,offscreen,subjects_dir=None,
               blnMaskUnknown=False):
    if offscreen:
        # create offscreen figure so that I can use the computer while it's saving
        from surfer.viz import _make_viewer
        figure = _make_viewer(None, 1, 2, stc.subject, (800,1600), True)[0][0]
    else:
        figure=None

    # This raises a vtk error that has something to do with smoothing_steps=None
    # (any time smoothing steps is big enough to cover all of the vertices)
    # but it still displays the correct figure.
    # can't catch the error since it's in c, printing to console.
    brain = stc.plot(surface=surf, hemi='split', views='medial',
                       clim=clim, colormap=colormap, transparent=transparent,time_unit='s',
                       time_label=time_label,size=[1600,800],figure=figure,smoothing_steps=None,
                     subjects_dir=subjects_dir)

    if blnMaskUnknown:
        subjects_dir = mne.utils.get_subjects_dir(subjects_dir=subjects_dir,
                                        raise_error=True)

        for hemi in ['lh','rh']:
            aparc_file = op.join(subjects_dir,stc.subject,"label",'{}.aparc.annot'.format(hemi))
            labels,_,names = fs.read_annot(aparc_file)
            masked_region_inds = np.arange(len(names))[np.in1d(names,['corpuscallosum','unknown'])]
            masked_region_inds = np.append(masked_region_inds,-1) # unlabeled
            mask = np.in1d(labels,masked_region_inds)

            brain.add_data(mask,hemi=hemi,min=0,max=1,thresh=0.5,colormap='gray',colorbar=False,alpha=0.99)

    return brain


def save_surf(brain,saveflag,outputpath,suffix):
    fname = op.join(outputpath, saveflag + '_medial' + suffix)
    brain.show_view('medial',row=0,col=1)
    brain.show_view('medial',row=0,col=0)
    brain.save_image(fname + '.png')

    brain.show_view('lateral',row=0,col=1)
    brain.show_view('lateral',row=0,col=0)

    fname = op.join(outputpath, saveflag + '_lateral' + suffix)
    brain.save_image(fname + '.png')


def remove_box(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_gram(axis,toplot,yl,cmap,tax,fax,bandwidth):
    axis.cla()
    axis.imshow(toplot,aspect='auto',interpolation='none',
                origin='lower',cmap=cmap,clim=yl,
                extent=[tax[0],tax[-1],
                       fax[0]-bandwidth/2,fax[-1]+bandwidth/2])


def add_colorbar(fig,cax,im):
    cax.cla()
    cpos = cax.get_position()
    x0,x1,y0,y1 = cpos.x0,cpos.x1,cpos.y0,cpos.y1
    cpos.x0 = x0 #(x1+x0)/2 -(x1-x0)/2
    cpos.x1 = x1 # (x1+x0)/2
    cpos.y0 = (y1+y0)/2 #-(y1-y0)/3
    cpos.y1 = (y1+y0)/2 +1.5*(y1-y0)/3

    cax.set_position(cpos)
    cbar = fig.colorbar(im, ax=cax, cax=cax, orientation='horizontal')

    return cbar
