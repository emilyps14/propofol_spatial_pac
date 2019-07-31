# Author: Emily P. Stephen <emilyps14@gmail.com>

from spatial_pac_utils.spatial_phase_amplitude_coupling \
    import pac_savename,read_SensorSpacePAC,prepare_roi_summary
from spatial_pac_utils.utils \
    import add_colorbar,plot_gram,remove_box,plot_topos,plot_surf,save_surf
import os.path as op
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec
import mne
import numpy as np


default_frontal_sel = ['EEG002','EEG006','EEG037','EEG005','EEG036']
default_posterior_sel = ['EEG028','EEG058','EEG057','EEG027','EEG022','EEG052']

def pac_summary_setup(figures_output_path,subject,phase_band,
                      swtPACmetric,blnPhaseHilbert,
                      frontal_sel=default_frontal_sel,
                      posterior_sel=default_posterior_sel):

    # Load PAC
    fname_session_comod = pac_savename(subject, None, blnPhaseHilbert, True, phase_band, [], swtPACmetric, None)
    session_pac_comod = read_SensorSpacePAC(figures_output_path,fname_session_comod)

    ch_names = session_pac_comod.ch_names
    session_timeaxis = session_pac_comod.session_timeaxis
    center_freqs = session_pac_comod.get_center_freqs()
    pos = session_pac_comod.pos


    ### Set up Variables
    pac_all, _ = session_pac_comod.compute_PAC_across_electrode_sets([frontal_sel,posterior_sel])

    pacgram_frontal = pac_all[:, :,0]
    pacgram_posterior = pac_all[:, :, 1]

    return pacgram_frontal,pacgram_posterior,session_timeaxis,center_freqs,pos,ch_names,fname_session_comod


def plot_electrode_positions(pos,ch_names,frontal_sel=default_frontal_sel,
                             posterior_sel=default_posterior_sel,
                             head_pos={'center':[0.5,0.5],'scale':[1,1]}):
    ### Figures indicating the frontal and posterior electrodes
    fig_electrodes = plt.figure(3, figsize=[5, 3])
    fig_electrodes.clf()
    fig_electrodes.patch.set_facecolor('w')

    gs_electrodes = gridspec.GridSpec(1, 2)
    mask_params = dict(marker='x', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=5)

    for i,(sel,title) in enumerate(zip([frontal_sel,posterior_sel],['Frontal','Posterior'])):
        ax = fig_electrodes.add_subplot(gs_electrodes[0,i])
        ax.cla()
        plot_electrodes(pos,ax,ch_names,sel,mask_params=mask_params,head_pos=head_pos)
        ax.set_title(title)

    return fig_electrodes

def plot_electrodes(pos,ax,ch_names,sel,mask_params=None,
                    head_pos={'center':[0.5,0.5],'scale':[1,1]}):
    N = len(pos)
    mask = np.array([ch_names[i] in sel for i in range(N)])

    im, cn = mne.viz.plot_topomap(np.zeros(N), pos, mask=mask,
                                  mask_params=mask_params,
                                  head_pos=head_pos)


def plot_single_subject_summary(subjects_dir, subject,
                                propofol_t,propofol_dose,key_times,
                                pr_resp_T,pr_resp_verbal,pr_resp_clicks,
                                savepath=None,
                                blnSigElectrodes=True,
                                phase_band=(0.1,4),amp_band=(8,16),
                                bandwidth=2.,event_filename='event_times_level.mat',
                                swtPACmetric='corr',
                                blnAmpHilbert=True,
                                swtBootstrapMethod='permtest_epochs_wi_level',
                                swtMultipleComparisons = 'FDR',
                                stats_alpha = 0.05,
                                frontal_sel=default_frontal_sel,
                                posterior_sel=default_posterior_sel):


    blnPhaseHilbert = swtPACmetric=='mean_vector'
    t_lims = pr_resp_T[[0,-1]]
    pac_ylim = [8,None]
    pac_yticks = np.arange(10,50,10)

    levels_path = op.join(subjects_dir, subject,
                           'Figures sensor space',
                           'PAC_{}-{}_{}-{}'.format(phase_band[0],phase_band[1],amp_band[0],amp_band[1]),
                           swtPACmetric)

    fname_levels = pac_savename(subject,event_filename,blnPhaseHilbert,blnAmpHilbert,phase_band,[amp_band],swtPACmetric,swtBootstrapMethod)
    levels_pac = read_SensorSpacePAC(levels_path,fname_levels)

    session_path = op.join(subjects_dir, subject,
                           'Figures sensor space',
                           'PAC_session_{}-{}_{}Hz'.format(phase_band[0], phase_band[1], bandwidth),
                           swtPACmetric)
    pacgram_frontal,pacgram_posterior,session_timeaxis,center_freqs,pos,\
    ch_names,fname_session_comod \
        =pac_summary_setup(session_path, subject, phase_band,
                           swtPACmetric, blnPhaseHilbert,
                           frontal_sel=frontal_sel,
                           posterior_sel=posterior_sel)

    all_data = np.concatenate((pacgram_frontal.flatten(), pacgram_posterior.flatten(), levels_pac.pac_all.flatten()))
    yl_diff = np.array([-1,1])*np.percentile(np.abs(all_data), 99)

    ### Set up figure
    fig = plt.figure(2, figsize=[9, 5])
    fig.clf()
    fig.patch.set_facecolor('w')

    g_kwargs = {'left': 0.05, 'right': 0.98, 'bottom': 0.02, 'top': 0.98, 'hspace':0.1, 'wspace': 0.01}
    gs = gridspec.GridSpec(6, 10, height_ratios=[1.25,1.25,2,2,3,0.75], width_ratios=np.ones(10), **g_kwargs)

    ### Propofol level
    axis_propofol = fig.add_subplot(gs[0, :])

    axis_propofol.cla()
    axis_propofol.plot(propofol_t,propofol_dose,'k')
    for time in key_times:
        axis_propofol.axvline(time,color='k',linestyle='-',linewidth=2)
    axis_propofol.set_xticklabels('')
    axis_propofol.set_yticks(np.arange(1,propofol_dose.max()+1,2))
    axis_propofol.set_xlim(t_lims)

    remove_box(axis_propofol)

    ### Pr(resp)
    axis_bhvr = fig.add_subplot(gs[1,:])

    axis_bhvr.cla()
    axis_bhvr.plot(pr_resp_T,pr_resp_verbal[:,1],'m',linewidth=2)
    axis_bhvr.plot(pr_resp_T,pr_resp_clicks[:,1],'c',linewidth=2)
    axis_bhvr.axhline(1,color='k',linestyle='--',linewidth=1)
    axis_bhvr.fill_between(pr_resp_T,pr_resp_verbal[:,0],pr_resp_verbal[:,2],facecolor='m',alpha=0.25)
    axis_bhvr.fill_between(pr_resp_T,pr_resp_clicks[:,0],pr_resp_clicks[:,2],facecolor='c',alpha=0.25)
    for time in key_times:
        axis_bhvr.axvline(time,color='k',linestyle='-',linewidth=2)
    axis_bhvr.set_xticklabels('')
    axis_bhvr.set_yticks([0,1])
    axis_bhvr.set_ylim([0,1.1])
    axis_bhvr.set_xlim(t_lims)
    axis_bhvr.legend(('Verbal','Clicks'),fontsize=8,loc='lower left')

    remove_box(axis_bhvr)

    ### Topoplots
    Nlevels = levels_pac.Nlevels
    pac_all_levels = levels_pac.pac_all[0,:,:]

    if blnSigElectrodes:
        reject = levels_pac.compute_rejection(stats_alpha,swtMultipleComparisons)
    else:
        reject = None

    axes = [fig.add_subplot(gs[4, j]) for j in range(Nlevels)]
    im = plot_topos(axes,pac_all_levels,yl_diff,pos,reject=reject)

    ### Frontal Average
    axis_frontal = fig.add_subplot(gs[2, :])

    plot_gram(axis_frontal, pacgram_frontal, yl_diff, 'seismic', session_timeaxis, center_freqs, bandwidth)
    axis_frontal.set_xticklabels('')
    axis_frontal.set_ylim(pac_ylim)
    axis_frontal.set_yticks(pac_yticks)
    axis_frontal.set_xlim(t_lims)

    ### Posterior Average
    axis_posterior = fig.add_subplot(gs[3, :])

    plot_gram(axis_posterior, pacgram_posterior, yl_diff, 'seismic', session_timeaxis, center_freqs, bandwidth)
    axis_posterior.set_ylim(pac_ylim)
    axis_posterior.set_yticks(pac_yticks)
    axis_posterior.set_xlim(t_lims)

    ### Colorbar
    cax = fig.add_subplot(gs[5, 2:8])
    add_colorbar(fig,cax,im)

    fig.canvas.draw()

    ### Figures indicating the frontal and posterior electrodes
    fig_electrodes = plot_electrode_positions(pos, ch_names,
                                              frontal_sel=frontal_sel,
                                              posterior_sel=posterior_sel)


    ### Save
    if savepath is not None:
        filepfx = op.join(savepath,fname_session_comod)
        if blnSigElectrodes:
            summary_fname = filepfx + '_summary_wsigelectrodes.png'
        else:
            summary_fname = filepfx + '_summary.png'

        fig.savefig(summary_fname, format='png')
        fig_electrodes.savefig(filepfx + '_front_post_electrodes.png',
                               format='png')


def plot_frequency_profiles(freq_profiles,freqs,savepath=None):
    '''
    Plot the first three frequency profiles and the percent of the total
    energy captured in each

    See Figure 2 in Stephen et al 2019

    Parameters
    ----------
    freq_profiles : dict
        The frequency profiles (output from spatial_phase_amplitude_coupling.compute_frequency_profiles
        Having at least items:
        'S' -> ndarray, (Nf,), singular values
        'U' -> ndarray, (Nf,Nf), frequency profiles (in columns)
    freqs : ndarray, (Nf,)
        The frequency axis (centers)
    savepath : basestring | None
        The path to save the figure (or None for no save)

    Returns
    -------

    '''
    S = freq_profiles['S']
    pct_energy = S**2/(S**2).sum() * 100
    U = freq_profiles['U']
    nprofiles = 3
    ylim = [-0.5,0.5]
    Nfreqs = len(freqs)

    colors = ['midnightblue','teal','darkmagenta']

    fig2 = plt.figure(8,figsize=[7, 3.2])
    fig2.clf()
    fig2.patch.set_facecolor('w')

    # g_kwargs = {'left': 0.05, 'right': 0.98, 'bottom': 0.02, 'top': 0.95, 'hspace':0.1, 'wspace': 0.01}
    g_kwargs = {'bottom': 0.2,'hspace':0.2}
    gs = gridspec.GridSpec(1, 2, **g_kwargs)

    ax = fig2.add_subplot(gs[0])
    ax.set_prop_cycle('color',colors)
    ax.plot(freqs,U[:,:nprofiles])
    ax.axhline(0,color='k')
    ax.set_ylim(ylim)
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('Correlation')
    ax.legend([str(i+1) for i in range(nprofiles)],labelspacing=0.1,fontsize='small',loc='lower right')
    ax.set_title('Modes')

    ax = fig2.add_subplot(gs[1])
    ax.plot(np.arange(Nfreqs)+1,pct_energy,'k.')
    ax.set_xlabel('Mode #')
    ax.set_title('% Total Energy')

    ### Save
    if savepath is not None:
        fname = op.join(savepath,'profilesFig_modes')
        fig2.savefig(fname+'.png',format='png')


def plot_projection_surfaces(projections, eventnames, vertices,
                             subject,subjects_dir,
                             savename, Nprojs=3,
                             blnOffscreen=False,savepath=None):
    '''
    Plot the source space projections of the frequency profiles

    See Figure 3 in Stephen et al 2019

    :param projections: ndarray, (components,levels,sources)
        The projections to plot, e.g. from spatial_phase_amplitude_coupling.compute_source_space_projections
    :param eventnames: list of basestring
        List of names of the levels
    :param vertices: list of numeric
        the vertices corresponding to the sources in projections
    :param subject: basestr
        The subject ID
    :param subjects_dir:
        The subject directory
    :param savename:
        The name of the file to save to
    :param Nprojs: int
        The number of projections to plot
    :param blnOffscreen: bool
        Whether to plot the surfaces offscreen
    :param savepath: basestr | None
        The path to save to (or None for no save)
    :return:
    '''
    def label_func(f):
        return eventnames[int(f)]

    for proji in range(Nprojs):
        toplot = projections[proji, :, :]
        toplot[np.isnan(toplot)] = 0

        yl_diff = np.array([-1,1])*np.percentile(np.abs(toplot[~np.isnan(toplot)]), 100)
        ctrl_pts = [yl_diff[0], 0, yl_diff[1]]
        stc_proj = mne.SourceEstimate(toplot.T,
                                      vertices=vertices,
                                      tmin=0, tstep=1,
                                      subject=subject)
        brain = plot_surf(stc_proj, {'kind': 'value', 'lims': ctrl_pts},
                                       'seismic', label_func,
                                       'semi-inflated', False,
                                       blnOffscreen,subjects_dir=subjects_dir,
                                       blnMaskUnknown=True)

        cmap = plt.get_cmap('seismic')
        norm = Normalize(vmin=ctrl_pts[0],vmax=ctrl_pts[-1])
        scalarMap = ScalarMappable(norm=norm,cmap=cmap)
        scalarMap.set_array(np.linspace(ctrl_pts[0],ctrl_pts[-1],100))

        if savepath is not None:
            fname = savename + '_FreqProfs_surfProj{}'.format(proji+1)
            for j,(t,level) in enumerate(zip(stc_proj.times,eventnames)):
                brain.set_time(t)
                save_surf(brain, fname, savepath,
                                   '_{}{}'.format(j, level))
            brain.close()


def plot_roi_summary(roi_pac_all, reject, xaxis, marker_size,eventnames,
                     lobe_bounds,lobe_names,ordered_rois,
                     yl_factor=1,ylim=None):
    # roi_pac_all:  (1 x Nlevels x Nrois) or ( Nsubjects x Nlevels x Nrois)
    # reject: (1 x Nlevels x Nrois) or ( Nsubjects x Nlevels x Nrois)
    # xaxis: (1 x Nrois) or (Nsubjects x Nrois)

    Nlevels = len(eventnames)

    if ylim is None:
        ylim = np.nanmax(np.abs(roi_pac_all)) * 1.1 * np.array([-1, 1])
    cmap = plt.get_cmap('seismic')
    norm = Normalize(vmin=ylim[0],vmax=ylim[1])
    scalarMap = ScalarMappable(norm=norm,cmap=cmap)

    fig = plt.figure()
    fig.clf()
    fig.set_size_inches([17,9.5])
    gs = gridspec.GridSpec(Nlevels+1, 1,height_ratios=[yl_factor]*(Nlevels-1)+[1,0.5],hspace=0)
    axs = []
    for level_i,level_name in enumerate(eventnames):
        ax = fig.add_subplot(gs[level_i])
        toplot_level = roi_pac_all[:, level_i, :]
        mask_level = reject[:, level_i, :]

        ax.scatter(xaxis[mask_level], toplot_level[mask_level], s=marker_size, c=toplot_level[mask_level], cmap=cmap, norm=norm, marker='o',edgecolors='k',zorder=3)
        ax.scatter(xaxis[np.logical_not(mask_level)], toplot_level[np.logical_not(mask_level)], s=marker_size, c='k', cmap=cmap, norm=norm, marker='+',zorder=3)
        ax.axhline(0,color='k')
        for lobe,next in zip(lobe_bounds[:-1],lobe_bounds[1:]):
            ax.axvline((lobe[1]+next[0])/2,color='k')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks(xaxis[0,:])
        if level_i==Nlevels-1:
            ax.set_xticklabels(ordered_rois,rotation='vertical')
        else:
            ax.set_xticklabels('')
        hdiff = (xaxis[0,1]-xaxis[0,0])
        ax.set_xlim([xaxis[0,0]-hdiff,xaxis[0,-1]+hdiff])

        ax.set_ylabel(level_name)
        if level_i<Nlevels-1:
            ax.set_ylim(ylim*yl_factor)
        else:
            ax.set_ylim(ylim)

        if level_i==0:
            for lobe,name in zip(lobe_bounds,lobe_names):
                ax.text(np.mean(lobe),ax.get_ylim()[1]*1.1,name,horizontalalignment='center')

        axs.append(ax)

    return fig,axs,gs


def plot_proj_roi_summary(roi_proj_all,CIs,eventnames,roilabels,yl_factor=1,
                          roi_proj_subjects=None,ylim=None):
    # roi_proj_all:  Left and Right hemisphere average projections (2 x Nlevels x Nrois)
    # CIs: Upper and Lower Left and Right CIs (2 x 2 x Nlevels x Nrois)
    # yl_factor: scale factor on the y axis of all but the last axis (so the last axis can be bigger if you want)
    # roi_proj_subjects: (optional) single subject data to superimpose on bars (2 x Nsubjects x Nlevels x Nrois)

    (_,Nlevels,Nrois) = roi_proj_all.shape

    fig = plt.figure(figsize=[6.5,7.5])
    fig.clf()
    gs = gridspec.GridSpec(Nlevels+1, 1,height_ratios=[yl_factor]*(Nlevels-1)+[1,0.5],hspace=0)
    ind = np.arange(Nrois)
    width = 0.35
    if ylim is None:
        if roi_proj_subjects is None:
            ylim = np.nanmax(np.abs(CIs))*np.array([-1,1])
        else:
            ylim = np.nanmax(np.abs(roi_proj_subjects))*np.array([-1,1])

    axs = []
    for level in range(Nlevels):
        ax = fig.add_subplot(gs[level])

        L_heights = roi_proj_all[0, level, :]
        R_heights = roi_proj_all[1, level, :]
        L_cis = CIs[:, 0, level, :]
        R_cis = CIs[:, 1, level, :]

        L_bars = ax.bar(ind - width / 2, L_heights, width, yerr=np.abs(np.subtract(L_cis, L_heights)), color='SkyBlue', label='Left')
        R_bars = ax.bar(ind + width / 2, R_heights, width, yerr=np.abs(np.subtract(R_cis, R_heights)), color='IndianRed', label='Right')

        if roi_proj_subjects is not None:
            Nsubjects = roi_proj_subjects.shape[1]
            L_scatter = roi_proj_subjects[0,:,level,:]
            R_scatter = roi_proj_subjects[1,:,level,:]
            repind = np.repeat(ind[np.newaxis,:],Nsubjects,axis=0)
            ax.scatter(repind - width / 2,L_scatter,16,c=None,marker='o',zorder=1.5,linewidth=1,edgecolor='k')
            ax.scatter(repind + width / 2,R_scatter,16,c=None,marker='o',zorder=1.5,linewidth=1,edgecolor='k')


        ax.axhline(0,color='k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(eventnames[level])
        if level<Nlevels-1:
            ax.set_ylim(ylim*yl_factor)
        else:
            ax.set_ylim(ylim)
        ax.set_xticks(ind)
        ax.set_xticklabels('')
        axs.append(ax)

    axs[Nlevels-1].set_xticklabels(roilabels,rotation='vertical')
    axs[Nlevels-1].legend(loc='best')
    plt.show()

    return axs,fig

def plot_lobe_projection_summary(pacs,freq_profiles,eventnames,stats_alpha=0.05,
                                 Nprojs=3,savepath=None):
    '''
    Plot a bar graph and a scatter plot of the breakdown of the projections by
    lobe, across subjects

    See Figure 4 in Stephen et al 2019

    Parameters
    ----------
    pacs : list of spatial_phase_amplitude_coupling.SourceSpacePAC
        The PAC objects for all subjects
    freq_profiles : dict
        The frequency profiles (output from spatial_phase_amplitude_coupling.compute_frequency_profiles)
        Having at least item:
        'U' -> ndarray, (Nf,Nf), frequency profiles (in columns)
    eventnames : list of basestring
        List of names of the levels
    stats_alpha : numeric
        The alpha level for a bootstrap confidence interval across subjects
    Nprojs : int
        The number of projections (there will be one plot for each projection)
    savepath : basestring | None
        The path to save to (or None for no save)

    Returns
    -------

    '''
    Nlevels = len(eventnames)
    Nsubjects = len(pacs)

    parc = 'lobes' # 'aparc.a2005s', 'aparc'
    ordered_rois = [
        # Frontal
        'frontal-lh',
        'frontal-rh',
        # Parietal
        'parietal-lh',
        'parietal-rh',
        # Temporal
        'temporal-lh',
        'temporal-rh',
        # Occipital
        'occipital-lh',
        'occipital-rh']#,
        # Other
        # 'cingulate-lh',
        # 'cingulate-rh',
        # 'insula-lh',
        # 'insula-rh']
    lobes = ['Frontal']*2 + ['Parietal']*2 + ['Temporal']*2 + ['Occipital']*2# + ['Other']*4

    yl_factor = 0.5
    Nbootstrap = 1000
    for proji in range(Nprojs):
        ordered_rois, lobe_names, lobe_bounds, roi_data_all_collected, \
        reject_collected, roi_data_all_average, reject_average =\
            prepare_roi_summary(pacs,eventnames,None,
                                None,None,parc=parc,amp_band_i=proji,
                                ordered_rois=ordered_rois,lobes=lobes,project_onto_U=freq_profiles['U'])

        Nrois = round(len(ordered_rois)/2)

        # Bootstrap over subjects for confidence intervals
        roi_data_average_bs = np.zeros((Nlevels, len(ordered_rois), Nbootstrap))
        for i in range(Nbootstrap):
            sample = np.random.randint(0,Nsubjects,(10))
            roi_data_average_bs[:, :, i] = np.nanmean(roi_data_all_collected[sample, :, :], axis=0) # Nsubjects x Nlevels x Nrois


        ## Bar plots
        CIs = np.percentile(roi_data_average_bs,[100*stats_alpha/2,100*(1-stats_alpha/2)],axis=2)
        CIs_reshaped = np.zeros((2,2,Nlevels,Nrois))
        CIs_reshaped[:,0,:,:] = CIs[:,:,::2] # Left Hemi
        CIs_reshaped[:,1,:,:] = CIs[:,:,1::2] # Right Hemi

        if proji==0:
            ylim_bars = np.nanmax(np.abs(CIs))*np.array([-1,1])

        roi_data_reshaped = np.zeros((2,Nlevels,Nrois))
        roi_data_reshaped[0,:,:] = roi_data_all_average[:,::2] # Left Hemi
        roi_data_reshaped[1,:,:] = roi_data_all_average[:,1::2] # Right Hemi

        roilabels = [label[:-3].capitalize() for label in ordered_rois[::2]]

        axs,fig = plot_proj_roi_summary(
            roi_data_reshaped,CIs_reshaped,eventnames,roilabels,
            yl_factor=yl_factor,roi_proj_subjects=None,ylim=ylim_bars)
        axs[0].set_title(f'Projections onto Frequency Mode #{proji+1}')
        fig.set_size_inches(5,8)

        #### Plot points
        # roi_data_all_collected # Nsubjects x Nlevels x Nrois
        mask = np.ones(roi_data_all_collected.shape).astype(bool)
        width = 0.35
        xaxis = np.array([val for i in range(Nrois) for val in [i-width/2,i+width/2]])
        xaxis_repeated = np.repeat(xaxis[np.newaxis,:],Nsubjects,axis=0)
        size = 36
        adjusted_lobe_bounds = [[xaxis[i] for i in tup] for tup in lobe_bounds]

        if proji==0:
            ylim_scatter = np.nanmax(np.abs(roi_data_all_collected)) * 1.1 * np.array([-1, 1])

        fig_subjects, ax_subjects, gs = plot_roi_summary(roi_data_all_collected, mask, xaxis_repeated, size, eventnames,
                                                               adjusted_lobe_bounds, lobe_names, ordered_rois, yl_factor=yl_factor,
                                                                      ylim=ylim_scatter)
        fig_subjects.set_size_inches(5, 8)
        ax_subjects[-1].set_xticklabels(['Left','Right']*Nrois)
        gs.update(left=0.18)
        fig_subjects.suptitle(f'Projections onto Frequency Mode #{proji+1}')

        if savepath is not None:
            fname = f'ROI_Proj{proji+1}_Average'
            fig.savefig(op.join(savepath,fname+'.png'))

            fname = f'ROI_Proj{proji+1}_All_Subjects'
            fig_subjects.savefig(op.join(savepath, fname + '.png'))

