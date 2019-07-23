from scipy.io import loadmat
import os.path as op
import numpy as np

def get_LOC_and_ROC(subject,subjects_dir):
    l = loadmat(op.join(subjects_dir,'input','eeganes_aligntimes.mat'), squeeze_me=False)
    subidx = dict(eeganes02=0,
                  eeganes03=1,
                  eeganes04=2,
                  eeganes05=3,
                  eeganes07=4,
                  eeganes08=5,
                  eeganes09=6,
                  eeganes10=7,
                  eeganes13=8,
                  eeganes15=9)
    LOC_time = l['aligntimes'][subidx[subject],3]*60
    ROC_time = l['aligntimes'][subidx[subject],5]*60

    return [LOC_time,ROC_time]


def load_bhvr(subject,subjects_dir):

    bhvr = loadmat(op.join(subjects_dir, subject, 'input', subject + 'bhvr.mat'), squeeze_me=False)

    propofol_t = np.array(bhvr['spump']['T'][0][0])/bhvr['Fs'][0][0]
    propofol_dose = np.array(bhvr['spump']['prpfol'][0][0])

    stim_t = np.array(bhvr['bhvr']['stim_time'][0][0]['T'][0][0])/bhvr['Fs'][0][0]
    stim_type = {}
    stim_type['burst'] = np.array(bhvr['bhvr']['stim_type'][0][0]['burst'][0][0])
    stim_type['name'] = np.array(bhvr['bhvr']['stim_type'][0][0]['name'][0][0])
    stim_type['word'] = np.array(bhvr['bhvr']['stim_type'][0][0]['word'][0][0])

    resp_type = {}
    resp_type['correct'] = np.array(bhvr['bhvr']['resp_type'][0][0]['correct'][0][0])
    resp_type['incorrect'] = np.array(bhvr['bhvr']['resp_type'][0][0]['incorrect'][0][0])
    resp_type['noresponse'] = np.array(bhvr['bhvr']['resp_type'][0][0]['noresponse'][0][0])

    return propofol_dose, propofol_t, stim_t, stim_type, resp_type


def load_pr_resp(subject,subjects_dir):
    l = loadmat(op.join(subjects_dir,subject,'input','{}bugsoutput2armmove.mat'.format(subject)), squeeze_me=False)
    pr_resp_T = np.array(l['T']).squeeze()/5000
    pr_resp_verbal = np.hstack((l['prob_verbal']['p025'][0][0],
                                l['prob_verbal']['p500'][0][0],
                                l['prob_verbal']['p975'][0][0]))
    pr_resp_clicks = np.hstack((l['prob_burst']['p025'][0][0],
                                l['prob_burst']['p500'][0][0],
                                l['prob_burst']['p975'][0][0]))

    return pr_resp_T,pr_resp_verbal,pr_resp_clicks


def propofol_paths(subjects_dir,subject,spacing_string):

    input_dir = op.join(subjects_dir,subject,'input')

    src_fname = f'{subject}{spacing_string}p-src.fif'
    srcfile = op.join(input_dir, src_fname)

    bemsol_fname = f'{subject}-5120-5120-5120-bem-sol.fif'
    bemfile = op.join(input_dir, bemsol_fname)

    if   subject=='eeganes02':
        trans_fname = 'COR-asalazar-090414-111922.fif'
    elif subject=='eeganes03':
        trans_fname = 'COR-asalazar-090623-114236.fif'
    elif subject=='eeganes04':
        trans_fname = 'COR-asalazar-081126-173601.fif'
    elif subject=='eeganes05':
        trans_fname = 'COR-asalazar-090619-160021.fif'
    elif subject=='eeganes07':
        trans_fname = 'COR-asalazar-090406-142236.fif'
    elif subject=='eeganes08':
        trans_fname = 'COR-asalazar-090414-144318.fif'
    elif subject=='eeganes09':
        trans_fname = 'COR-asalazar-100430-182227.fif'
    elif subject=='eeganes10':
        trans_fname = 'COR-asalazar-090911-172720.fif'
    elif subject=='eeganes13':
        trans_fname = 'COR-asalazar-100721-200830.fif'
    elif subject=='eeganes15':
        trans_fname = 'COR-asalazar-100721-203247.fif'
    else:
        raise RuntimeError('Trans file has not been identified for subject: ' + subject)

    transfile = op.join(input_dir, trans_fname)

    raw_fname = f'{subject}-crop-0-None-raw.fif'
    rawfile = op.join(input_dir, raw_fname)

    badchannelfile_eeg = op.join(subjects_dir,'input','badchannelmat.mat')
    neighborfile = op.join(subjects_dir,'input','channelneighbors.mat')

    paths_dict = {}
    paths_dict['subjects_dir'] = subjects_dir
    paths_dict['srcfile'] = srcfile
    paths_dict['bemfile'] = bemfile
    paths_dict['transfile'] = transfile
    paths_dict['rawfile'] = rawfile
    paths_dict['events_dir'] = input_dir
    paths_dict['badchannels_eeg'] = badchannelfile_eeg
    paths_dict['badchannels_meg'] = None
    paths_dict['badchannels_addPfx'] = True
    paths_dict['neighborfile'] = neighborfile

    return paths_dict


def parse_event_suffix(event_suffix):
    if event_suffix=='_sedation':
        event_filename = 'event_times_level.mat'
        load_suffix = '_byLevel'

        event_savename = 'sedation.xxx'

        eventnames=['Baseline','Sedation','Low Dose','High Dose']
        event_mapping={'eeganes02':[('Baseline', 'baseline'),('Low Dose', 'level2'),('High Dose','level4')],
                         'eeganes03':[('Baseline', 'baseline'),('Sedation', 'level2'),('Low Dose','level4'),('High Dose','level5')],
                         'eeganes04':[('Baseline', 'baseline'),('Sedation', 'level2'),('Low Dose','level3'),('High Dose','level4')],
                         'eeganes05':[('Baseline', 'baseline'),('Sedation', 'level2'),('Low Dose','level3'),('High Dose','level5')],
                         'eeganes07':[('Baseline', 'baseline'),('Sedation', 'level1'),('Low Dose','level3'),('High Dose','level5')],
                         'eeganes08':[('Baseline', 'baseline'),('Sedation', 'level1'),('Low Dose','level2'),('High Dose','level5')],
                         'eeganes09':[('Baseline', 'baseline'),('Low Dose','level1'),('High Dose','level5')],
                         'eeganes10':[('Baseline', 'baseline'),('Sedation', 'level2'),('Low Dose','level4'),('High Dose','level5')],
                         'eeganes13':[('Baseline', 'baseline'),('Sedation', 'level1'),('Low Dose','level2'),('High Dose','level4')], # Correction described in notes 2/19/19
                         'eeganes15':[('Baseline', 'baseline'),('Sedation', 'level3'),('Low Dose','level4'),('High Dose','level5')]}
    else:
        raise RuntimeError(f'Unknown event_suffix: {event_suffix}')
    return eventnames, event_mapping, event_filename, load_suffix, event_savename

