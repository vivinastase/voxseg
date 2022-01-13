# Module for evaluating performance of vad models,
# may also be run directly as a script
# Author: Nick Wilkinson 2021
import argparse

import sys
import logging

import numpy as np
import pandas as pd
import math

from scipy.io import wavfile
from typing import Dict
from voxseg import utils

import itertools

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score

import tensorflow as tf


def score(wav_scp: pd.DataFrame, sys_segs: pd.DataFrame, ref_segs: pd.DataFrame, wav_segs: pd.DataFrame, winlen: float):
    '''Function for calculating the syllable scores
    
    Args:
        wav_scp: A pd.DataFrame containing information about the wavefiles that have been segmented.
        sys_segs: A pd.DataFrame containing the endpoints produced by a VAD system.
        ref_segs: A pd.DataFrame containing the ground truth reference endpoints. 
        winlen: The time interval covered by a column in the spectrogram -- not used right now   
    '''

    for _, row in wav_scp.iterrows():
        f = row['extended filename']
        rec_id = row['recording-id']
        rate, signal = wavfile.read(f) 
        len_data = len(signal)

        t_labels = getWavLabels(len_data, rate, rec_id, ref_segs, winlen) 
        p_labels = getWavLabels(len_data, rate, rec_id, sys_segs, winlen)
        
        cm = confusion_matrix(t_labels, p_labels)
        prec = precision_score(t_labels, p_labels)
        rec = recall_score(t_labels, p_labels)
        f1 = f1_score(t_labels, p_labels)
        acc = accuracy_score(t_labels, p_labels) 
            
    logging.info("Confusion matrix: {}".format(cm))
    logging.info("Speech: {}, Non-speech: {}, All: {}".format(np.sum(cm[1]), np.sum(cm[0]), np.sum(cm)))
    
    logging.info('\n\n\tPrecision = {}\n\tRecall = {}\n\tF1 = {}\n\tAccuracy = {}'.format(round(prec, 3), round(rec, 3), round(f1, 3), round(acc, 3)))    
    

def ___score(wav_scp: pd.DataFrame, sys_segs: pd.DataFrame, ref_segs: pd.DataFrame, wav_segs: pd.DataFrame, winlen: float) -> Dict[str,Dict[str,int]]:
    '''Function for calculating the TP, FP, FN and TN counts from VAD segments and ground truth reference segments.

    Args:
        wav_scp: A pd.DataFrame containing information about the wavefiles that have been segmented.
        sys_segs: A pd.DataFrame containing the endpoints produced by a VAD system.
        ref_segs: A pd.DataFrame containing the ground truth reference endpoints.
        wav_segs (optional): A pd.DataFrame containing endpoints used prior to VAD segmentation. Only
        required if VAD was applied to subsets of wavefiles rather than the full files. (Default: None)

    Return:
        A dictionary of dictionaries containing TP, FP, FN and TN counts 
    '''

    ref_segs_masks = _segments_to_mask(wav_scp, ref_segs, winlen)
    sys_segs_masks = _segments_to_mask(wav_scp, sys_segs, winlen) 
    if wav_segs is not None:
        wav_segs_masks = _segments_to_mask(wav_scp, wav_segs, winlen)
    scores = {}
    for i in ref_segs_masks:
        if wav_segs is not None:
            score_array = wav_segs_masks[i] * (ref_segs_masks[i] - sys_segs_masks[i])
            num_ground_truth_p = int(np.sum(wav_segs_masks[i] * ref_segs_masks[i]))
            num_frames = int(np.sum(wav_segs_masks[i]))
        else:
            score_array = ref_segs_masks[i] - sys_segs_masks[i]
            num_ground_truth_p = int(np.sum(ref_segs_masks[i]))
            num_frames = len(ref_segs_masks[i])
        num_ground_truth_n = num_frames - num_ground_truth_p
        num_fn = (score_array == 1.0).sum()
        num_fp = (score_array == -1.0).sum()
        num_tp = num_ground_truth_p - num_fn
        num_tn = num_ground_truth_n - num_fp
        scores[i] = {'TP': num_tp, 'FP': num_fp, 'FN': num_fn, 'TN': num_tn}
        
    print("Scores: {}".format(pd.DataFrame(scores)))
    return scores


def score_syllables(wav_scp: pd.DataFrame, sys_segs: pd.DataFrame, ref_segs: pd.DataFrame, winlen: float):
    '''Function for calculating the syllable scores
    
    Args:
        wav_scp: A pd.DataFrame containing information about the wavefiles that have been segmented.
        sys_segs: A pd.DataFrame containing the endpoints produced by a VAD system.
        ref_segs: A pd.DataFrame containing the ground truth reference endpoints. 
        winlen: The time interval covered by a column in the spectrogram -- not used right now   
    '''

    max_tolerance = 6
    #scores = [{'TP': 0, 'FP': 0, 'Nr_syll': 0}] * max_tolerance
    scores = []
    for i in range(max_tolerance):
        scores.append({'TP': 0, 'FP': 0, 'Nr_syll': 0})

    for _, row in wav_scp.iterrows():
        f = row['extended filename']
        rec_id = row['recording-id']
        rate, signal = wavfile.read(f) 
        len_data = len(signal)

        (t_onsets, t_offsets) = getWavBoundaries(len_data, rate, rec_id, ref_segs) 
        (p_onsets, p_offsets) = getWavBoundaries(len_data, rate, rec_id, sys_segs)
        
        #logging.debug("Processing file: {}".format(f))
        #logging.debug("predictions:\n\t onsets: {}\n\t offsets: {}".format(p_onsets, p_offsets))
        #logging.debug("reference:\n\t onsets: {}\n\t offsets: {}".format(t_onsets, t_offsets))
    
        
        for t in range(max_tolerance):
            (TP, FP, nr_syll) = score_boundaries(p_onsets, p_offsets, t_onsets, t_offsets, int(t * winlen * rate))
            scores[t]['TP'] += TP 
            scores[t]['FP'] += FP
            scores[t]['Nr_syll'] += nr_syll
                        
    for t in range(max_tolerance):
        scores[t]["Prec"] = scores[t]['TP'] / (scores[t]['TP'] + scores[t]['FP'] + sys.float_info.epsilon)
        scores[t]["Rec"]  = scores[t]['TP']/ (scores[t]['Nr_syll'] + sys.float_info.epsilon)
        scores[t]["F1"] = 2 * scores[t]["Prec"] * scores[t]["Rec"] / (scores[t]["Prec"] + scores[t]["Rec"] + sys.float_info.epsilon)
        
    logging.info("Syllable scores for tolerance between 0 and {}:\n{}".format(max_tolerance-1, pd.DataFrame(scores)))
    
    


def _segments_to_mask(wav_scp: pd.DataFrame, segments: pd.DataFrame, frame_length: float = 0.01) -> Dict[str,np.ndarray]:
    '''Auxillary function used by score(). Creates a dictionary of recording-ids to np.ndarrays,
    which are boolean masks indicating the presence of segments within a recording.

    Args:
        wav_scp: A pd.DataFrame containing wav file data in the following columns:
            [recording-id, extended filename]
        segments: A pd.DataFrame containing segments file data in the following columns:
            [utterance-id, recording-id, start, end]
        frame_length (optional): The length of the frames used for scoring in seconds. (Default: 0.01)

    Returns:
        A dictionary mapping recording-ids to np.ndarrays, which are boolean masks of the frames
        which makeup segments within a recording.

    Example:
        A 0.1 second clip with a segment starting at 0.03 and ending 0.07 would yeild a mask:
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    '''

    wav_scp = wav_scp.copy()
    wav_scp['duration'] = wav_scp['extended filename'].apply(lambda x: len(wavfile.read(x)[1])/wavfile.read(x)[0]).astype(float)    
    wav_scp['mask'] = round(wav_scp['duration'] / frame_length).astype(int).apply(np.zeros)

    segments = segments.copy()    
    segments['frames'] = (round(segments['end'] / frame_length).astype(int) - \
                          round(segments['start'] / frame_length).astype(int)).apply(np.ones)
   
    #segments['frames'] = segments.apply(lambda x: np.ones(math.ceil(x['end']/frame_length) - math.floor(x['start']/frame_length)), axis=1)
    
    temp = wav_scp.merge(segments, how='left', on='recording-id')

    for n,_ in enumerate(temp['mask']):
                
        if not np.isnan(temp['start'][n]):                        
            temp['mask'][n][round(temp['start'][n] / frame_length):round(temp['end'][n] / frame_length)] = temp['frames'][n]
            #temp['mask'][n][math.floor(temp['start'][n] / frame_length):math.ceil(temp['end'][n] / frame_length)] = temp['frames'][n]
            
    if len(wav_scp.index) > 1:
        wav_scp['mask'] = temp['mask'].drop_duplicates().reset_index(drop=True)
    else:
        wav_scp['mask'] = temp['mask']
    return wav_scp[['recording-id', 'mask']].set_index('recording-id')['mask'].to_dict()

    
def print_scores(labels_true, labels_pred):

    ## there is some conflict with the sklearn and the joblib library, and the import doesn't work                
    #print("Confusion matrix: \n{}\n".format(confusion_matrix(labels_true, labels_pred)))
    #print("F1 score: {}".format(f1_score(labels_true, labels_pred)))           
    #print("Accuracy: {}".format(accuracy_score(labels_true, labels_pred)))  
        
    scores = [[0,0],[0,0]]
    for i in range(0, len(labels_true)):
        scores[labels_true[i]][labels_pred[i]] += 1
        
    logging.info("Scores: {}".format(scores))
    prec = scores[1][1] / (scores[1][1] + scores[0][1] + sys.float_info.epsilon)
    rec = scores[1][1] / (scores[1][1] + scores[1][0] + sys.float_info.epsilon)
    accuracy = (scores[0][0] + scores[1][1]) / (scores[0][0] + scores[0][1] + scores[1][0] + scores[1][1])
    
    logging.info("Precision = {}\nRecall = {}\nF-score = {}\nAccuracy = {}\n".format(prec, rec, 2*prec*rec/(prec+rec + sys.float_info.epsilon), accuracy))
    

def score_mat(targets, eval_data, n_columns, interval_length, winstep, thresh):

    labels_pred = make_labels_seq(targets, "pred", n_columns, interval_length, winstep, thresh)
    #labels_true = make_labels_seq(eval_data, "true", n_columns, interval_length, winstep, thresh)
    labels_true = targets['labels'][0] 
                
    print_scores(labels_true, labels_pred)    
    return syllable_score(labels_pred, labels_true, tolerance=3)
            

def make_labels_seq(data, data_type, n_columns, interval_length, winstep, thresh ):

    labels = [0] * n_columns
    
    prev_pos = 0    
    for _, row in data.iterrows():
        
        if data_type == "true":
            cls_label = classLabel(row['labels'])
            if cls_label == 1:
                labels[row['start']:row['end']+1] = [1] * (row['end'] + 1 - row['start'])
        else:

            k = 0
            prev_pos = row['start']                
            for [p0,p1] in row['predicted-targets']:
                k += 1
                if p1 >= thresh:
                    cls_label = 1
                else:
                    cls_label = 0
                    
                if k == len(row['predicted-targets']):
                    prev_pos = row['end'] + 1 - interval_length
                
                for i in range(prev_pos, min(n_columns,prev_pos + interval_length)):
                    labels[i] += cls_label
                    if labels[i] > 1:
                        labels[i] = 1
                prev_pos += winstep

    return labels


        
def classLabel(string):
    
    if string == "speech":
        return 1
    return 0


def syllable_score(predict, true, tolerance=0):
    
    (p_onset, p_offset) = getBoundaries(predict,"pred")
    (t_onset, t_offset) = getBoundaries(true,"true")

    (TP, FP, nr_syll) =  score_boundaries(p_onset, p_offset, t_onset, t_offset, tolerance)

    precision = TP / (FP + TP + sys.float_info.epsilon)
    recall = TP / (nr_syll + sys.float_info.epsilon)
    f1_score = 2 * precision * recall / (precision + recall + 1e-12)
     
    logging.info("Syllable scores for tolerance = {}:\n\tprecision = {}\n\trecall = {}\n\tf1score = {}".format(tolerance, precision, recall, f1_score))

    return (precision, recall, f1_score)


def score_boundaries(p_onset, p_offset, t_onset, t_offset, tolerance):

    TP = 0
                
    for i in range(len(p_onset)):
        for j in range(len(t_onset)):
            if abs(p_onset[i]-t_onset[j]) <= tolerance:
                if abs(p_offset[i] - t_offset[j]) <= tolerance:
                    TP += 1
                    
    return (TP, len(p_onset)-TP, len(t_onset))



def getBoundaries(labels, l_type):
    
    ## insert "buffers" at the beginning and end
    labels = np.insert(labels, 0, 0)
    labels = np.append(labels, 0)
    
    onsets, offsets, idx = [], [], []
    for i in range(1, len(labels) - 1):
        if labels[i] == 1:
            if labels[i - 1] == 0:
                onsets.append(i)
            if labels[i + 1] == 0:
                offsets.append(i)
                
      
    if l_type == "pred":          
        for i in range(len(onsets)):
            if offsets[i] - onsets[i] < 7:
                idx.append(i)
        onsets = [i for num, i in enumerate(onsets) if num not in idx]
        offsets = [i for num, i in enumerate(offsets) if num not in idx]
        
    if not len(onsets) == len(offsets):
        print("WARNING:The tonset length is not equal to the toffset length!")

    return (onsets, offsets)



def getWavBoundaries(len: int, rate: int, rec_id: str, segments: pd.DataFrame):
    '''
    Make an array with the labels, to score syllables
    '''
    
    onsets = []
    offsets = []
    
    for _, row in segments.loc[segments['recording-id'] == rec_id].iterrows():
        onsets.append(round(row['start'] * rate))
        offsets.append(round(row['end'] * rate))
        
    return (onsets, offsets)
    
    
    
def getWavLabels(data_len: int, rate: int, rec_id: str, segments: pd.DataFrame, winlen: float):
    '''
    Make an array with the labels, to score syllables
    '''

    labels = [0] * round(data_len / (winlen * rate))
        
    for _, row in segments.loc[segments['recording-id'] == rec_id].iterrows():
        for i in range(round(row['start'] / winlen), round(row['end'] / winlen)):
            labels[i] = 1
        
    return labels
    
 

# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='evaluate.py',
                                     description='Evaluate the performance of VAD model output.')

    parser.add_argument('vad_input_dir', type=str,
                        help='a path to a Kaldi-style data directory that was used as input for the VAD experiment')
    
    parser.add_argument('vad_out_dir', type=str,
                        help='a path to a Kaldi-style data directory that was the output of the VAD experiment')

    parser.add_argument('ground_truth_dir', type=str,
                        help='a path to a Kaldi-style data directory containing the ground truth VAD segments')

    args = parser.parse_args()
    wav_scp, wav_segs, _ = utils.process_data_dir(args.vad_input_dir)
    _, sys_segs, _ = utils.process_data_dir(args.vad_out_dir)
    _, ref_segs, _ = utils.process_data_dir(args.ground_truth_dir)
    score(wav_scp, sys_segs, ref_segs, wav_segs)
    
