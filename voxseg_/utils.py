# Various utility functions for sub-tasks frequently used by the voxseg module
# Author: Nick Wilkinson 2021
import pickle
from tensorflow.lite.python.schema_py_generated import AbsOptionsStart,\
    AddNOptionsStart
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd
import numpy as np
import os
import sys

from scipy.io import wavfile
from scipy.io import loadmat

import collections
import itertools

from typing import Iterable, TextIO, Tuple
import warnings
import logging


from sklearn.metrics import confusion_matrix, precision_recall_curve

import glob

import voxseg


def load(path: str) -> pd.DataFrame:
    '''Reads a pd.DataFrame from a .h5 file.

    Args:
        path: The filepath of the .h5 file to be read.

    Returns:
        A pd.DataFrame of the data loaded from the .h5 file
    '''

    return pd.read_hdf(path)


def process_data_dir(path: str, params, mode: str='train') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Function for processing Kaldi-style data directory containing wav.scp,
    segments (optional), and utt2spk (optional).

    Args:
        path: The path to the data directory.
        params: sound processing parameters -- used to extend slightly each segment, such that in training we can have frames that overlap segment boundaries (if most of the segment is part of a vocalization, it's label will be 1, if not, it will be 0)

    Returns:
        A tuple of pd.DataFrame in the format (wav_scp, segments, utt2spk), where
        pd.DataFrame contain data from the original files -- see docs for read_data_file().
        If a file is missing a null value is returned eg. a directory without utt2spk would
        return:
        (wav_scp, segments, None)

    Raises:
        FileNotFoundError: If wav.scp is not found.
    '''
    
    logging.info("Processing directory: {}".format(path))

    files = [f for f in os.listdir(path) if os.path.isfile(f'{path}/{f}')]
    try:
        wav_scp = read_data_file(f'{path}/wav.scp')
        wav_scp.columns = ['recording-id', 'extended filename']
    except FileNotFoundError:
        print('ERROR: Data directory needs to contain wav.scp file to be processed.')
        raise
    
    if 'segments' not in files:
        segments = None
    else:
        segments = read_data_file(f'{path}/segments')
        segments.columns = ['utterance-id', 'recording-id', 'start', 'end']
        segments[['start', 'end']] = segments[['start', 'end']].astype(float)
        
        '''
        ## extend the segments that are shorter than what is necessary to make a frame of the given frame length
        if mode != 'eval':
            wav_lens = dict()
            for _, row in wav_scp.iterrows():        
                rate, signal = wavfile.read(row['extended filename'])
                wav_lens[row['recording-id']] = len(signal)/rate

            min_len = (params['frame_length']-1) * params['winstep'] + params['winlen']            
            segments[['start','end']] = segments.apply(lambda x: _change_ends(x, min_len, wav_lens), axis=1)
        '''

    
    if 'utt2spk' not in files:
        utt2spk = None
    else:
        utt2spk = read_data_file(f'{path}/utt2spk')
        utt2spk.columns = ['utterance-id', 'speaker-id']
    return wav_scp, segments, utt2spk


def progressbar(it: Iterable, prefix: str = "", size: int = 45, file: TextIO = sys.stdout) -> None:
    '''Provides a progress bar for an iterated process.

    Args:
        it: An Iterable type for which to provide a progess bar.
        prefix (optional): A string to print before the progress bar. Defaults to empty string.
        size (optional): The number of '#' characters to makeup the progressbar. Defaults to 45.
        file (optional): A text file type for output. Defaults to stdout.

    Code written by eusoubrasileiro, found at:
    https://stackoverflow.com/questions/3160699/python-progress-bar/34482761#34482761
    '''

    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def read_data_file(path: str) -> pd.DataFrame:
    '''Function for reading standard Kaldi-style text data files (eg. wav.scp, utt2spk etc.)

    Args:
        path: The path to the data file.

    Returns:
        A pd.DataFrame containing the enteries in the data file.
    
    Example:
        Given a file 'data/utt2spk' with the following contents:
        utt0    spk0
        utt1    spk1
        utt1    spk2

        Running the function yeilds:
        >>> print(read_data_file('data/utt2spk'))
                0       1
        0    utt0    spk0
        1    utt1    spk1
        2    utt2    spk2
    
    '''

    with open(path, 'r') as f:
        return pd.DataFrame([i.split() for i in f.readlines()], dtype=str)


def read_sigs(data: pd.DataFrame) -> pd.DataFrame:
    '''Reads audio signals from a pd.DataFrame containing the directories of
    .wav files, and optionally start and end points within the .wav files. 

    Args:
        data: A pd.DataFrame created by prep_data().

    Returns:
        A pd.DataFrame with columns 'recording-id' and 'signal', or if segments were provided
        'utterance-id' and 'signal'. The 'signal' column contains audio as np.ndarrays.

    Raises:
        AssertionError: If a wav file is not 16k mono.
    '''

    wavs = {}
    ret = []
    for i, j in zip(data['recording-id'].unique(), data['extended filename'].unique()):
        rate, wavs[i] = wavfile.read(j)
        #assert rate == 16000 and wavs[i].ndim == 1, f'{j} is not formatted in 16k mono.'
        assert wavs[i].ndim == 1, f'{j} is not formatted in mono.'
        logging.info("File: {}\tlength: {}\tsampling rate: {}".format(j, len(wavs[i]), rate))
        
    if 'utterance-id' in data:
        for _, row in data.iterrows():
            ret.append([row['utterance-id'], wavs[row['recording-id']][int(float(row['start']) * rate): int(float(row['end']) * rate)]])
        return (rate, pd.DataFrame(ret, columns=['utterance-id', 'signal']))
    else:
        for _, row in data.iterrows():
            ret.append([row['recording-id'], wavs[row['recording-id']]])
        return (rate, pd.DataFrame(ret, columns=['recording-id', 'signal']))


def save(data: pd.DataFrame, path: str) -> None:
    '''Saves a pd.DataFrame to a .h5 file.

    Args:
        data: A pd.DataFrame for saving.
        path: The filepath where the pd.DataFrame should be saved.
    '''

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    data.to_hdf(path, key='data', mode='w')


def time_distribute(data: np.ndarray, sequence_length: int, stride: int = None, z_pad: bool = True) -> np.ndarray:
    '''Takes a sequence of features or labels and creates an np.ndarray of time
    distributed sequences for input to a Keras TimeDistributed() layer.

    Args:
        data: The array to be time distributed.
        sequence_length: The length of the output sequences in samples.
        stride (optional): The number of samples between sequences. Defaults to sequence_length.
        z_pad (optional): Zero padding to ensure all sequences to have the same dimensions.
        Defaults to True.

    Returns:
        The time ditributed data sequences.

    Example:
        Given an np.ndarray of data:
        >>> data.shape
        (10000, 32, 32, 1)
        >>> time_distribute(data, 10).shape
        (1000, 10, 32, 32, 1)
        The function yeilds 1000 training sequences, each 10 samples long.
    '''
    
    if stride is None:
        stride = sequence_length
    if stride > sequence_length:
        print('WARNING: Stride longer than sequence length, causing missed samples. This is not recommended.')
    td_data = []
    if len(data) < stride:
        td_data.append(data)
    else:
        for n in range(0, len(data)-sequence_length+1, stride):
            td_data.append(data[n:n+sequence_length])
                
    if z_pad:
        if len(td_data)*stride+sequence_length != len(data)+stride:
            z_needed = len(td_data)*stride+sequence_length - len(data)
            z_padded = np.zeros(td_data[0].shape)
            for i in range(sequence_length-z_needed):
                z_padded[i] = data[-(sequence_length-z_needed)+i]
            td_data.append(z_padded)
            
    return np.array(td_data)



## adjust the segment boundaries so all segments have at least a given min length, but make sure the new ends do not overshoot the length of the signals
def _change_ends(row, min_len, wav_lens):

    ## if the segment is too close to the beginning or the end to be extended symmetrically, maybe I should extend it asymetrically ... 
    if row['end'] - row['start'] < min_len:    
        return pd.Series({'start':max(0, row['start']-min_len/2), 'end': min(row['end']+min_len/2, wav_lens[row['recording-id']])})
        
    return row[['start','end']]


def get_threshold(y_pred, y_true):
    
    y_pred = extract_flat_labels(y_pred, "pred")
    y_true = list(map(int, extract_flat_labels(y_true, "true")))
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2*recall*precision/(recall+precision)   
    #weights = confusion_matrix(y_true, y_pred).sum(axis=1)
    #weighted_f1_scores = np.average(f1_scores, weights=weights)
    
    
    print('Best threshold: ', thresholds[np.argmax(f1_scores)])    
    print('Best F1-Score: ', np.max(f1_scores))
    
    return float(thresholds[np.argmax(f1_scores)])



def extract_flat_labels(pred_list, labels_type):

    flat_list = []
    
    for frame in pred_list:
        flat_list.extend([x[1] for x in frame])
    
    return flat_list




## compute the length of the signal, such that it returns a number of frames that matches the frame_length
def get_interval_length(rate, params):
            
    return int(rate * ((params['frame_length']-1) * params['winstep'] + params['winlen']))-1 ## size of the interval length to obtain exactly mum_frames frames (as required by the CNN)


def get_mat_interval_length(params):
    
    overlap = 0.25  ## to make overlapping frames 
    return (params['frame_length'], int(params['frame_length'] * (1-overlap)))


def print_frame(pdframe, name):
    
    print("\n____________________________\n{} data frame\n".format(name))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pdframe)
    print("_____________\n")
    #with pd.option_context('display.max_columns', None):  # more options can be specified also
    print("One row:\n{}".format(pdframe.iloc[0]))
    
    '''
    print("{} -- one row:".format(name))
    for k in pdframe.keys():
        print("\tkey={}".format(k))
        atr_len = 0
        if pdframe[k] is not None:
            if isinstance(pdframe[k][0], list):
                atr_len = len(pdframe[k][0])
            elif isinstance(pdframe[k][0], np.ndarray):
                atr_len = pdframe[k][0].shape
        print("\t{}\t\t({}) {}".format(k, atr_len, pdframe[k][0]))
    '''
    print("______________________________\n")
    


def print_full_frame(pdframe, name):
    
    print("\n____________________________\n{} data frame\n".format(name))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pdframe)
    print("_____________\n")
    #with pd.option_context('display.max_columns', None):  # more options can be specified also
    
    
    for i,row in pdframe.iterrows():
        
#        print("row {}: {}".format(i, row))
        for k in pdframe.keys():            
            atr_len = 0
            if row[k] is not None:
                if isinstance(row[k], list):
                    atr_len = len(row[k])
                elif isinstance(row[k], np.ndarray):
                    atr_len = row[k].shape
            
            print("{}:\t({})\t{}".format(k, atr_len, row[k]))
    
    print("______________________________\n")
    

def print_feature_stats(pdframe, name):
    
    print("\n____________________________\n{} feature statistics\n".format(name))

    k = 'normalized-features'

    s = pdframe[k][0].shape
    
    for i in range(1, len(pdframe)):
        s = np.add(s, pdframe[k][i].shape)
        
    print("Sum of shapes: {}".format(s))
    #print("Average dimensions: {}".format(s/len(pdframe)))
    print("_____________________________\n")
    
    
def load_mat_data(data_dir, data_type):  #data_type is train or test, because of the naming conventions for the mat files
    
    logging.info("Loading data from {}".format(data_dir + "/*" + data_type + "*.mat"))
    
    data_file = glob.glob(data_dir + "/*" + data_type + "*classify.mat")[0]
    labels_file = glob.glob(data_dir + "/*" + data_type + "*label.mat")[0]

    data = loadmat(data_file)["Xs_all"].T.astype(np.float32)
    labels = loadmat(labels_file)["label_two"][:,1].astype(int)
    
    N_IN = data.shape[1]
      
    logging.info("\nLoading {} data from {} ...".format(data_type, data_dir))                  
    logging.info("Data dimension: {}".format(N_IN))
    logging.info("Training data info: {} / {} labels".format(data.shape, labels.shape))
    #print("\t data sample: {} \n\t labels sample: {}".format(train_data[0:3], train_label[0:3]))
    logging.info("Class counts: {}".format(collections.Counter(labels)))
    logging.info("Positive label distribution in training data: {}\n".format(sum(labels)/len(labels)))
     
    data = convert_to_pd(data, labels, data_file, data_type) 
    
    return (data, N_IN, len(labels))


'''
Convert the data read from a mat file to the pd format used by VoxSeg
'''
def convert_to_pd(data, labels, data_file, data_type):
        
    rec_id = "rec_0000"
    i = 0
    start = 0
      
    frame = {'recording-id':[], 'utterance-id':[], 'start':[], 'end':[], 'labels':[], 'signal':[], 'extended filename': []}
    
    if data_type == "test":
        frame['extended filename'].append(data_file)
        frame['recording-id'].append(rec_id)
        frame['utterance-id'].append(rec_id + "_" + str(i).zfill(8))
        frame['signal'].append(data)
        frame['start'].append(0)
        frame['end'].append(len(data))
        frame['labels'].append(labels)           ##.append("unk")            
    else:
        for label, seq in itertools.groupby(labels):
            seq_len = len(list(seq))
            frame['extended filename'].append(data_file)
            frame['recording-id'].append(rec_id)
            frame['utterance-id'].append(rec_id + "_" + str(i).zfill(8))
            i += 1
            frame['signal'].append(data[start:start+seq_len])
            frame['start'].append(start)
            start += seq_len
            frame['end'].append(start-1)
            if label == 0:
                frame['labels'].append('non-speech')
            else:
                frame['labels'].append('speech')
                        
    return pd.DataFrame(frame)


## when processing an entire file that will have one row with tens or hundred of thousand of features, 
## there may be an OOM error at prediction time
## this cannot be done at data loading time, because pre-segmenting the file may cause a break in the middle of a speech segment

def resegment(features, rate):
    max_len = 5000
    segmented = {}
    for k in features.keys():
        segmented[k] = []
        
#    del segmented['endpoints']
    
    segmented['utterance-id'] = []
    segmented['endpoints'] = []
    segmented['start'] = []
    segmented['end'] = []
        
    start = 0
    prev_rec_id = ""
    for _, row in features.iterrows():
               
        rec_id = row['recording-id']
        row_len = len(row['features'])
        (row_start, start) = get_start(row,start,prev_rec_id == rec_id)
        
        if row_len > max_len:
            k = 0
            for j in range(0, row_len, max_len):
                segmented['features'].append(row['features'][j:min(j+max_len,row_len)])
                segmented['endpoints'].append(row['endpoints'][j:min(j+max_len,row_len)])
                segmented['recording-id'].append(rec_id)
                segmented['extended filename'].append(row['extended filename'])
                segmented['utterance-id'].append(rec_id + "_" + str(k).zfill(8))
                
                segmented['start'].append(row_start + row['endpoints'][j][0]/rate)
                segmented['end'].append(row_start + row['endpoints'][min(j+max_len,row_len-1)][1]/rate)
                                    
                k += 1
        else:
            segmented['features'].append(row['features'])
            segmented['endpoints'].append(row['endpoints'])
            segmented['recording-id'].append(rec_id)
            segmented['extended filename'].append(row['extended filename'])
            segmented['utterance-id'].append(rec_id + "_" + str(0).zfill(8))
            
            segmented['start'].append(row_start + row['endpoints'][0][0]/rate)
            segmented['end'].append(row_start + row['endpoints'][row_len-1][1]/rate)

        start += segmented['end'][-1]
        prev_rec_id = rec_id

                        
    return pd.DataFrame(segmented)
    
   
def get_start(row, start, isSameRec):
    
    if not isSameRec:
        return (0,0)
    
    if 'start' in row.keys():
        return (row['start'], start)
    return (start, start)


def compute_time(ind, frame_length, rate):
    
    return ind/(rate * frame_length)


def test_file_type(data_dir):
    
    if len(glob.glob(data_dir + "/*classify.mat")) > 0:
        return "mat"
    
    return "voxseg"
