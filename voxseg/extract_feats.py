# Module for extracting log-mel spectrogram features,
# may also be run directly as a script
# Author: Nick Wilkinson 2021
import argparse
import os
import logging

import math
import numpy as np
import pandas as pd

from voxseg import utils
from python_speech_features import logfbank


def extract(data: pd.DataFrame, params: dict, rate: int) -> pd.DataFrame:
    '''Function for extracting log-mel filterbank spectrogram features.

    Args:
        data: A pd.DataFrame containing dataset information and signals -- see docs for prep_data().
        params: a dictionary containing
            frame_length -- the size of the input to the CNN, 
            nfilt -- number of filters for logfbank
            winlen -- size of the window for logfbank
            winstep -- step for logfbank 
        rate: Sample rate.

    Returns:
        A pd.DataFrame containing features and metadata.
    '''
    
    print('--------------- Extracting features ---------------')
    interval_length = utils.get_interval_length(rate, params)
    
    logging.info("\tsampling rate = {}".format(rate))
    logging.info("\tinterval length = {}".format(interval_length))
    
    data = data.copy()
    #data['features'] = data.apply(lambda x: _calculate_feats(x, params, rate, interval_length), axis=1)
    #'''
    features = []
    endpoints = []
    for i, row in data.iterrows():
        (feats, ends) = _calculate_feats(row, params, rate, interval_length)
        features.append(feats)
        endpoints.append(ends)
    
    data['features'] = features
    data['endpoints'] = endpoints
    #'''
    data = data.drop(['signal'], axis=1)
    data = data.dropna().reset_index(drop=True)
    
    return data


def extract_from_mat(data, params) -> pd.DataFrame:
    '''
    Function to extract features from the already "spectrogramed" input (read from mat files)
    data: A pd.DataFrame containing the spectrogram (or other information obtained by processing the recording)
    params: a dictionary containing
                frame_length -- the size of the input to the CNN, 
                nfilt -- number of filters for logfbank
                winlen -- size of the window for logfbank
                winstep -- step for logfbank 

    '''
    
    print('--------------- Extracting features ---------------')
    
    data = pd.DataFrame(data)
    data['features'] = data.apply(lambda x: _calculate_mat_feats(x, params), axis=1)
    data = data.drop(['signal'], axis=1)
    data = data.dropna().reset_index(drop=True)
    return data
    
    

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    '''Function for normalizing features using z-score normalization.

    Args:
        data: A pd.DataFrame containing dataset information and features generated by extract().
    
    Returns:
        A pd.DataFrame containing normalized features and metadata.
    '''

    data = data.copy()
    mean_std = data['features'].groupby(data['recording-id']).apply(_get_mean_std)
    mean_std = mean_std.reset_index()
    if 'level_1' in mean_std.columns:
        mean_std = mean_std.drop(['level_1'], axis=1)
    else:
        mean_std = mean_std.drop(['index'], axis=1)
    if 'recording-id' in mean_std.columns:
        data = data.merge(mean_std, on='recording-id')
    else:
        data = pd.concat([data, mean_std], axis=1)
                
    print('--------------- Normalizing features --------------')
    data['normalized-features'] = data.apply(_calculate_norm, axis = 1)
    data = data.drop(['features', 'mean', 'std'], axis=1)
   
    return data


def prep_data(data_dir: str, mode: str="train"):
    '''Function for creating pd.DataFrame containing dataset information specified by Kaldi-style
    data directory.

    Args:
        data_dir: The path to the data directory.

    Returns:
        A pd.DataFrame of dataset information. For example:

            recording-id  extended filename        utterance-id  start  end  signal
        0   rec_00        ~/Documents/test_00.wav  utt_00        10     20   [-49, -43, -35...
        1   rec_00        ~/Documents/test_00.wav  utt_01        50     60   [-35, -23, -12...
        2   rec_01        ~/Documents/test_01.wav  utt_02        135    163  [25, 32, 54...

        Note that 'utterance-id', 'start' and 'end' are optional, will only appear if data directory
        contains 'segments' file.
    '''

    wav_scp, segments, _  = utils.process_data_dir(data_dir)

    # check for segments file and process if found
    if (segments is None) or (mode == "test"):
        print('WARNING: Segments file not found, entire audio files will be processed.')
        (rate, pd_data) = utils.read_sigs(wav_scp)
        wav_scp = wav_scp.merge(pd_data)
        return (rate, wav_scp)
    else:
        data = wav_scp.merge(segments)
        (rate, pd_data) = utils.read_sigs(data)
        data = data.merge(pd_data)
        return (rate, data)


def _calculate_feats(row: pd.DataFrame, params: dict, rate: int, interval_length: int) -> np.ndarray:
    '''Auxiliary function used by extract(). Extracts log-mel spectrograms from a row of a pd.DataFrame
    containing dataset information created by prep_data().

    Args:
        row: A row of a pd.DataFrame created by prep_data().
        params: a dictionary containing
                    frame_length -- the size of the input to the CNN, 
                    nfilt -- number of filters for logfbank
                    winlen -- size of the window for logfbank
                    winstep -- step for logfbank 
        rate: Sample rate.

    Returns:
        An np.ndarray of features.
    '''

    sig = row['signal']
    if 'utterance-id' in row:
        id = row['utterance-id']
    else:
        id = row['recording-id']
        
    try:
        nfft = int(params['winlen'] * rate)
        len_sig = len(sig)
        
        assert len(range(0, len_sig-1-interval_length, interval_length)) > 0
        feats = []
        ends = []
        
        for j in utils.progressbar(range(0, len_sig-1-interval_length, interval_length), id):
            feats.append(np.flipud(logfbank(sig[j:j + interval_length], rate, winlen=params['winlen'], winstep=params['winstep'], nfilt=params['nfilt'], nfft=nfft).T))
            ends.append([j, j+interval_length])
            
        ## add last incomplete interval
        if ends[-1][1] < len_sig:
            feats.append(np.flipud(logfbank(sig[len_sig-interval_length:len_sig], rate, winlen=params['winlen'], winstep=params['winstep'], nfilt=params['nfilt'], nfft=nfft).T))
            ends.append([len_sig-interval_length, len_sig])    
                        
        return (np.array(feats), np.array(ends))
        #return np.array(feats)
    
    except AssertionError:
        print(f'WARNING: {id} is too short to extract features, will be ignored.')
        return (np.nan, np.nan)
        #return np.nan


def _calculate_mat_feats(row: pd.DataFrame, params: dict) -> np.ndarray:
    '''Auxiliary function used by extract(). Extracts log-mel spectrograms from a row of a pd.DataFrame
    containing dataset information created by prep_data().

    Args:
        row: A row of a pd.DataFrame created by prep_data().
        params: a dictionary containing
                    frame_length -- the size of the input to the CNN, 
                    nfilt -- number of filters for logfbank
                    winlen -- size of the window for logfbank
                    winstep -- step for logfbank 

    Returns:
        An np.ndarray of features.
    '''

    sig = row['signal']
    if 'utterance-id' in row:
        id = row['utterance-id']
    else:
        id = row['recording-id']
        
    try:
        (interval_length, winstep) = utils.get_mat_interval_length(params) ## to make overlapping frames 
               
        len_sig = len(sig) 
        assert len(range(0, len_sig-1-interval_length, interval_length)) > 0
        feats = []
        end = 0
        for j in utils.progressbar(range(0, len_sig-1-interval_length, winstep), id):
            feats.append(np.flipud(np.array(sig[j:j+interval_length]).T))
            end = j+interval_length

        ## add last incomplete interval
        if end < len_sig-1:
            feats.append(np.flipud(np.array(sig[len_sig-interval_length:len_sig]).T))
        
        return np.array(feats)

    except AssertionError:
        print(f'WARNING: {id} is too short to extract features, will be ignored.')



def _calculate_norm(row: pd.DataFrame) -> np.ndarray:
    '''Auxiliary function used by normalize(). Calculates the normalized features from a row of
    a pd.DataFrame containing features and mean and standard deviation information (as generated
    by _get_mean_std()).

    Args:
        row: A row of a pd.DataFrame created by extract, with additional mean and standard deviation
        columns created by  _get_mean_std().

    Returns:
        An np.ndarray containing normalized features.
    '''

    return np.array([(i - row['mean']) / row['std'] for i in row['features']])


def _get_mean_std(group: pd.core.groupby) -> pd.DataFrame:
    '''Auxiliary function used by normalize(). Calculates mean and standard deviation of a
    group of features.

    Args:
        group: A pd.GroupBy object referencing the features of a single wavefile (could be
        from multiple utterances).

    Returns:
        A pd.DataFrame with the mean and standard deviation of the group of features.
    '''

    return pd.DataFrame({'mean': [np.mean(np.vstack(group.to_numpy()))],
                         'std': [np.std(np.vstack(group.to_numpy()))]})

# Handle args when run directly
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract_feats.py',
                                     description='Extract log-mel spectrogram features.')

    parser.add_argument('data_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', and optionally \'segments\'')
    
    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where features and metadata will be saved as feats.h5')
    
        
    parser.add_argument('-f', '--frame_length', type=int, default=32,
                        help='the frame length to consider (originally was defaulted to 0.32 -- smaller values have an impact on the CNN architecture -- the information gets compressed quickly and it cannot pass through 3 layers')
    
    parser.add_argument('-n', '--nfilt', type=int, default=32,
                        help='the number of filters to extract from the signal data -- will be one of the dimensions of the input to the CNN (nfilt x frame-length (as array length))')

    parser.add_argument('-l', '--winlen', type=float, default=0.025,
                        help='the window length parameter for extracting features from the signal data with logfbank')

    parser.add_argument('-w', '--winstep', type=float, default=0.01,
                        help='the window step parameter for extracting features with logfbank (which determines how much the windows from which features are extracted overlap)')


    args = parser.parse_args()
    
    params = {"frame_length": args.frame_length, "nfilt": args.nfilt, "winlen": args.winlen, "winstep": args.winstep}
        
    (rate, data) = prep_data(args.data_dir)
    feats = extract(data, params, rate)
    feats = normalize(feats)
    
    if not os.path.exists(args.out_dir):
        print(f'Directory {args.out_dir} does not exist, creating it.')
        os.mkdir(args.out_dir)
    utils.save(feats, f'{args.out_dir}/feats.h5')
