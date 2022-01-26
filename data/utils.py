'''
Created on Dec 8, 2021

@author: vivi
'''

import re
import os
import glob

import random

import collections

import numpy as np

from scipy import signal
from scipy.io import wavfile, loadmat

import pandas as pd

from data import annotations_utils

'''
    generate output formatted for voxseg:
      https://github.com/NickWilkinson37/voxseg
    
    ==> wav.scp - this file provides the paths to the audio files one wishes to use for training, and assigns them a unique recording-id. It is structured as follows: <recording-id> <extended-filename>. Each entry should appear on a new line, for example:

    rec_000 wavs/some_raw_audio.wav
    rec_001 wavs/some_more_raw_audio.wav

    Note that the <extended-filename> may be an absolute path or relative path, except when using Docker or Singularity, where paths relative to the mount point must be used.

    ==> segments - this file specifies the start and end points of each labelled segment within the audio file. Note, this is different to the way this file is used when provided for decoding. This file is structured as follows: <utterance-id> <recording-id> <segment-begin> <segment-end>, where <segment-begin> and <segment-end> are in seconds. Each entry should appear on a new line, for example:

    rec_000_00 rec_000 0.0 4.3
    rec_000_01 rec_000 4.3 7.2
    ...
    rec_001_01 rec_001 8.5 12.2
    rec_001_02 rec_001 12.2 16.1
    ...
    
    ==> utt2spk - this file specifies the label attached to each segment defined within the segments file. This file is structured as follows: <utterance-id> <label>. Each entry should appear on a new line, for example:

    rec_000_00 speech
    rec_000_01 non_speech
    rec_000_02 speech
    rec_000_03 non_speech
'''
def make_voxseg(files_dir, name, data_path, split, wav_ext, annot_ext, sep = "\t"):

    files = glob.glob(files_dir + '/*.' + wav_ext)
    print("Processing {} files".format(len(files)))
    
    if len(glob.glob(files_dir + "/*." + annot_ext)) == 0:
        #no annotations file, make a directory with test data only
        make_voxseg_data(files, list(range(len(files))), data_path, name + ".test", wav_ext, annot_ext, sep, "test")

    else:
    
        if split != "none":
            #data_split = loadmat(split)  ## there is something not clear about the indices in this mat file
            data_split = get_data_split(split, files)
        else:
            data_split = {'i_train': [list(range(len(files)))], 'i_test': [list(range(len(files)))]}
        
        print("Processing {} wav files".format(len(files)))
        splits = ['train', 'valid', 'test']
        
        for x in splits:
            k = 'i_' + x
            if k in data_split.keys():
                make_voxseg_data(files, data_split[k][0], data_path, name + "." + x, wav_ext, annot_ext, sep, x)
        
        ## make a complete version of the test dir, for evaluation purposes
        make_voxseg_data(files, data_split['i_test'][0], data_path, name + ".test_eval", wav_ext, annot_ext, sep, "test")
     
     
def get_data_split(files):
    
    N = len(files)
    split = [0.75, 0.05, 0.2]
    
    ind_list = []
    for i in range(len(split)):
        ind_list.extend([i] * int(N * split[i]))
        
    if len(ind_list) < N:
        ind_list.extend([0] * (N-len(ind_list)))
        
    random.shuffle(ind_list)
    
    split_map = {'i_train': 0, 'i_valid': 1, 'i_test': 2}
    data_split = dict()
    for k in split_map.keys():
        data_split[k] = [[i for i, x in enumerate(ind_list) if x == split_map[k]]]
        print("\t{} => {} files".format(k, len(data_split[k])))
        
    return data_split
    
     
     
def make_voxseg_data(files, indices, data_path, name, wav_ext, annot_ext, sep, split_type):
     
    data_path += "/" + name + "/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    wav_scp  = open(data_path + "/wav.scp", "w")
    segments = open(data_path + "/segments", "w")
    utt2spk  = open(data_path + "/utt2spk", "w") 
    
    n = 0
    nr_segs = 0
    
    print("Reformatting data for voxseg ...")
    print("\toutputs will go to:\n\t{}\n\t{}\n\t{}\n".format(wav_scp, segments, utt2spk))
    print("Files: {}".format(files))
    print("Indices: {}".format(indices))
            
    for i in indices:
        #file_stem = os.path.splitext(files[i-1])[0]  ## doesn't work if a directory contains a "."
        file_path = "/".join(files[i-1].split("/")[:-1])
        file_stem = file_path + "/" + ".".join(files[i-1].split("/")[-1].split(".")[:-1])
        
        wav_file = file_stem + "." + wav_ext
        annot_file = file_stem + "." + annot_ext

        rec_id = make_id("rec", n)
        wav_scp.write("{} {}\n".format(rec_id,wav_file))
        
        n += 1
                
        annots = annotations_utils.get_annotations(annot_file, annot_ext, sep)
        #segs_info = os.popen("wc " + annot_file).read()
        print("{} (i={}) Processing file {} ({}) -- {} segments".format(n, i, wav_file, rec_id, len(annots))) 
        
        nr_segs += len(annots)
        
        process_file4voxseg(annots, segments, utt2spk, split_type == "test", rec_id)
                    
    wav_scp.close()
    segments.close()
    utt2spk.close()
    
    ## if not removed it will confuse voxseg
    if nr_segs == 0:
        os.system("rm " + data_path + "/segments")
        os.system("rm " + data_path + "/utt2spk")
    
    '''
    ## make files only readable so they don't get overwritten by mistake
    if not split_type == "test":
        os.system(" chmod 444 " + data_path + "/wav.scp")
        os.system(" chmod 444 " + data_path + "/segments")
        os.system(" chmod 444 " + data_path + "/utt2spk")
    '''
        
    print("Number of segments for {} split: {}".format(split_type, nr_segs))
    
    
'''
  the zebra finch data is exported from flatclust, 
  and there would be one directory containing all recording files, and one file with all annotations
'''
def make_voxseg_zf(files_dir, name, data_path, annot_file):
    
    print("\nReading information from annotations file {}".format(annot_file))

    (train_annots, test_annots) = split_annots(annot_file)
    
    print("\nExtracting training data from files: {}".format(train_annots['file'].unique().tolist()))
    print("\nExtracting test data from files: {}".format(test_annots['file'].unique().tolist()))
    
    write_voxseg_zf(files_dir, name + ".train", data_path, train_annots, False)
    write_voxseg_zf(files_dir, name + ".test", data_path, test_annots, True)
    

def write_voxseg_zf(files_dir, name, data_path, annots, isTest):

    data_path += "/" + name + "/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    wav_scp  = open(data_path + "/wav.scp", "w")
    segments = open(data_path + "/segments", "w")
    utt2spk  = open(data_path + "/utt2spk", "w") 
    
    print("Reformatting data for voxseg ...")
    print("\toutputs will go to:\n\t{}\n\t{}\n\t{}\n".format(wav_scp, segments, utt2spk))
    
    rec_ids = dict()
        
    n_utt = 0
    prev_end = 0
    
    print("Processing {} instances ... ".format(len(annots)))
    
    for _i, (file, onset, duration, _cluster_id) in annots.iterrows():
          
        wav_file = files_dir + "/" + file
        
        (rec_id, n_utt, prev_end) = get_rec_id(wav_file, rec_ids, wav_scp, n_utt, prev_end)
        samplingRate = getSamplingRate(wav_file)
        
        begin = onset/samplingRate
        end = (onset + duration)/samplingRate
        
        if not isTest:
            add_instance(rec_id, n_utt, segments, utt2spk, prev_end, begin, "non_speech")
            n_utt +=1
            
        add_instance(rec_id, n_utt, segments, utt2spk, begin, end, "speech")   ##call_type)
        prev_end = end

        n_utt += 1
                   
    wav_scp.close()
    segments.close()
    utt2spk.close()    



def get_rec_id(wav_file, rec_ids, wav_scp, n_utt, prev_end):
    
    if not wav_file in rec_ids.keys():
        rec_id = make_id("rec", len(rec_ids))    
        rec_ids[wav_file] = rec_id
        
        wav_scp.write("{} {}\n".format(rec_id,wav_file))
        
        n_utt = 0
        prev_end = 0
        
    return (rec_ids[wav_file], n_utt, prev_end)
    
    
    
def process_file4voxseg(annots, segments, utt2spk, isTest, rec_id):   
    
    n_utt = 0
    prev_end = 0
    for k, row in annots.iterrows():  
        
        (begin, end, call_type) = annotations_utils.get_annot_info(row)
        
        #print("processing segment: {}\t{}\t{}".format(begin, end, call_type))
        
        if is_acceptable(call_type, end, prev_end):  
        
            begin = max(begin, prev_end)
        
            if not isTest and begin > prev_end:
                add_instance(rec_id, n_utt, segments, utt2spk, prev_end, begin, "non_speech")
                n_utt +=1
                
            add_instance(rec_id, n_utt, segments, utt2spk, begin, end, "speech")   ##call_type)
            prev_end = end

            n_utt += 1

    



def process_file(file_stem, wav_ext, annot_ext, annots = None):
    
    print("\nProcessing {} ...".format(file_stem))

    if not "."+wav_ext in file_stem: 
        wav_file = file_stem + "." + wav_ext
    else:
        wav_file = file_stem
                    
    #nperseg = 20
    
    samplingFrequency, signalData = wavfile.read(wav_file)
    frequencies, times, spectrogram = signal.spectrogram(signalData, samplingFrequency)   ##, nperseg=nperseg)  #, scaling='spectrum')

    #spectrogram, frequencies, times = mlab.specgram(signalData, Fs=samplingFrequency)
    #spectrogram = python_speech_features.logfbank(signalData, samplingFrequency, nfilt=128).T  
        
    print("audio length: {}".format(len(signalData)))
    print("sampling frequency: {}".format(samplingFrequency))
    print("spectrogram shape: {}".format(spectrogram.shape))

    return (spectrogram, annotations_utils.get_labels(spectrogram.shape[1], samplingFrequency, annots, file_stem, annot_ext))



  
# when the signal is transformed with the log (in the make_mat_files function), smooth the 0 values, 
def smooth(np_array):
    
    print("frequencies array: {}".format(np_array))
    min_val = np.amin(np_array[np.nonzero(np_array)])/2  ## make the min smaller than the min value in the array
    print("min val in spectrogram for smoothing: {}".format(min_val))

    ## if the value is negative, it means I won't log the spectrogram, so no smoothing necessary
    if min_val > 0:
        np_array += min_val
       
    return np_array
      

def get_file_stems(files_dir, ext):
    
    file_stems = []
    for f in glob.glob(files_dir + '/*.' + ext):
        file_stems.append(os.path.splitext(f)[0])
        
    return file_stems
    

def add_instance(rec_id, n_utt, segments, utt2spk, begin, end, inst_class):
    utt_id = make_id(rec_id,n_utt)
    segments.write("{} {} {} {}\n".format(utt_id, rec_id, begin, end))
    utt2spk.write("{} {}\n".format(utt_id, inst_class))
    
#    print("ADDED: {} {} {} {} {}".format(inst_class, begin, end, utt_id, rec_id))
        

def write_instance(segments, utt2spk, rec_id, n_utt, begin, end, inst_class):
    utt_id = make_id(rec_id,n_utt)
    segments.write("{} {} {} {}\n".format(utt_id, rec_id, begin, end))
    utt2spk.write("{} {}\n".format(utt_id, inst_class))



def make_id(prefix, n):    
    return "{}_{:04d}".format(prefix,n)


def is_acceptable(call_type, end, prev_end):
    
    if end < prev_end:
        return False
    
    if re.match('speech', call_type, re.IGNORECASE):
        return call_type.lower()
    
    ## for meerkats I would test if it contains "_type" (as opposed to "_element") maybe
    
    call_type = re.sub(r'^nf_',"",call_type)
    
    m = re.match(r'^(.*?)_type',call_type) 
    if m:
        return m.group(1)
    #if "_element" in call_type:
    #    return True
    
    if "Marker" in call_type:
        return "Marker"
    
    m = re.match(r"^\d+\_([A-Z]+)", call_type) 
    if m:
        return m.group(1)
    
    return False





## it doesn't matter for this processing whether the call was made by a focal or non-focal animal (which is marked with "nf_" in the call type
def get_call_id(call_dict, call_type, start_id):
    
    call = re.sub("^nf_","",call_type)
    
    if not call in call_dict.keys():
        call_dict[call] = len(call_dict) + start_id   ## in flatclust there is no cluster 0, and cluster 1 has a special usage

    return call_dict[call]


'''
 get the annotations file exported from flatclust, with all annotations 
'''
def get_annot_file(files_dir, ext):
    
    files = glob.glob(files_dir + "/*." + ext)
    if len(files) > 1:
        print("too many potential annotation files ... don't know how to choose\nStopping.")
        exit
        
    return files[0]

'''
  split the annotations exported form flatclust into train and test
'''
def split_annots(annot_file):

    split = [0.8, 0.2]  ## approximate annotations split ratio
    annots = pd.read_csv(annot_file, sep=',')
    
    instances = annots["file"].tolist()
    n_inst = len(instances)

    (train_files, test_files) = split_files(collections.Counter(instances), n_inst * split[0])

    return(annots[annots["file"].isin(train_files)], 
           annots[annots["file"].isin(test_files)])


def split_files(counts, max_inst):
    
    train_files = []
    test_files = []
    
    n = 0
    for f in counts.keys():
        if counts[f] + n > max_inst:
            test_files.append(f)
        else:
            train_files.append(f)
        n += counts[f]
        
    return (train_files, test_files)
        

def getSamplingRate(wav_file):
    
    samplingFrequency, _signalData = wavfile.read(wav_file)
    #print("Sampling rate of {} = {}".format(wav_file, samplingFrequency))
    
    return samplingFrequency


def write_annots(annots, file, dir):
    
    annots_file = os.path.splitext(dir + "/" + file)[0] + "_annotations.csv"
    annots.to_csv(annots_file, index=False)


   
def add_dict_elem(hash, key, val):
    if not key in hash.keys():
        hash[key] = []
    
    hash[key].append(val)

