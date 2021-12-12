'''
Created on Dec 8, 2021

@author: vivi
'''

import re
import os
import glob

import collections

import numpy as np

from scipy import signal
from scipy.io import wavfile 

import pandas as pd

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
def make_voxseg(files_dir, name, data_path, isTest, isStandAlone, wav_ext, annot_ext, sep = "\t"):
    
    if isTest:
            if not isStandAlone:
                files_dir += ".test/"
            name += ".test"
    else:
            if not isStandAlone:
                files_dir += ".train/" 
            name += ".train"
     
    data_path += "/" + name + "/"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    wav_scp  = open(data_path + "/wav.scp", "w")
    segments = open(data_path + "/segments", "w")
    utt2spk  = open(data_path + "/utt2spk", "w") 
    
    print("Reformatting data for voxseg ...")
    print("\toutputs will go to:\n\t{}\n\t{}\n\t{}\n".format(wav_scp, segments, utt2spk))
    
    n = 0
        
    print("\nReading information from directory {}".format(files_dir))
        
    for f in glob.glob(files_dir + '/*.' + wav_ext):
        file_stem = os.path.splitext(f)[0]
        wav_file = file_stem + "." + wav_ext
        annot_file = file_stem + "." + annot_ext

        rec_id = make_id("rec", n)
        wav_scp.write("{} {}\n".format(rec_id,wav_file))
        n += 1
        
        print("\n\nProcessing file {}\n".format(wav_file))
        
        ## write the begin and end of segments -- these must include the "negative segments" as well
        if annot_ext == "txt":
            annots = pd.read_csv(annot_file, sep=sep, header = None)
        else:
            annots = pd.read_csv(annot_file, sep=sep)
        
        process_file4voxseg(annots, segments, utt2spk, isTest, rec_id)
                    
    wav_scp.close()
    segments.close()
    utt2spk.close()
    
    
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
    
    annots.sort_values('start', inplace=True) 
    n_utt = 0
    prev_end = 0
    for k, row in annots.iterrows():  
        
        (begin, end, call_type) = get_annot_info(row)
        
        #print("processing segment: {}\t{}\t{}".format(begin, end, call_type))
        
        if is_acceptable(call_type, end, prev_end):  
        
            begin = max(begin, prev_end)
        
            if not isTest and begin > prev_end:
                add_instance(rec_id, n_utt, segments, utt2spk, prev_end, begin, "non_speech")
                n_utt +=1
                
            add_instance(rec_id, n_utt, segments, utt2spk, begin, end, "speech")   ##call_type)
            prev_end = end

            n_utt += 1


def get_annot_info(annots_row):

    if 'start' in annots_row.keys():
        return (annots_row['start'], annots_row['end'], "speech")
    
    if len(annots_row) == 3:
        return annots_row
        
    return (annots_row[2], annots_row[3], "speech")


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
    
    headers = {'csv': {'start' : 'Start', 'duration': 'Duration', 'call_type': 'Name'},
               'txt': {'start': 0, 'end': 1, 'call_type': 2},
               'zf': {'start': 'onset', 'duration': 'duration', 'call_type': 'cluster_id'}
               }
    
    annot_type = annot_ext
    if isinstance(annots, pd.DataFrame):
        annots = annots[annots['file'] == file_stem]
        samplingFrequency = 1 ## because the data exported from flatclust is already scaled
        annot_type = "zf"
    else:
        annot_file = file_stem + "." + annot_ext
        if annot_ext == "txt":
            annots = pd.read_csv(annot_file, sep="\t", header = None)
        else:
            annots = pd.read_csv(annot_file, sep="\t")
                     
    labels = transform_annotations(spectrogram.shape[1], annots, spectrogram.shape[1]/len(signalData), headers[annot_type])

    return(spectrogram, labels)




  
# when the signal is transformed with the log (in the make_mat_files function), smooth the 0 values, 
def smooth(np_array):
    
    print("frequencies array: {}".format(np_array))
    min_val = np.amin(np_array[np.nonzero(np_array)])/2  ## make the min smaller than the min value in the array
    print("min val in spectrogram for smoothing: {}".format(min_val))

    ## if the value is negative, it means I won't log the spectrogram, so no smoothing necessary
    if min_val > 0:
        np_array += min_val
       
    return np_array
      
    
    
def transform_annotations(N, array, samplingFrequency, headers):
    
    print("transforming annotations ({})".format(N))
    
    labels = np.zeros(N)
    for id, row in array.iterrows():
           
        begin = convert_time(row[headers['start']])
        if 'end' in headers.keys():
            end = convert_time(row[headers['end']])
        else:
            end = begin + convert_time(row[headers['duration']])
        
        for i in range(int((begin-1)*samplingFrequency),int((end-1)*samplingFrequency)):   ##+1):
            labels[i] = 1
            
    return labels



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


def get_time(start, duration): 
  
    begin = convert_time(start)
    end = begin + convert_time(duration)        
  
    return (begin,end)


def convert_time(time_str):

    if int(time_str):
        return int(time_str)
    
    if float(time_str):
        return float(time_str)
    
    m = re.match(r'^(\d+)\:(\d+)\.(\d+)$', time_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) * pow(10,(-1)*len(m.group(3)))
    
    
    print("I don't recognize the file format: {}".format(time_str))
    return 0



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
        print("oo many potential annotation files ... don't know how to choose\nStopping.")
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
    print("Sampling rate of {} = {}".format(wav_file, samplingFrequency))
    
    return samplingFrequency


def write_annots(annots, file, dir):
    
    annots_file = os.path.splitext(dir + "/" + file)[0] + "_annotations.csv"
    annots.to_csv(annots_file, index=False)



   
def add_dict_elem(hash, key, val):
    if not key in hash.keys():
        hash[key] = []
    
    hash[key].append(val)

