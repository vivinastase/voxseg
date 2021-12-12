'''
Created on Dec 8, 2021

@author: vivi
'''

import argparse
import os
import sys
sys.path.append("../voxseg/")

import pandas as pd

from data import utils


'''

    reformat the predictions of voxseg to the annotation format
    
    The segments file contains only the predicted speech segments
    The time format of the prediction is in seconds, the annotation time formal is decimal (min:sec.subsec)
    
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
   
   ALSO CONVERT TO SIMPLE .txt: start<tab>end<tab>call_type
    
'''
def reformat(data_path):

    (annots, files) = read_annotations(data_path)
        
    for rec_id in annots.keys():
        f = open(data_path + "/" + files[rec_id][0] + ".prediction", 'w')
        f_simple = open(data_path + "/" + files[rec_id][0] + ".txt", 'w')
                
        i = 1
        f.write("Name\tStart\tDuration\tTime Format\tType\tDescription\n")
        for (start_dec, duration_dec, _end_dec, start, end) in annots[rec_id]:
            f.write("Marker {}\t{}\t{}\tdecimal\tCue\t\n".format(i, start_dec, duration_dec))
            i += 1
            f_simple.write("{}\t{}\t{}\n".format(start, end, i))
         
        f_simple.close()
        f.close()
        



def read_annotations(data_path):
    
    wav_scp = get_data(data_path + "/wav.scp", " ")
    segments = get_data(data_path + "/segments", " ")
    
    files = { rec_id: (os.path.basename(file), utils.getSamplingRate(file)) for _id, (rec_id, file) in wav_scp.iterrows() }

    annots = dict()
    for _id, (_utt_id, rec_id, start, end) in segments.iterrows():
        add_element(annots, rec_id, start, end)
     
    return (annots, files)

   
   
def overlap(a,b,x,y):
    return not ((b < x) or (a > y)) 
        
'''
    convert the time stamp to decimal format (min:sec.subsec)
    and include the duration as well as the end (meerkats data uses duration, but keep the end just in case)
'''
def add_element(annots, rec_id, start, end):
    
    if not rec_id in annots.keys():
        annots[rec_id] = []
        
    duration = end - start
    
    start_dec = convert_time_to_decimal(start)
    duration_dec = convert_time_to_decimal(duration)
    end_dec = convert_time_to_decimal(end)
    
    annots[rec_id].append((start_dec, duration_dec, end_dec, start, end))
    
    
def convert_time_to_decimal(time):
    
    mins = int(time/60)
    secs = time - mins * 60
    
    return "{:d}:{:09.6f}".format(mins, secs)
    
    
def get_data(file, delim):
    
    if os.path.exists(file):
        return pd.read_csv(file, delimiter=delim, header=None)
    
    print("Input file {} could not be found\nexiting ... ".format(file))
    sys.exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path',  default="/home/vivi/Data/Segmentation/voxseg/shipibo_lena.johani.test.predictions")

    args = parser.parse_args()
    
    reformat(args.data_path)
    
