'''
Created on Jan 14, 2022

@author: vivi
'''


import os

import re

import pandas as pd

import numpy as np


headers = {'csv': {'start' : 'Start', 'duration': 'Duration', 'call_type': 'Name'},
           'txt': {'start': 0, 'end': 1, 'call_type': 2},
           'zf': {'start': 'onset', 'duration': 'duration', 'call_type': 'cluster_id'}
          }


def get_annotations(file, ext, sep):
        ## write the begin and end of segments -- these must include the "negative segments" as well
      
    print("processing annotations file {} (ext={}, sep={})".format(file, ext, sep))  
    if os.path.isfile(file):            
        if ext == "txt":
            annots = pd.read_csv(file, sep=sep, header = None)
            annots.columns = ['start', 'end', 'call_type']
        else:
#            print("Reading csv annotations: ")
            annots = pd.read_csv(file, sep=sep)
#            print("header: {}".format(annots.columns))
            annots = annots.rename(columns={'name':'call_type'})            
    else:
        annots = pd.DataFrame(columns = ['start', 'end', 'call_type'])
        
    return merge_overlaps(annots)
    #return annots
 
 
def merge_overlaps(annots):

#    print("annotation columns: {}".format(annots.head()))
    annots.sort_values('start', inplace=True)

    starts = []
    ends = []
    call_types = []
    
    for _, row in annots.iterrows():
        start = row['start']
        end = row['end']
        call_type = row['call_type']
        
        if (len(starts) == 0) or (start > ends[-1]):
            starts.append(start)
            ends.append(end)
            call_types.append(call_type)
        else:
            ends[-1] = end
            call_types[-1] = call_type
                
    new_annots = {'start': starts, 'end': ends, 'call_type': call_types}
    return pd.DataFrame(new_annots)
    

def get_labels(N, samplingFrequency, annots, file_stem, annot_ext):

    if isinstance(annots, pd.DataFrame):
        annots = annots[annots['file'] == file_stem]
        samplingFrequency = 1 ## because the data exported from flatclust is already scaled
        annots['start'] = annots['onset']
        annots['end'] = annots['start']+annots['duration']
    else:
        annot_file = file_stem + "." + annot_ext
        annots = get_annotations(annot_file, annot_ext, "\t")
                     
    return transform_annotations(N, annots, samplingFrequency)


    
def transform_annotations(N, annots, samplingFrequency):
    
    print("transforming annotations ({})".format(N))
    
    labels = np.zeros(N)
    
    for _id, row in annots.iterrows():           
        begin = convert_time(row['start'])
        end = convert_time(row['end'])
        
        for i in range(int((begin-1)*samplingFrequency),int((end-1)*samplingFrequency)):   ##+1):
            labels[i] = 1
            
    return labels



def get_annot_info(annots_row):

    return (annots_row['start'], annots_row['end'], "speech")



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


if __name__ == '__main__':
    pass