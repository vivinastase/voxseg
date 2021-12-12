'''
Created on Dec 8, 2021

@author: vivi
'''


import argparse

import sys
sys.path.append("../voxseg/")

from data import utils

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('-files_dir', default="/home/vivi/Data/Segmentation/humans/shipibo_lena.jaico", help="Path to the dataset")    
    parser.add_argument('-files_dir', default="/home/vivi/Data/Segmentation/birds/R3428", help="Path to the dataset")

#    parser.add_argument('-data_path', default="/home/vivi/Data/Segmentation/classification", help="top level for classification data. each animal will have its own subdir, with the necessary mat files")    
    parser.add_argument('-data_path', default="/home/vivi/Data/Segmentation/voxseg/", help="top level for classification data. each animal will have its own subdir, with the necessary mat files")
    
    #parser.add_argument('-isTest', default=True, help="Taken into account when : whether the files to be processed will be a test file (in which case the segments file should only contain the positive instances -- what happens with the multi-class setting, though?)")
    
    parser.add_argument('-isStandAlone', default=False, help="Used when processing single directories, that do not have train and test variations")    
    #parser.add_argument('-wav_ext', default="WAV", help="Extension for the wav files (could be WAV or wav)")
    parser.add_argument('-wav_ext', default="wav", help="Extension for the wav files (could be WAV or wav)")
    
    parser.add_argument('-annot_ext', default="csv", help="Extension for the annotation files (usually txt or csv")
    
    parser.add_argument('-sep', default=",", help="field separator in the annotations file")
    
    args = parser.parse_args()

    args.files_dir = args.files_dir.rstrip("/")
    name = args.files_dir.split("/")[-1]
    
    if args.isStandAlone == "False":
        args.isStandAlone = False
    
    if "birds" in args.files_dir:  ##this is because this data is exported from flatclust, and has one annotations file for all recordings
        utils.make_voxseg_zf(args.files_dir, name, args.data_path, utils.get_annot_file(args.files_dir, args.annot_ext))
    else:
        utils.make_voxseg(args.files_dir, name, args.data_path, False, args.isStandAlone, args.wav_ext, args.annot_ext, args.sep)
        utils.make_voxseg(args.files_dir, name, args.data_path, True, args.isStandAlone, args.wav_ext, args.annot_ext, args.sep)
    
