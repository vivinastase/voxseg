import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json

from datetime import datetime
import logging

import tensorflow as tf
from tensorflow.keras import models

from voxseg import extract_feats, run_cnnlstm, utils, evaluate

import sys
sys.path.append("../voxseg/")

import train_mat
import train

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU, quick enough for decoding
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
        
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10,inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(config=session_conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='Extracts features and run VAD to generate endpoints.')
    
    parser.add_argument('-c', '--config_file', type=str,
                        help='a path to a json file containing all the information about the pretrained model (including parameters).\nIf this option is provided, there is no need to specify the parameters listed below (safer to do that, so the correct parameter values are used).')
 
    parser.add_argument('-M', '--model_path', type=str,
                        help='a path to a trained vad model saved as in .h5 format, overrides default pretrained model')
    
    
    ## model parameters -- must be the same as those used when training the specified model -- best use the corresponding json config file
    parser.add_argument('-f', '--frame_length', type=int, default=16,
                        help='the frame length to consider (originally was defaulted to 0.32 -- smaller values have an impact on the CNN architecture -- the information gets compressed quickly and it cannot pass through 3 layers')
    parser.add_argument('-n', '--nfilt', type=int, default=128,
                        help='the number of filters to extract from the signal data -- will be one of the dimensions of the input to the CNN (nfilt x frame-length (as array length))')
    parser.add_argument('-l', '--winlen', type=float, default=0.025,
                        help='the window length parameter for extracting features from the signal data with logfbank')
    parser.add_argument('-w', '--winstep', type=float, default=0.01,
                        help='the window step parameter for extracting features with logfbank (which determines how much the windows from which features are extracted overlap)')
    
    ## classification parameters -- specify if different from the default values
    parser.add_argument('-t', '--speech_thresh', type=float, default=0.5,
                       help='a decision threshold value between (0,1) for speech vs non-speech, defaults to 0.5')
    parser.add_argument('-m', '--speech_w_music_thresh', type=float, default=0.5,
                       help='a decision threshold value between (0,1) for speech_with_music vs non-speech, defaults to 0.5, \
                       increasing will remove more speech_with_music, useful for downsteam ASR')
    parser.add_argument('-k', '--median_filter_kernel', type=int, default=1, 
                       help='a kernel size for a median filter to smooth the output labels, defaults to 1 (no smoothing)')
    parser.add_argument('-r', '--eval_res', default=0.01,
                         help="the resolution (time interval) of the evaluation. Suggested values: for human speech 0.01, for animal calls 0.004")
            
    ## evaluation information 
    parser.add_argument('-e', '--eval_dir', type=str,
                       help='a path to a Kaldi-style data directory containing the ground truth VAD segments for evaluation')
    
    ## test data and output dir information
    parser.add_argument('data_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', and optionally \'segments\'')
    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the output segments will be saved')
    
    args = parser.parse_args()

    config_info = dict()
    if args.config_file is not None:
        with open(args.config_file, "r") as json_file:
            config_info = json.load(json_file)
        params = {"frame_length": config_info["frame_length"], "nfilt": config_info["nfilt"], "winlen": config_info["winlen"], "winstep": config_info["winstep"]}
        model_name = os.path.splitext(config_info["model"])[0]
    else:
        params = {"frame_length": args.frame_length, "nfilt": args.nfilt, "winlen": args.winlen, "winstep": args.winstep}
        
    print("Config info: {}".format(config_info))

    if args.model_path is not None:
        model_name = os.path.splitext(args.model_path)[0]
        model = models.load_model(args.model_path)
    elif "model" in config_info.keys():
        model = models.load_model(config_info["model"])
    else:
        print("Model must be specified either on the command line, or through a json config file.\nExiting")
        sys.exit(0)

    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")
    logfile = model_name + "__main__" + timestamp + ".log"
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)  

    logging.info("_______\nmain.py {}\n__________\n".format(args))

    if utils.test_file_type(args.data_dir) == "mat":
        train_mat.test_model(model, args.data_dir, args.out_dir, params, speech_thresh=args.speech_thresh, res=args.eval_res)
    else:
        train.test_model(model, args.data_dir, args.out_dir, args.eval_dir, params, speech_thresh=args.speech_thresh, speech_w_music_thresh=args.speech_w_music_thresh, filt=args.median_filter_kernel, res=args.eval_res)
    

