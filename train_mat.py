# Script for training custom VAD model for the voxseg toolkit

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json

import logging
from datetime import datetime

#import voxseg_.prep_labels
#import voxseg_.evaluate
#import voxseg_.extract_feats
#import voxseg_.run_cnnlstm
#import voxseg_.utils

import prep_labels, evaluate, extract_feats, run_cnnlstm, utils

import numpy as np
import pandas as pd
import argparse


import tensorflow as tf

from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU, quick enough for decoding
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

#session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10,inter_op_parallelism_threads=10)
#sess = tf.compat.v1.Session(config=session_conf)
'''

#tf.debugging.set_log_device_placement(True)


# Define Model

# Define Model

class CNN2LSTM(models.Sequential):
    
    def __init__(self, output_layer_width, params):

        super().__init__()
        ## parameters for building the features -- included in the model so they can be used for the testing scenario
        self.params = params
        frame_length = self.params['frame_length']
        
        d = min(5, frame_length-1)  ## for very small frame lengths, a 5x5 filter in the first layer is too big
        
        self.model = models.Sequential()
        self.model.add(layers.TimeDistributed(layers.Conv2D(64, (d, d), activation='elu'), input_shape=(None, self.params['nfilt'], frame_length, 1)))
        self.model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))

        ## this is to adjust the architecture, because for lower frame lengths the signal cannot pass through multiple conv. layers        
        if frame_length >= 16:
            self.model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
            self.model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
        
        if frame_length >= 32:
            self.model.add(layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='elu')))
            self.model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
        
        self.model.add(layers.TimeDistributed(layers.Flatten()))
        self.model.add(layers.TimeDistributed(layers.Dense(128, activation='elu')))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.TimeDistributed(layers.Dense(output_layer_width, activation='softmax')))
        #self.compile(
        self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=[   
                             'accuracy'
                             #tf.keras.metrics.BinaryAccuracy()
                             #, tfa.metrics.F1Score(num_classes=2, threshold=0.5)
                             , tf.keras.metrics.Recall()
                             , tf.keras.metrics.AUC(num_thresholds=10)
                             ])

# Define training parameters
def train_model(model, x_train, y_train, validation_split, x_dev=None, y_dev=None, epochs=25, batch_size=64, callbacks=None):
    if validation_split:
        return model.fit(x_train[:,:,:,:,np.newaxis], y_train, validation_split = validation_split,
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    elif x_dev is not None:
        return model.fit(x_train[:,:,:,:,np.newaxis], y_train,
                     validation_data=(x_dev[:,:,:,:,np.newaxis], y_dev),
                     epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    else:
        print('WARNING: no validation data, or validation split provided.')
        return model.fit(x_train[:,:,:,:,np.newaxis], y_train,
                     epochs=epochs, batch_size=batch_size)



def test_model(model, data_dir, out_dir, params, speech_thresh=0.5, speech_w_music_thresh=0.5, filt=1):
    
    logging.info("testing model\nparams: {}".format(locals()))
    
    eval_dir = data_dir
    
    (data_test, _nfilt, n_columns) = voxseg.utils.load_mat_data(eval_dir, "test")
    (data_eval, _nfilt, n_columns) = voxseg.utils.load_mat_data(eval_dir, "eval")

    feats = voxseg.extract_feats.extract_from_mat(data_test, params)
    feats = voxseg.extract_feats.normalize(feats)
        
    (interval_length, winstep) = voxseg.utils.get_mat_interval_length(params) ## to make overlapping frames 
    logging.info("interval length = {} / winstep = {}".format(interval_length, winstep))
    
    targets = voxseg.run_cnnlstm.predict_targets(model, feats)

    voxseg.evaluate.score_mat(targets, data_eval, n_columns, interval_length, winstep, speech_thresh)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CHPC_VAD_train.py',
                                     description='Train an instance of the voxseg VAD model.')

    parser.add_argument('-v', '--validation_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'utt2spk\' and \'segments\'')

    parser.add_argument('-s', '--validation_split', type=float,
                        help='a percetage of the training data to be used as a validation set, if an explicit validation \
                              set is not defined using -v')
    
    parser.add_argument('-f', '--frame_length', type=int, default=32,
                        help='the frame length to consider (originally was defaulted to 0.32 -- smaller values have an impact on the CNN architecture -- the information gets compressed quickly and it cannot pass through 3 layers')
 
# cannot specify these anymore, because mat data has fixed dimensions   
#    parser.add_argument('-n', '--nfilt', type=int, default=32,
#                        help='the number of filters to extract from the signal data -- will be one of the dimensions of the input to the CNN (nfilt x frame-length (as array length))')


    parser.add_argument('-l', '--winlen', type=float, default=0.025,
                        help='the window length parameter for extracting features from the signal data with logfbank')

    parser.add_argument('-w', '--winstep', type=float, default=0.01,
                        help='the window step parameter for extracting features with logfbank (which determines how much the windows from which features are extracted overlap)')

    parser.add_argument('-c', '--config_file', type=str,
                        help='a json file containing all the information to train a model, except training directory, model name and output directory')

    parser.add_argument('-t', '--test_dir', default='', 
                        help='the directory with the test data, if testing is wanted')

    parser.add_argument('train_dir', type=str,
                        help='a path to a Kaldi-style data directory containting \'wav.scp\', \'utt2spk\' and \'segments\'')

    parser.add_argument('model_name', type=str,
                        help='a filename for the model, the model will be saved as <model_name>.h5 in the output directory')

    parser.add_argument('out_dir', type=str,
                        help='a path to an output directory where the model will be saved as <model_name>.h5')

    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")
    logfile = args.out_dir + "/" + args.model_name + "__train_mat__" + timestamp + ".log"
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)  

    logging.info("_______\ntrain_mat.py {}\n__________\n".format(args))

    model_file_name = ''
    
    if not args.config_file :
        ## save parameters and information to json file, to be used when testing using a pretrained model
        model_file_name = f'{args.out_dir}/{args.model_name}.h5'

        config_file = f'{args.out_dir}/{args.model_name}.json'
        config_info = dict()
        config_info['model'] = model_file_name
        config_info['logfile'] = logfile
        for arg in vars(args):
            config_info[arg] = getattr(args, arg)
            
        with open(config_file, "w") as json_file:
            json.dump(config_info, json_file)
    else:
        with open(args.config_file, "r") as json_file:
            config_info = json.load(json_file)
        model_file_name = config_info['model']
    
    
    (data_train, nfilt, _) = voxseg.utils.load_mat_data(args.train_dir, "train")
    if config_info['validation_dir']:
        (data_dev, _, _) = voxseg.utils.load_mat_data(config_info['validation_dir'], "dev")
        
    params = {"frame_length": args.frame_length, "nfilt": nfilt, "winlen": args.winlen, "winstep": args.winstep}


    # Extract features
    feats_train = voxseg.extract_feats.extract_from_mat(data_train, params)
    feats_train = voxseg.extract_feats.normalize(feats_train)
    
    if config_info['validation_dir']:
        feats_dev = voxseg.extract_feats.extract_from_mat(data_dev, params)
        feats_dev = voxseg.extract_feats.normalize(feats_dev)

    # Extract labels
    labels_train = voxseg.prep_labels.get_mat_labels(data_train, params)
    labels_train['labels'] = voxseg.prep_labels.one_hot(labels_train['labels'])
    if args.validation_dir:
        labels_dev = voxseg.prep_labels.get_mat_labels(data_dev, params)
        labels_dev['labels'] = voxseg.prep_labels.one_hot(labels_dev['labels'])


    seq_len = config_info['frame_length'] * 2

    # Train model
    X = voxseg.utils.time_distribute(np.vstack(feats_train['normalized-features']), seq_len)
    y = voxseg.utils.time_distribute(np.vstack(labels_train['labels']), seq_len)
    if config_info['validation_dir']:
        X_dev = voxseg.utils.time_distribute(np.vstack(feats_dev['normalized-features']), seq_len)
        y_dev = voxseg.utils.time_distribute(np.vstack(labels_dev['labels']), seq_len)
    else:
        X_dev = None
        y_dev = None
                
    checkpoint = ModelCheckpoint(filepath=model_file_name,
                                 save_weights_only=False,
                                 #monitor='val_loss', 
                                 #mode='min', 
                                 #monitor='val_recall',
                                 #mode='max',
                                 monitor='val_auc',
                                 mode='max',
                                 save_best_only=True)

    voxseg_model = CNN2LSTM(y.shape[-1], params)
    
    if y.shape[-1] == 2 or y.shape[-1] == 4:
        #hist = train_model(cnn_bilstm(y.shape[-1], args.frame_length, nfilt), X, y, args.validation_split, X_dev, y_dev, callbacks=[checkpoint])
        hist = train_model(voxseg_model.model, X, y, config_info['validation_split'], X_dev, y_dev, callbacks=[checkpoint])

        df = pd.DataFrame(hist.history)
        df.index.name = 'epoch'
        #df.to_csv(f'{args.out_dir}/{args.model_name}_training_log.csv')
        logging.info("Training results:\n{}".format(df))
    else:
        print(f'ERROR: Number of classes {y.shape[-1]} is not equal to 2 or 4, see README for more info on using this training script.')
        
    
    if config_info['test_dir'] != '':
        pred_dir = config_info['test_dir'].rstrip('/') + '.predictions/'
        test_model(voxseg_model.model, config_info['test_dir'], pred_dir, params, speech_thresh=0.5)
            
