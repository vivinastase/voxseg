# Voxseg

This is a parameterized version of Voxseg, whose original version is here:

https://github.com/NickWilkinson37/voxseg.git

Voxseg is a Python package for voice activity detection (VAD), for speech/non-speech audio segmentation, and it is based on work presented [here](https://arxiv.org/abs/2103.03529).

## Installation

These are the installation instructions from the original, they should work for this version as well:

To install this package, clone the repository from GitHub to a directory of your choice and install using pip:
```bash
git clone [this repo]
pip install ./voxseg
```
In future, installation directly from the package manager will be supported.

To test the installation run:
```bash
cd voxseg
python -m unittest
```
The test will run the full VAD pipeline on two example audio files. This pipeline includes feature extraction, VAD and evaluation. The output should include the following*:
- A progress bar monitoring the feature extraction process, followed by a DataFrame containing normalized-features and metadata.
- A DataFrame containing model generated endpoints, indicating the starts and ends of discovered speech utterances.
- A confusion matrix of speech vs non-speech, with the following values: TPR 0.935, FPR 0.137, FNR 0.065, FPR 0.863

*The order in which these outputs appear may vary.

## Data Preparation

### Input Data Preparation

The script "reformat4voxseg.py" in the data package will convert the wav files and their annotations to the Voxseg format described below. There are several assumptions about the data, so this may need to be changed.

Parameters:

* -files_dir: the path to the dataset (if there are test and train versions, then the path to the common stem)
* -data_dir: the path to the directory where the results of the processing will be put
* -isTest: whether the data to be processed will be used only for prediction (the segments and utt2spk files will only contain speech segments -- for evaluation, if wanted)
* -isStandAlone: I usually process pairs of directories (for train and test), so set this to True if there is a "one off" directory to be processed
* -wav_ext: extension of the recordings file (could be wav or WAV)
* -annot_ext: extension of the annotations file (could be txt or csv) -- the assumptions about their headers and such is hard coded
* -sep: field separator for the annotations file

Example of usages:

```
python3.8 data/reformat4voxseg.py -files_dir ~/Data/Segmentation/marmosets/garetta_raw -data_path ~/Data/Segmentation/voxseg/ -wav_ext WAV -annot_ext txt -isStandAlone False -sep "\t"
```

This will process the garetta_raw.train and garetta_raw.test directories in the ~/Data/Segmentation/marmosets/ directory, and make two parallel directories under ~/Data/Segmentation/voxseg


The files used as input for Voxseg (that could be generated as described above)  are the same as those used by the Kaldi toolkit. Extensive documentation on the data preparation process for Kaldi may be found [here](https://kaldi-asr.org/doc/data_prep.html). Only the files required by the Voxseg toolkit are described here.

1. `wav.scp` - this file provides the paths to the audio files one wishes to process, and assigns them a unique recording-id. It is structured as follows:
    `<recording-id> <extended-filename>`. Each entry should appear on a new line, for example:
    ```
    rec_000 wavs/some_raw_audio.wav
    rec_001 wavs/some_more_raw_audio.wav
    rec_002 wavs/yet_more_raw_audio.wav
    ```
    Note that the `<extended-filename>` may be an absolute path or relative path, except when using Docker or Singularity, where paths relative to the mount point must be used.

2. `segments` - this file is optional and specifies segments within the audio file to be processed by the VAD (useful if one only wants to run the VAD on a subset of the full audio files). If this file is not present the full audio files will be processed. This file is structured as follows:
    `<utterance-id> <recording-id> <segment-begin> <segment-end>`, where `<segment-begin>` and `<segment-end>` are in seconds. Each entry should appear on a new line, for example:
    ```
    rec_000_part_1 rec_000 20.5 142.6
    rec_000_part_2 rec_000 362.1 421.0
    rec_001_part_1 rec_001 45.9 89.4
    rec_001_part_2 rec_001 97.7 130.0
    rec_001_part_3 rec_001 186.9 241.8
    rec_002_full rec_002 0.0 350.0
    ```

These two files should be placed in the same directory, usually named `data`, however you may give it any name. This is the directory that is provided as input to voxseg’s feature extraction.

### Output Data Preparation

Voxseg produces output files in the same format as the input. The script "voxseg2txt.py" in the data package converts voxseg's output to a simple txt format (that can be loaded into audacity for example). This output -- one file for each wav file mentioned in the wav.scp file -- will be produced in the same directory. 

Parameters:
* -data_path: the path to the output of Voxseg

Example usage:

```
python3.8 data/voxseg2txt.py -data_path ~/Data/Segmentation/voxseg/garetta_raw.test.predictions/
```


## Usage

There are a number of additional parameters that must be specified for each step of the VAD pipeline.

### Train model (and also test, if wanted)

To run the full VAD pipeline -- train, test -- use the following command in the top level of the code repository (voxseg):
```
python3.8 train.py [params] train_data_directory 
```

or 

```
python3.8 train_mat.py [params] train_data_directory model_name out_dir
```

for mat files (of particular format ...).


An example for running on the shipibo data:

```
python3.8 train.py -s 0.1 -n 128 -f 32 -t [test_directory] -e [eval_directory] [train_directory] [model_name] [output_directory]
```

This would train a model on the data in the train_directory, write the model into the [output_directory]/[model_name].h5 file.
It saves all the information, including trained model and its parameters, into a json file named: [outdir]/[model_name].json
(When testing a pretrained model, provide this config file as a parameter.)

It would evaluate the performance on the data in the eval_directory, and predict annotations for the test data.

It will build an input to the CNN consisting of 32 (frame length) vectors of dimension 128 (nfilt). 128 is the numbe rof filters used by logfbank. The length of the segment of the signal at which it looks to build this representation is computed based on the frame length, and the winlen and winstep parameters (default values 0.025 and 0.01 for human speech, but they can be explicitly given)
It uses 0.1 of the training data as development (validation) data.

To explore the available flags for changing settings call for either train.py or train_mat.py:
```
python3.8 train.py -h
```

* positional arguments:

**  train_dir             a path to a Kaldi-style data directory containting 'wav.scp', 'utt2spk' and 'segments'
**  model_name            a filename for the model, the model will be saved as <model_name>.h5 in the output directory
**  out_dir               a path to an output directory where the model will be saved as <model_name>.h5, and where the log of the training will be saved as well

*optional arguments:

** -v VALIDATION_DIR, --validation_dir VALIDATION_DIR: a path to a Kaldi-style data directory containting 'wav.scp', 'utt2spk' and 'segments'
** -s VALIDATION_SPLIT, --validation_split VALIDATION_SPLIT: a percetage of the training data to be used as a validation set, if an explicit validation set is not defined using -v
** -f FRAME_LENGTH, --frame_length FRAME_LENGTH: the frame length to consider (originally was defaulted to 0.32 -- smaller values have an impact on the CNN architecture -- the information gets compressed quickly and it cannot pass through 3 layers
** -n NFILT, --nfilt NFILT: the number of filters to extract from the signal data -- will be one of the dimensions of the input to the CNN (nfilt x frame-length (as array length))
** -l WINLEN, --winlen WINLEN: the window length parameter for extracting features from the signal data with logfbank
** -w WINSTEP, --winstep WINSTEP: the window step parameter for extracting features with logfbank (which determines how much the windows from which features are extracted overlap)
** -t TEST_DIR, --test_dir TEST_DIR: the directory with the test data, if wanted: this will be the data for which automatic annotations will be made. It is not used for evaluating the goodness of the model!
** -e EVAL_DIR, --eval_dir EVAL_DIR: the directory with the evaluation data, if evaluation is wanted. It could be the same as the test_dir, if that directory contains gold-standard annotations


### Run an existing model on new data

(This used to be the script for running the entire pipeline, but somehow I do that with the train script, and this is left for applying an already built model)

Run a model using a config file for a pretrained model:
```
python3.8 voxseg/main.py -c [config_file (.json)] -e [eval_directory]  [test_directory] [predictions_directory]
```

Run a model by explicitly providing the model to use and parameters (the parameters must be the same as for training the model):
```
python3.8 voxseg/main.py -n 128 -f 32 -t 0.2 -M [model (.h5)] -e [eval_directory] [test_directory] [predictions_directory]
```

This will load the model from the given model file (...h5), and produce automatic annotations on the data in the test_directory, and write the predictions to the predictions_directory. The classification threshold is 0.2, and the model will build features of size 128 for a frame length of 32.

The evaluation and test data must be the same, but the evaluation directory should contain the same information as a train data (i.e. segments, utt2spk), while the test data must contain only wav.scp, so it knows what files to process. 

To explore the available flags for changing settings call for either train.py or train_mat.py:
```
python3.8 voxseg/main.py -h
```

* positional arguments:
**  data_dir              the test directory: a path to a Kaldi-style data directory containting 'wav.scp', and optionally 'segments'
**  out_dir               the predictions directory: a path to an output directory where the output segments will be saved

* optional arguments:
**  -h, --help            show this help message and exit
**  -c CONFIG_FILE, --config_file CONFIG_FILE
                        a path to a json file containing all the information about the pretrained model (including parameters). If
                        this option is provided, there is no need to specify the parameters listed below (safer to do that, so the
                        correct parameter values are used).
                        
  If the config file is provided, the next 5 arguments are not necessary (actually, should not be provided, because the frame length, nfilt, winlen and winstep parameters are linked to the trained model)

**  -M MODEL_PATH, --model_path MODEL_PATH: a path to a trained vad model saved as in .h5 format, overrides default pretrained model
**  -f FRAME_LENGTH, --frame_length FRAME_LENGTH: the frame length to consider (originally was defaulted to 0.32 -- smaller values have an impact on the CNN architecture -- the information gets compressed quickly and it cannot pass through 3 layers
**  -n NFILT, --nfilt NFILT: the number of filters to extract from the signal data -- will be one of the dimensions of the input to the CNN (nfilt x frame-length (as array length))
**  -l WINLEN, --winlen WINLEN: the window length parameter for extracting features from the signal data with logfbank
**  -w WINSTEP, --winstep WINSTEP: the window step parameter for extracting features with logfbank (which determines how much the windows from which features are extracted overlap)
**  -t SPEECH_THRESH, --speech_thresh SPEECH_THRESH: a decision threshold value between (0,1) for speech vs non-speech, defaults to 0.5
**  -m SPEECH_W_MUSIC_THRESH, --speech_w_music_thresh SPEECH_W_MUSIC_THRESH: a decision threshold value between (0,1) for speech_with_music vs non-speech, defaults to 0.5, increasing will remove more speech_with_music, useful for downsteam ASR
**  -k MEDIAN_FILTER_KERNEL, --median_filter_kernel MEDIAN_FILTER_KERNEL: a kernel size for a median filter to smooth the output labels, defaults to 1 (no smoothing)
**  -r EVAL_RES, --eval_res EVAL_RES: the resolution (time interval) of the evaluation. Suggested values: for human speech 0.01, for animal calls 0.004
                        
Argument necessary only if evaluation is wanted.
** -e EVAL_DIR, --eval_dir EVAL_DIR: a path to a Kaldi-style data directory containing the ground truth VAD segments for evaluation


## License
[MIT](https://choosealicense.com/licenses/mit/)
