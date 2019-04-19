# speech-classification-full-text-search

### Description

This is an example application created to simulate a voice enabled search of popular search engines such as Google, Bing, and others.



### Prerequisites
- `Python v3.6+`
    - this application is written in `Python 3.6.0` and uses libraries compatible with this version
- `scikit-learn v0.18.1`
    - `sklearn` is used for training a speech classification algorithm and to parse documents to retrieve their relevancy scores for full text search
- `nltk v3.4`
    - `nltk` is used for documents pre-processing to generate their relevancy scores with accuracy 
- `numpy v1.16.2`
    - `numpy` is used to parse audio files to multi-dimensional numeric array for training and processing
- `pydub v0.23.1`
    - audio files are read in and parsed and split using `pydub` for pre-processing and training
- `python-speech-features v0.6`
    - function for extraction of audio features for training the classification model is provided by `python_speech_features`  
- `sounddevice v0.3.13`
    - `sounddevice` is used to play audio file or a chunk of audio for debugging purposes

The dependant libraries are listed in the `./requirements.txt` file and can be installed using `pip` via the following command:
```buildoutcfg
pip install -r requirements.txt
```

### Training and Testing Data
This application makes use of two kinds of data sets:
1. audio files containing spoken words containing data for training and testing purposes saved in `./audio` and `./audio/test` directories, respectively.
    
    - There are also files containing metadata of words spoken in audio files used for training and testing saved in `./data/training_words.txt` and `./data/testing_words.txt`, respectively.
2. a corpus of text documents containing excerpts from various books and quotations along with their authors to use for full text search saved in `./data/corpus.txt`
 

### Getting Started
