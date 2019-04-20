# Voice Enabled Full Text Search

### Description

This is an application which simulates a voice enabled search engine; similar to popular search engines such as Google, Bing, and others, to retrieved documents which are relevant to the words which the user has spoken.

It involves a speech classification model which needs to be trained using the training data set the first time the application runs. Once the model is trained, it can be used to predict words spoken in testing data set which are used to run full text search on a collection of documents in the corpus.

The algorithm that is used for speech classification is Logistic Regression classifier and for full text search analysis Term Frequency/Inverse Document Frequency (TF/IDF) scores are used to find the relevant documents.   


### Training and Testing Data
This application makes use of two kinds of data sets:
1. audio files containing spoken words containing data for training and testing purposes saved in `./audio` and `./audio/test` directories, respectively.
    
    - There are also files containing metadata of words spoken in audio files used for training and testing saved in `./data/training_words.txt` and `./data/testing_words.txt`, respectively.
2. a corpus of text documents containing excerpts from various books and quotations along with their authors to use for full text search saved in `./data/corpus.txt`


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


### Getting Started
After the `Python` environment has been set up and all the dependencies have been installed the application can run. 

The entrypoint of the application is the `main()` function in the script called `voice_enabled_search.py`. To execute the script run the following command in the terminal:

```buildoutcfg
python voice_enabled_search.py
```  

If using an IDE, such as PyCharm, the script can also be executed within the IDE.

The output will be sent to `stdout` and will be visible in the terminal if run in the CLI or within the IDE if run in the IDE.

_Note:_ the first time the application is executed, the `train` flag in the `main()` method needs to be set to `True` in order for the speech classification algorithm to be trained and be ready for predictions.


### Program Flow

The application process is as follows:
1. If the `train` flag is set to `True` the application will use the training audio files in `./data/audio` directory to train the Logistic Regression classifier for speech classification and pickle and save the trained model for future use.
2. Next the `./data/audio/test` directory is traversed and testing audio files are used to do the following:
    1. split the audio into objects of individual spoken words
    2. the spoken words are then sent to the classifier for prediction
    3. the text of the spoken words is retrieved 
3. The predicted text is then sent to the search algorithm to do full text search on the corpus to retrieve the relevant documents; the steps of which are as follows:
    1. calculate the TF/IDF scores of the corpus
    2. sort the scores in descending order so the highest scores are at the top
    3. return the documents with the highest scores, whose scores are greater than zero.
4. The returned documents are then looped through and printed to `stdout` along with the intended and predicted search terms.