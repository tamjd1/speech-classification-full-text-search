from sklearn.linear_model import LogisticRegression
from pydub.silence import split_on_silence
from pydub import AudioSegment
from python_speech_features import mfcc
import numpy as np
import glob
import sounddevice as sd
import pickle
from time import sleep
import os
import sys
f = open(os.devnull, 'w')
sys.stderr = f

MODEL_PICKLE_FILE_NAME = "model.pickle"
MIN_SILENCE_MS = 1000  # milliseconds of silence between spoken words in training and testing audio files
SILENCE_THRESHOLD_DB = 32  # decibels to consider quietness to eliminate ambient noise in the audio files


def prepare(play=False):
    """
    Prepare data for model training
    :param play: flag to play the audio that is being used for training
    :return: list of training data and labels
    """
    data = []

    # get all training audio file names
    audio_filenames = glob.glob("data/audio/*.wav")
    for fn in audio_filenames:
        # for each file split the audio into chunks each of which contain
        # one utterance of the spoken word (each file contains multiple utterances of the same word)
        print(fn)
        audio_file = AudioSegment.from_wav(fn)
        audio_chunks = split_on_silence(
            audio_file,
            min_silence_len=MIN_SILENCE_MS,
            silence_thresh=-SILENCE_THRESHOLD_DB
        )

        # determine the label name from the filename
        label_name = fn.split("/")[-1].split(".")[0]

        for i, audio in enumerate(audio_chunks):
            # for each chunk (utterance of the word)
            # convert audio to np array
            # retrieve its mfcc attributes
            # resize and reshape to maintain uniformity on the training data
            np_audio = np.frombuffer(audio.raw_data, np.int32)
            features = mfcc(np_audio, audio.frame_rate)
            features = features[:20, :]
            features = features.reshape(features.shape[0] * features.shape[1])
            print("{}: {}".format(i+1, label_name))

            if play:
                # play the audio file if the flag is set to True
                # for debugging purposes
                sd.play(np_audio, audio.frame_rate, blocking=True)

            # append label, mfcc features, and audio to the list of training data
            data.append({
                "label": label_name,
                "features": features,
                "audio": audio
            })

    return data


def train(data):
    """
    Once data pre-processing is done, it is ready for training
    :param data: pre-processed data for model training
    :return: None
    """
    # let X represent the mfcc features and y represent the labels
    X = np.array([_["features"] for _ in data])
    y = np.array([_["label"] for _ in data])

    # use X and y to train a multinomial Logistic Regression model
    # which will be used to classify several classes of spoken text
    model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    model.fit(X, y)

    # determine and print the score to verify quality of training
    score = model.score(X, y)
    print(score)

    # pickle and save the model so it can be used in the future
    # without having to retrain
    pickle.dump(model, open(MODEL_PICKLE_FILE_NAME, 'wb'))


def predict(test_audio, play=False):
    """
    This function uses the trained model to predict the words spoken in audio files
    :param test_audio: audio file containing the text to predict
    :param play: flag to play the audio that is being used for prediction
    :return: string of predicted word(s)
    """
    # load the trained model from pickled file
    model = pickle.load(open(MODEL_PICKLE_FILE_NAME, 'rb'))

    predicted = []

    # in case there are multiple words, split them on silence between words
    # to create an audio file for each word
    audio_chunks = split_on_silence(
        test_audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=-SILENCE_THRESHOLD_DB
    )

    for audio in audio_chunks:
        # for each audio chunk (word)
        # convert audio to np array
        # retrieve its mfcc attributes
        # resize to maintain uniformity with the training data
        np_audio = np.frombuffer(audio.raw_data, np.int32)
        features = mfcc(np_audio, audio.frame_rate)
        features = features[:20, :]

        # let X represent the flattened mfcc features
        X = features.reshape(features.shape[0] * features.shape[1])

        # use X to predict spoken word
        _y = model.predict(X)

        if play:
            # play the audio file and print the predicted word
            # if the flag is set to True
            # for debugging purposes
            print("Predicted word: {}".format(_y))
            sd.play(np_audio, audio.frame_rate, blocking=True)
            sleep(1)

        # append it to the predicted words list
        predicted.append(_y[0])

    # convert predicted words list to string and return
    return " ".join(predicted)
