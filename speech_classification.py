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
MIN_SILENCE_MS = 1000  # milliseconds
SILENCE_THRESHOLD_DB = 32  # decibels


def prepare(play=False):
    training_data = []

    audio_filenames = glob.glob("data/audio/*.wav")
    for fn in audio_filenames:
        print(fn)
        audio_file = AudioSegment.from_wav(fn)
        audio_chunks = split_on_silence(
            audio_file,
            min_silence_len=MIN_SILENCE_MS,
            silence_thresh=-SILENCE_THRESHOLD_DB
        )

        label_name = fn.split("/")[-1].split(".")[0]

        for i, audio in enumerate(audio_chunks):
            np_audio = np.frombuffer(audio.raw_data, np.int32)
            features = mfcc(np_audio, audio.frame_rate)
            features = features[:20, :]
            features = features.reshape(features.shape[0] * features.shape[1])
            print("{}: {}".format(i+1, label_name))

            if play:
                sd.play(np_audio, audio.frame_rate, blocking=True)

            training_data.append({
                "label": label_name,
                "features": features,
                "audio": audio
            })

    return training_data


def train(data):
    X = np.array([_["features"] for _ in data])
    y = np.array([_["label"] for _ in data])

    model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    model.fit(X, y)
    score = model.score(X, y)

    print(score)

    pickle.dump(model, open(MODEL_PICKLE_FILE_NAME, 'wb'))


def predict(test_audio, play=False):

    model = pickle.load(open(MODEL_PICKLE_FILE_NAME, 'rb'))

    predicted = []

    audio_chunks = split_on_silence(
        test_audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=-SILENCE_THRESHOLD_DB
    )

    for audio in audio_chunks:
        np_audio = np.frombuffer(audio.raw_data, np.int32)
        features = mfcc(np_audio, audio.frame_rate)
        features = features[:20, :]

        X = features.reshape(features.shape[0] * features.shape[1])
        _y = model.predict(X)

        if play:
            print("Predicted word: {}".format(_y))
            sd.play(np_audio, audio.frame_rate, blocking=True)
            sleep(1)

        predicted.append(_y[0])

    return " ".join(predicted)
