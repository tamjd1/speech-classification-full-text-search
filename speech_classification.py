from sklearn.linear_model import LogisticRegression
from pydub.silence import split_on_silence
from pydub import AudioSegment
from python_speech_features import mfcc
import numpy as np
import glob
import sounddevice as sd
import pickle
from time import sleep

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
            features = features[:10, :]
            # print(features.shape)
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
    X = np.array([_["features"].reshape(10*13) for _ in data])
    y = np.array([_["label"] for _ in data])

    model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    model.fit(X, y)
    score = model.score(X, y)

    print(score)

    pickle.dump(model, open(MODEL_PICKLE_FILE_NAME, 'wb'))

    return model


def test(model, test_audio, play=False):
    predicted = []

    audio_chunks = split_on_silence(
        test_audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=-SILENCE_THRESHOLD_DB
    )

    for audio in audio_chunks:
        np_audio = np.frombuffer(audio.raw_data, np.int32)

        if play:
            sd.play(np_audio, audio.frame_rate, blocking=True)
            sleep(1)

        features = mfcc(np_audio, audio.frame_rate)
        features = features[:10, :]

        X = features.reshape(10 * 13)
        _y = model.predict(X)

        print(_y)

        predicted.append(_y)

    return predicted


def main(retrain=False):
    if retrain:
        training_data = prepare(play=False)
        trained_model = train(training_data)
    else:
        trained_model = pickle.load(open(MODEL_PICKLE_FILE_NAME, 'rb'))

    audio = AudioSegment.from_wav("data/audio/test/love_and_light.wav")
    predicted_values = test(trained_model, audio, play=True)
    print(predicted_values)


if __name__ == '__main__':
    # todo : train on "light" again
    main(retrain=False)
