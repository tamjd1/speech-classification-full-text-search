from sklearn.linear_model import LogisticRegression
from pydub.silence import split_on_silence
from pydub import AudioSegment
from python_speech_features import mfcc
import numpy as np
import glob


def prepare():
    training_data = []

    sound_filenames = glob.glob("data/audio/*.wav")
    for fn in sound_filenames:
        sound_file = AudioSegment.from_wav(fn)
        audio_chunks = split_on_silence(
            sound_file,
            min_silence_len=50,  # milliseconds
            silence_thresh=-32  # decibels
        )

        label_name = fn.split("/")[-1].split(".")[0]

        for audio in audio_chunks:
            np_audio = np.frombuffer(audio.raw_data, np.int32)
            features = mfcc(np_audio, audio.frame_rate)
            features = features[:10, :]
            # print(features.shape)

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

    return model


def test(model, test_sound):
    predicted = []
    # parse test_sound into sound_chunks
    # for each chunk
    #   convert chunk raw data to np array
    #   extract mfcc features of chunk np array
    #   predict features and determine class
    #   generate array of features and classes
    # return features/classes array
    return predicted


def main():
    training_data = prepare()
    trained_model = train(training_data)
    predicted_values = test(trained_model, object)


if __name__ == '__main__':
    main()
