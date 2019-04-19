from pydub import AudioSegment

import speech_classification as sc
import full_text_search as fts


def test_speech_classifier(retrain=False):
    if retrain:
        data = sc.prepare(play=False)
        sc.train(data)

    audio = AudioSegment.from_wav("data/audio/test/love_and_light.wav")
    predicted_values = sc.predict(audio, play=False)
    assert predicted_values == "love and light"


def test_full_text_search():
    docs = fts.search("love and light")
    for doc in docs:
        assert ("love" in doc.lower() or "light" in doc.lower())

    docs = fts.search("O Canada")
    assert len(docs) == 0
