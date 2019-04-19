from pydub import AudioSegment
import glob
import speech_classification as sc
import full_text_search as fts
import os
import sys
f = open(os.devnull, 'w')
sys.stderr = f


def voice_to_text_search(audio_search_term, text_search_term):

    print("Intended search term: \"{}\"".format(text_search_term))

    predicted_search_term = sc.predict(audio_search_term)
    print("Predicted search term: \"{}\"".format(predicted_search_term))

    documents_found = fts.search(predicted_search_term)

    if documents_found:
        print("\nThe following documents were found containing \"{}\":".format(predicted_search_term))
        for i, document in enumerate(documents_found):
            print("{}\t{}".format(i+1, document))
    else:
        print("\nNo documents were found containing \"{}\":".format(predicted_search_term))


def main():
    audio_filenames = glob.glob("data/audio/test/*.wav")

    for i, fn in enumerate(audio_filenames):
        print("\n---------- TEST {} ----------\n".format(i + 1))
        audio_file = AudioSegment.from_wav(fn)
        label_name = fn.split("/")[-1].split(".")[0]
        audio_text = " ".join(label_name.split("_"))
        voice_to_text_search(audio_file, audio_text)
        print("\n-----------------------------\n")


if __name__ == '__main__':
    # todo : train on "light" again
    # todo : train on words which don't exist in the CORPUS
    main()
