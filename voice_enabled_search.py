from pydub import AudioSegment
import glob
import speech_classification as sc
import full_text_search as fts
import os
import sys
f = open(os.devnull, 'w')
sys.stderr = f


def voice_to_text_search(audio_search_term, text_search_term):
    """
    This function takes an audio file, runs it through the speech classification model
        to get the words converted to text, and then runs the predicted text
        through the full text search algorithm on the documents in the CORPUS
        to retrieve the relevant documents and prints them on stdout
    :param audio_search_term: audio file of spoken text
    :param text_search_term: string of spoken text
    :return: None
    """

    print("Actual search term: \"{}\"".format(text_search_term))

    predicted_search_term = sc.predict(audio_search_term)
    print("Predicted search term: \"{}\"".format(predicted_search_term))

    documents_found = fts.search(predicted_search_term)

    if documents_found:
        print("\nThe following documents were found containing \"{}\":".format(predicted_search_term))
        for i, document in enumerate(documents_found):
            print("{}\t{}".format(i+1, document))
    else:
        print("\nNo documents were found containing \"{}\":".format(predicted_search_term))


def main(train=False):
    if train:
        # the first time this application runs this flag
        #   needs to be set to True so the speech classification model
        #   can be trained and be ready for predictions
        training_data = sc.prepare(play=False)
        sc.train(training_data)

    # get all the testing files
    audio_filenames = glob.glob("data/audio/test/*.wav")

    for i, fn in enumerate(audio_filenames):
        # for each test file, use it to do full text search on the CORPUS
        print("\n---------- TEST {} ----------\n".format(i + 1))
        audio_file = AudioSegment.from_wav(fn)
        label_name = fn.split("/")[-1].split(".")[0]
        audio_text = " ".join(label_name.split("_"))
        voice_to_text_search(audio_file, audio_text)


if __name__ == '__main__':
    # todo : train on "light" again
    # todo : train on words which don't exist in the CORPUS
    main(True)
