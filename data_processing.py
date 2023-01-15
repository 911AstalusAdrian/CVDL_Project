import json
import math
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

from constants import SAMPLE_RATE, SAMPLES_PER_TRACK, JSON_PATH


def create_data(dataset_path, json_path, nr_mfcc=15, nr_fft=2048, hop_len=512, nr_segments=10):
    final_data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / nr_segments)
    mfccs_per_segment = math.ceil(samples_per_segment / hop_len)

    # loop through the sounds folder
    # we use enumerate of walk, so we have a corresponding value to each music type (blues -> 0, classical -> 1, and so on)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:

            # we save the name of each genre in our data dictionary
            genre_label = dirpath.split("\\")[
                -1]  # we extract the genre from the folder path (ex: 'genres_original/blues' -> 'blues')
            final_data["mapping"].append(genre_label)
            print("\nProcessing the {} genre dataset".format(genre_label))

            # process each file in the directory
            for f in filenames:

                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)  # load the audio file using librosa

                # process each segment of the audio file
                for d in range(nr_segments):

                    # compute the starting and ending samples for the segment
                    start_sample = samples_per_segment * d
                    end_sample = start_sample + samples_per_segment

                    # extract the mfcc for the segment
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:end_sample],
                                                sr=sample_rate, n_mfcc=nr_mfcc,
                                                n_fft=nr_fft, hop_length=hop_len)
                    mfcc = mfcc.T

                    # store only mfcc features with the expected number of vectors
                    if len(mfcc) == mfccs_per_segment:
                        final_data["mfcc"].append(mfcc.tolist())
                        final_data["labels"].append(i - 1)
                        # print("{} - segment {}".format(file_path, d + 1))

    # save processed data to the provided json file
    with open(json_path, "w") as fp:
        json.dump(final_data, fp, indent=4)


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(JSON_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]  # 4d array -> (nr_samples, 150, 15, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test
