import os
import librosa
import numpy as np

DATA_PATH = "data"

def read_data(): # Read the data (music files) from each genre
    features = np.full((4, 10, 100), np.nan, dtype=object)  # shape: (4, 10, 100) - 4 ekstraktory, 10 genres, 100 probek
    labels = np.full((4, 10, 100), np.nan, dtype='<U32')  # shape: (4, 10, 100) - 4 ekstraktory, 10 genres, 100 probek

    for genre_id, genre in enumerate(os.listdir(DATA_PATH)):
        folder = f"{DATA_PATH}/{genre}"

        for file_id, file in enumerate(os.listdir(folder)):
            path = f"{folder}/{file}"
            print(path)
            y, sr = librosa.load(path, sr=22050, duration=30)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            features[0, genre_id, file_id] = mfcc_mean
            labels[0, genre_id, file_id] = genre

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)

            features[1, genre_id, file_id] = chroma_mean
            labels[1, genre_id, file_id] = genre

            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            centroid_mean = np.mean(centroid)

            features[2, genre_id, file_id] = centroid_mean
            labels[2, genre_id, file_id] = genre

            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)

            features[3, genre_id, file_id] = zcr_mean
            labels[3, genre_id, file_id] = genre
    return features, labels

if __name__ == '__main__':
    features, labels = read_data()
    with open("features_labels.npz", "wb") as f:
        np.savez(f, features=features, labels=labels)