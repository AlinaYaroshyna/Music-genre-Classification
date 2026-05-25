import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

DATA_PATH = "data"


def read_data():
    features = np.full((4, 10, 100), np.nan, dtype=object)
    labels = np.full((4, 10, 100), np.nan, dtype='<U32')
    viz_samples = []
    genres = sorted(os.listdir(DATA_PATH))

    for genre_id, genre in enumerate(genres):
        folder = os.path.join(DATA_PATH, genre)
        files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])[:100]

        for file_id, file in enumerate(files):
            path = os.path.join(folder, file)
            print(f"Przetwarzanie: {path}")
            y, sr = librosa.load(path, sr=22050, duration=30)
            mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            if file_id < 10:
                viz_samples.append((genre, mfcc_raw))
            features[0, genre_id, file_id] = np.mean(mfcc_raw.T, axis=0)
            labels[0, genre_id, file_id] = genre

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features[1, genre_id, file_id] = np.mean(chroma.T, axis=0)
            labels[1, genre_id, file_id] = genre

            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features[2, genre_id, file_id] = np.mean(centroid)
            labels[2, genre_id, file_id] = genre

            zcr = librosa.feature.zero_crossing_rate(y)
            features[3, genre_id, file_id] = np.mean(zcr)
            labels[3, genre_id, file_id] = genre

    print("Generowanie wykresu...")
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    for i, (genre_name, data) in enumerate(viz_samples):
        row = i // 10
        col = i % 10
        ax = axes[row, col]
        librosa.display.specshow(data, sr=22050, ax=ax, cmap='magma')
        if col == 0:
            ax.set_ylabel(genre_name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("mfcc_grid.png")
    plt.show()

    return features, labels

if __name__ == '__main__':
    features, labels = read_data()
    with open("features_labels.npz", "wb") as f:
        np.savez(f, features=features, labels=labels)