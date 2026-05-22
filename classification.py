import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json

if __name__ == '__main__':
    klassifikatory = ["SVM", "MLPClassifier", "KNN"]
    ekstraktory = ["MFCC", "Chroma features", "Spectral centroid", "Zero crossing rate"]


    base_models = [
        make_pipeline(StandardScaler(), SVC()),
        make_pipeline(StandardScaler(), MLPClassifier(
            solver='adam',
            max_iter=1500,
            tol=1e-2,
            random_state=42)),
        make_pipeline(StandardScaler(), KNeighborsClassifier())
    ]

    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    total_folds = n_splits * n_repeats

    results = np.full((len(klassifikatory), (len(ekstraktory) + 1), total_folds), np.nan)

    data = np.load("features_labels.npz", allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    all_cm = np.full((len(klassifikatory), (len(ekstraktory) + 1), total_folds, 10, 10), np.nan)  # confusion matrices for each model and fold

    for ekstr_id, ekstr_name in enumerate(ekstraktory):
        X = np.array(features[ekstr_id].reshape(-1).tolist())
        print(f"Shape of X for {ekstr_name}: {X.shape}")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = labels[ekstr_id].reshape(-1)

        print(f"Przetwarzanie ekstraktora: {ekstr_name}...")

        for fold_id, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for model_id, base_model in enumerate(base_models):
                # Klonujemy model, aby mieć pewność czystego stanu przed trenowaniem
                model = clone(base_model)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                all_cm[model_id, ekstr_id, fold_id] = cm
                report = classification_report(y_test, y_pred, output_dict=True)
                with open(f"./reports/report_{model_id}_{ekstr_id}_{fold_id}.json", "w") as f:
                    json.dump(report, f, indent=4)
                results[model_id, ekstr_id, fold_id] = balanced_accuracy_score(y_test, y_pred)

    # klasyfikacja z ogolnym zestawem cech (wszystkie ekstraktory razem)
    X_all = []
    y_all = []
    for genre_id in range(features.shape[1]):
        for file_id in range(features.shape[2]):
            mfcc = features[0, genre_id, file_id]
            chroma = features[1, genre_id, file_id]
            centroid = features[2, genre_id, file_id]
            zcr = features[3, genre_id, file_id]
            combined = np.concatenate([
                mfcc,
                chroma,
                [centroid],
                [zcr]
            ])
            X_all.append(combined)
            y_all.append(labels[0, genre_id, file_id])
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)

    for fold_id, (train_index, test_index) in enumerate(rskf.split(X_all, y_all)):
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_all[train_index], y_all[test_index]

        for model_id, base_model in enumerate(base_models):
            # Klonujemy model, aby mieć pewność czystego stanu przed trenowaniem
            model = clone(base_model)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            all_cm[model_id, len(ekstraktory), fold_id] = cm
            report = classification_report(y_test, y_pred, output_dict=True)
            with open(f"./reports/report_{model_id}_4_{fold_id}.json", "w") as f:
                json.dump(report, f, indent=4)
            results[model_id, len(ekstraktory), fold_id] = balanced_accuracy_score(y_test, y_pred)

    print("\n--- WYNIKI KOŃCOWE (Średnia Balanced Accuracy) ---")
    for klass_id, klass in enumerate(klassifikatory):
        for ekstr_id, ekstr in enumerate(ekstraktory):
            mean_score = np.mean(results[klass_id, ekstr_id])
            print(f"{klass} + {ekstr}: {mean_score:.4f}")

    for klass_id, klass in enumerate(klassifikatory):
        mean_score = np.mean(results[klass_id, len(ekstraktory)])
        print(f"{klass} + All features combined: {mean_score:.4f}")

    # save data for visualization
    np.savez("classification_results.npz", results=results, klassifikatory=klassifikatory, ekstraktory=ekstraktory)

    genres = np.unique(y_all)
    np.savez(
        "confusion_matrices.npz",
        all_cm=all_cm,
        genres=genres,
        klassifikatory=klassifikatory,
        ekstraktory=ekstraktory
    )


    #jak bardzo niezbalansowana
    #pca i tsne -> wizualizacja
    #похожие взять и очень разные жанры и показать ломаной на время