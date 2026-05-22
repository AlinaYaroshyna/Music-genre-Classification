import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

if __name__ == '__main__':
    klassifikatory = ["SVM", "MLPClassifier", "KNN"]
    ekstraktory = ["MFCC", "Chroma features", "Spectral centroid", "Zero crossing rate"]

    # Definiujemy bazowe potoki (pipelines) raz
    base_models = [
        make_pipeline(StandardScaler(), SVC()),
        make_pipeline(StandardScaler(), make_pipeline(StandardScaler(), MLPClassifier(solver='lbfgs', max_iter=1000, random_state=42)),
        # dodano random_state dla powtarzalności
        make_pipeline(StandardScaler(), KNeighborsClassifier())
    ]

    # Parametry walidacji
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    total_folds = n_splits * n_repeats

    # Dynamiczne tworzenie tablicy na wyniki
    results = np.full((len(klassifikatory), len(ekstraktory), total_folds), np.nan)

    # Ładowanie danych
    data = np.load("features_labels.npz", allow_pickle=True)
    features = data["features"]
    labels = data["labels"]

    for ekstr_id, ekstr_name in enumerate(ekstraktory):
        # Bezpieczne wyciąganie danych
        X = np.array(features[ekstr_id].reshape(-1).tolist())
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
                results[model_id, ekstr_id, fold_id] = balanced_accuracy_score(y_test, y_pred)

    print("\n--- WYNIKI KOŃCOWE (Średnia Balanced Accuracy) ---")
    for klass_id, klass in enumerate(klassifikatory):
        for ekstr_id, ekstr in enumerate(ekstraktory):
            mean_score = np.mean(results[klass_id, ekstr_id])
            print(f"{klass} + {ekstr}: {mean_score:.4f}")