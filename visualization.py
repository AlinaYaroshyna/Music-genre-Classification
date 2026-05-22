import numpy as np
import json
import os
import pandas as pd

"""
    results содержит значения balanced accuracy
    для каждого классификатора, набора признаков
    и fold-а кросс-валидации.
    
    Размерность:
    (классификатор, экстрактор признаков, fold)
    
    Можно вычислять:
    - среднее качество модели
    - стандартное отклонение
    - строить boxplot и barplot
    
    это под комментом
"""

def read_classification_results():
    data = np.load("classification_results.npz", allow_pickle=True)

    results = data["results"]
    klassifikatory = data["klassifikatory"]
    ekstraktory = data["ekstraktory"]

    print(results.shape)
    print(klassifikatory)
    print(ekstraktory)
    # srednia accuracy:
    for model_id, model_name in enumerate(klassifikatory):
        for ekstr_id, ekstr_name in enumerate(ekstraktory):
            mean_score = np.mean(results[model_id, ekstr_id])
            print(model_name, ekstr_name, mean_score)

    return klassifikatory, ekstraktory

"""
    all_cm содержит confusion matrix
    для каждого классификатора,
    набора признаков и fold-а.
    
    Размерность:
    (model, extractor, fold, true_class, predicted_class)
    
    Можно анализировать:
    - какие жанры чаще путаются
    - насколько хорошо разделяются классы
    - среднюю confusion matrix по всем fold
    - строить heatmap по средней confusion matrix
    
    опять же, ф-ция под комментом
"""

def read_confusion_matrices():
    data = np.load("confusion_matrices.npz", allow_pickle=True)

    all_cm = data["all_cm"]
    genres = data["genres"]
    klassifikatory = data["klassifikatory"]
    ekstraktory = data["ekstraktory"]

    print(all_cm.shape)
    print(genres)

    # dla svm mfcc fold 0
    cm = all_cm[0, 0, 0]
    print(cm)

    # srednia dla svm mfcc (dla heatmapow):
    mean_cm = np.mean(all_cm[0, 0], axis=0)
    print(mean_cm)

    return klassifikatory, ekstraktory, all_cm

"""
    Каждый JSON-файл содержит classification report
    для конкретной комбинации:
    
    (model, extractor, fold)
    
    Имя файла:
    report_MODEL_EXTRACTOR_FOLD.json
    
    Например:
    report_2_4_7.json
    
    означает:
    - модель KNN
    - все признаки вместе
    - fold номер 7
    
    Внутри файла находятся:
    - precision
    - recall
    - f1-score
    - support
    
    для каждого жанра отдельно.
"""

def read_raports():

    # dla pojedynczego raportu:
    with open("./reports/report_0_0_0.json") as f:
        report = json.load(f)
    print(report)

    # konkretne metryki:
    rock_f1 = report["rock"]["f1-score"]
    print(rock_f1)

    # wczytanie wszystkich raportów do słownika:
    reports = {}
    for filename in os.listdir("./reports"):
        if filename.endswith(".json"):
            path = os.path.join("./reports", filename)
            with open(path) as f:
                reports[filename] = json.load(f)
    print(len(reports))

    # rozkodowanie nazwy pliku:

    filename = "report_2_4_7.json"

    parts = filename.replace(".json", "").split("_")

    model_id = int(parts[1])
    extractor_id = int(parts[2])
    fold_id = int(parts[3])

    print(model_id, extractor_id, fold_id)

    #tablica do analizy:
    rows = []

    for filename, report in reports.items():
        parts = filename.replace(".json", "").split("_")

        model_id = int(parts[1])
        extractor_id = int(parts[2])
        fold_id = int(parts[3])

        rows.append({
            "model": model_id,
            "extractor": extractor_id,
            "fold": fold_id,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        })

    df = pd.DataFrame(rows)

    print(df.head())
    # sredni dla f1 dla modeli:
    df.groupby("model")["macro_f1"].mean()
    # sredni dla f1 dla ekstraktorow:
    df.groupby("extractor")["macro_f1"].mean()

    return 0


if __name__ == '__main__':
    read_raports()
    read_classification_results()
    read_confusion_matrices()