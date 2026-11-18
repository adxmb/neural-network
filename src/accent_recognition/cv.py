"""Speaker-grouped, class-stratified k-fold cross-validation

Every speaker lands in exactly one test fold, so no fold is ever tested on a voice
it trained on. Stratification keeps the accents roughly balanced across folds, which
a plain random split of a few dozen speakers does not
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .evaluate import (accuracy, confusion_matrix, logistic_regression_baseline,
                       majority_baseline, per_class_report)
from .model import NeuralNetwork, Standardiser


def cross_validate(X, y, groups, classes, k=5, seed=42, hidden=64, lr=0.05, l2=1e-3,
                   epochs=300, batch_size=32, patience=30, verbose=True):
    """Return per-fold accuracies, their mean/std, and a pooled confusion matrix"""
    n_classes = len(classes)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    pooled = np.zeros((n_classes, n_classes), dtype=int)

    for fold, (trainval, test) in enumerate(splitter.split(X, y, groups), 1):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, va = next(inner.split(trainval, y[trainval], groups[trainval]))
        train, val = trainval[tr], trainval[va]

        scaler = Standardiser().fit(X[train])
        X_train, X_val, X_test = (scaler.transform(X[i]) for i in (train, val, test))

        net = NeuralNetwork(X.shape[1], hidden, n_classes, lr=lr, l2=l2, seed=seed)
        net.fit(X_train, y[train], X_val, y[val], epochs=epochs, batch_size=batch_size,
                patience=patience, verbose=False)
        pred = net.predict(X_test)
        pooled += confusion_matrix(y[test], pred, n_classes)

        row = {
            "fold": fold,
            "test_speakers": int(len(set(groups[test]))),
            "test_clips": int(len(test)),
            "network": accuracy(y[test], pred),
            "logistic_regression": logistic_regression_baseline(
                X_train, y[train], X_test, y[test]),
            "majority": majority_baseline(y[train], y[test]),
        }
        folds.append(row)
        if verbose:
            print(f"fold {fold}/{k}: {row['test_speakers']:>2} test speakers  "
                  f"network {row['network']:.3f}  logreg {row['logistic_regression']:.3f}  "
                  f"majority {row['majority']:.3f}")

    summary = {}
    for key in ("network", "logistic_regression", "majority"):
        values = np.array([f[key] for f in folds])
        summary[key] = {"mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    if verbose:
        print()
        for key, label in (("network", "NumPy network"), ("logistic_regression", "logistic regression"),
                           ("majority", "majority class")):
            s = summary[key]
            print(f"{label:<20} {s['mean']:.3f} +/- {s['std']:.3f}  (mean +/- std over {k} folds)")

    return {
        "k": k, "seed": seed, "classes": list(classes),
        "folds": folds, "summary": summary,
        "pooled_confusion_matrix": pooled.tolist(),
        "pooled_per_class": per_class_report(pooled, classes),
    }