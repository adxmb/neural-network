"""Speaker-grouped, class-stratified k-fold cross-validation

Every speaker lands in exactly one test fold, so no fold is ever tested on a voice
it trained on. Stratification keeps the accents roughly balanced across folds, which
a plain random split of a few dozen speakers does not
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .evaluate import (accuracy, balanced_accuracy, confusion_matrix,
                       logistic_regression_predict, majority_baseline, per_class_report)
from .model import NeuralNetwork, Standardiser


def cross_validate(X, y, groups, classes, k=5, seed=42, hidden=64, lr=0.05, l2=1e-3,
                   epochs=300, batch_size=32, patience=30, verbose=True, balanced=False):
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
                patience=patience, verbose=False, balanced=balanced)
        pred = net.predict(X_test)
        pooled += confusion_matrix(y[test], pred, n_classes)

        lr_pred = logistic_regression_predict(X_train, y[train], X_test, balanced=balanced)
        maj = np.full(len(test), np.bincount(y[train], minlength=n_classes).argmax())
        row = {
            "fold": fold,
            "test_speakers": int(len(set(groups[test]))),
            "test_clips": int(len(test)),
            "network": accuracy(y[test], pred),
            "logistic_regression": accuracy(y[test], lr_pred),
            "majority": majority_baseline(y[train], y[test]),
            "network_bal": balanced_accuracy(y[test], pred, n_classes),
            "logistic_regression_bal": balanced_accuracy(y[test], lr_pred, n_classes),
            "majority_bal": balanced_accuracy(y[test], maj, n_classes),
        }
        folds.append(row)
        if verbose:
            print(f"fold {fold}/{k}: {row['test_speakers']:>2} test speakers  "
                  f"network {row['network']:.3f}  logreg {row['logistic_regression']:.3f}  "
                  f"majority {row['majority']:.3f}")

    summary = {}
    for key in ("network", "logistic_regression", "majority",
                "network_bal", "logistic_regression_bal", "majority_bal"):
        values = np.array([f[key] for f in folds])
        summary[key] = {"mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    if verbose:
        print()
        print(f"{'':<20} {'accuracy':<16} balanced accuracy")
        for key, label in (("network", "NumPy network"), ("logistic_regression", "logistic regression"),
                           ("majority", "majority class")):
            a, b = summary[key], summary[key + "_bal"]
            print(f"{label:<20} {a['mean']:.3f} +/- {a['std']:.3f}  {b['mean']:.3f} +/- {b['std']:.3f}")

    return {
        "k": k, "seed": seed, "balanced": balanced, "classes": list(classes),
        "folds": folds, "summary": summary,
        "pooled_confusion_matrix": pooled.tolist(),
        "pooled_per_class": per_class_report(pooled, classes),
    }