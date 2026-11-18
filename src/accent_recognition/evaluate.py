"""Evaluation helpers: metrics and baselines to compare the model against"""
import numpy as np


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, n_classes):
    """Rows are true classes, columns are predicted classes"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_report(cm, classes):
    report = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return report


def majority_baseline(y_train, y_test):
    """Accuracy of always predicting the most common training class"""
    majority = np.bincount(y_train).argmax()
    return float((y_test == majority).mean())


def logistic_regression_baseline(X_train, y_train, X_test, y_test):
    """A strong, simple reference model (scikit-learn) on the same features"""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    return float(clf.score(X_test, y_test))