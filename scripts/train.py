"""Train the accent classifier, evaluate it on held-out speakers, save the model
python train.py --data ./dataset --out model.npz
"""
import argparse
import json
from pathlib import Path

from accent_recognition.data import build_dataset, split_by_group
from accent_recognition.evaluate import (accuracy, confusion_matrix, logistic_regression_baseline,
                      majority_baseline, per_class_report)
from accent_recognition.model import NeuralNetwork, Standardiser


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="./dataset")
    p.add_argument("--out", default="artifacts/model.npz")
    p.add_argument("--metrics", default="artifacts/metrics.json")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    X, y, groups, classes, _ = build_dataset(args.data)
    print(f"{len(y)} clips, {len(set(groups))} speakers, classes: {classes}")

    train, val, test = split_by_group(y, groups, seed=args.seed)
    print(f"split -> train {len(train)}, val {len(val)}, test {len(test)}")

    scaler = Standardiser().fit(X[train])
    X_train, X_val, X_test = (scaler.transform(X[i]) for i in (train, val, test))

    net = NeuralNetwork(X.shape[1], args.hidden, len(classes),
                        lr=args.lr, l2=args.l2, seed=args.seed)
    net.fit(X_train, y[train], X_val, y[val], epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience)

    pred = net.predict(X_test)
    cm = confusion_matrix(y[test], pred, len(classes))
    metrics = {
        "test_accuracy": accuracy(y[test], pred),
        "majority_baseline": majority_baseline(y[train], y[test]),
        "logistic_regression_baseline": logistic_regression_baseline(
            X_train, y[train], X_test, y[test]),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_report(cm, classes),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "seed": args.seed,
    }

    for out in (args.out, args.metrics):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    net.save(args.out, scaler, classes)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))

    print(f"\ntest accuracy:        {metrics['test_accuracy']:.3f}")
    print(f"majority baseline:    {metrics['majority_baseline']:.3f}")
    print(f"logistic regression:  {metrics['logistic_regression_baseline']:.3f}")
    print(f"saved model -> {args.out}, metrics -> {args.metrics}")


if __name__ == "__main__":
    main()