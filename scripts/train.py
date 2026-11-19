"""Train the accent classifier, evaluate it on held-out speakers, save the model
python train.py --data ./dataset --out model.npz
"""
import argparse
import json
from pathlib import Path

from accent_recognition.data import build_dataset, limit_clips_per_group, split_by_group
from accent_recognition.evaluate import (accuracy, balanced_accuracy, confusion_matrix,
                                         logistic_regression_baseline, majority_baseline,
                                         per_class_report)
from accent_recognition.cv import cross_validate
from accent_recognition.model import NeuralNetwork, Standardiser

from sklearn.model_selection import GroupShuffleSplit

def final_model(args, X, y, groups, classes):
    """Train on every speaker except a small early-stopping group, then save"""
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train, val = next(inner.split(X, y, groups))
    scaler = Standardiser().fit(X[train])
    X_train, X_val = scaler.transform(X[train]), scaler.transform(X[val])
    net = NeuralNetwork(X.shape[1], args.hidden, len(classes),
                        lr=args.lr, l2=args.l2, seed=args.seed)
    net.fit(X_train, y[train], X_val, y[val], epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience, balanced=True)
    pred = net.predict(X_val)
    cv_summary = None
    if Path(args.cv_metrics).exists():
        cv_summary = json.loads(Path(args.cv_metrics).read_text()).get("summary")
    metrics = {
        "mode": "final",
        "note": "Early-stopping speakers were used to pick the weights, so the "
                "early-stopping numbers are optimistic. Use the cross-validation "
                "summary as the honest estimate of accuracy on new speakers.",
        "classes": classes,
        "n_train_clips": int(len(train)), "n_val_clips": int(len(val)),
        "n_train_speakers": int(len(set(groups[train]))),
        "n_val_speakers": int(len(set(groups[val]))),
        "early_stopping_balanced_accuracy": balanced_accuracy(y[val], pred, len(classes)),
        "cross_validation_summary": cv_summary,
        "hyperparameters": {"hidden": args.hidden, "lr": args.lr, "l2": args.l2,
                            "batch_size": args.batch_size, "seed": args.seed},
    }
    for out in (args.out, args.metrics):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    net.save(args.out, scaler, classes)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(f"final model: {len(set(groups[train]))} train speakers, "
          f"{len(set(groups[val]))} early-stopping speakers")
    print(f"saved model -> {args.out}, metrics -> {args.metrics}")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="./dataset")
    p.add_argument("--out", default="artifacts/model.npz")
    p.add_argument("--metrics", default="artifacts/metrics.json")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-clips-per-speaker", type=int, default=0, metavar="N",
                   help="use at most N randomly chosen clips per speaker (0 = use all)")
    p.add_argument("--cv", type=int, default=0, metavar="K",
                   help="run speaker-grouped K-fold cross-validation instead of a single split "
                        "(reports mean and std; does not save a model)")
    p.add_argument("--balanced", action="store_true",
                   help="weight classes equally in the loss (network and logistic regression)")
    p.add_argument("--final", action="store_true",
                   help="train the shipped model on all speakers (20%% of speakers are used "
                        "only for early stopping) and save it. Use after cross-validation")
    p.add_argument("--cv-metrics", default="artifacts/cv_metrics.json")
    args = p.parse_args()

    X, y, groups, classes, _ = build_dataset(args.data)
    if args.max_clips_per_speaker:
        keep = limit_clips_per_group(groups, args.max_clips_per_speaker, seed=args.seed)
        X, y, groups = X[keep], y[keep], groups[keep]
    print(f"{len(y)} clips, {len(set(groups))} speakers, classes: {classes}")

    if args.cv:
        result = cross_validate(X, y, groups, classes, k=args.cv, seed=args.seed,
                                hidden=args.hidden, lr=args.lr, l2=args.l2,
                                epochs=args.epochs, batch_size=args.batch_size,
                                patience=args.patience, balanced=args.balanced)
        Path(args.cv_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.cv_metrics).write_text(json.dumps(result, indent=2))
        print(f"saved cross-validation results -> {args.cv_metrics}")
        return

    if args.final:
        final_model(args, X, y, groups, classes)
        return

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