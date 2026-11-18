"""Dataset loading and speaker-aware splitting

Expected layout:  dataset/<accent>/<speaker>_<anything>.wav

The speaker id is the filename prefix before the first underscore. Splitting by
speaker stops the same voice appearing in both train and test, which would
otherwise inflate accuracy. If filenames have no underscore, every file is
treated as its own speaker (and the split is NOT speaker-independent)
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.accent_recognition.features import extract_features

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

def speaker_id(path):
    stem = Path(path).stem
    return stem.split("_")[0] if "_" in stem else stem

def build_dataset(root):
    """Return X, y, groups, classes, paths for every audio file under root"""
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))

    if not classes:
        raise ValueError(f"No accent subdirectories found in {root}")

    X, y, groups, paths = [], [], [], []
    for label, accent in enumerate(classes):
        for file in sorted((root / accent).iterdir()):
            if file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            try:
                X.append(extract_features(file))
            except Exception as exc:
                print(f"skipping {file} : {exc}")
                continue
            y.append(label)
            groups.append(f"{accent}/{speaker_id(file)}")
            paths.append(str(file))

    return np.array(X), np.array(y), np.array(groups), classes, paths

def split_by_group(y, groups, test_size=0.2, val_size=0.2, seed=42):
    """Return index arrays with no speaker shared across splits"""
    idx = np.arange(len(y))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(outer.split(idx, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train, val = next(inner.split(trainval, y[trainval], groups[trainval]))
    return trainval[train], trainval[val], test