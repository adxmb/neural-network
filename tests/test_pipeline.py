"""Tests gradient correctness, end-to-end training on synthetic audio, and the API"""
import numpy as np
import pytest
import soundfile as sf

from accent_recognition.data import build_dataset, split_by_group
from accent_recognition.features import FEATURE_SIZE, SAMPLE_RATE
from accent_recognition.model import NeuralNetwork, Standardiser

ACCENT_FREQS = {"alpha": 220.0, "beta": 440.0, "gamma": 880.0}

def make_dataset(root, speakers_per_accent=6, clips_per_speaker=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(1.5 * SAMPLE_RATE)) / SAMPLE_RATE
    for accent, freq in ACCENT_FREQS.items():
        folder = root / accent
        folder.mkdir(parents=True)
        for s in range(speakers_per_accent):
            for c in range(clips_per_speaker):
                f = freq * (1 + rng.normal(0, 0.03))
                wave = np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(t.size)
                sf.write(folder / f"spk{s}_{c}.wav", (0.5 * wave).astype(np.float32), SAMPLE_RATE)

@pytest.mark.parametrize("weighted", [False, True])
def test_gradients_match_numerical(weighted):
    rng = np.random.default_rng(1)
    net = NeuralNetwork(6, 5, 3, l2=1e-2, seed=1)
    X = rng.standard_normal((8, 6))
    y = rng.integers(0, 3, 8)

    sw = rng.uniform(0.3, 3.0, 8) if weighted else None
    probs = net.forward(X)
    grads = net.gradients(probs, y, sw)

    eps = 1e-5
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(net, name)
        for idx in [tuple(rng.integers(0, s) for s in param.shape) for _ in range(5)]:
            original = param[idx]
            param[idx] = original + eps
            plus = net.loss(net.forward(X), y, sw)
            param[idx] = original - eps
            minus = net.loss(net.forward(X), y, sw)
            param[idx] = original
            numeric = (plus - minus) / (2 * eps)
            assert numeric == pytest.approx(grads[name][idx], rel=1e-4, abs=1e-7)


def test_split_has_no_speaker_overlap(tmp_path):
    make_dataset(tmp_path)
    X, y, groups, classes, _ = build_dataset(tmp_path)
    assert X.shape[1] == FEATURE_SIZE
    assert classes == sorted(ACCENT_FREQS)
    train, val, test = split_by_group(y, groups)
    sets = [set(groups[i]) for i in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_end_to_end_training_learns(tmp_path):
    make_dataset(tmp_path)
    X, y, groups, classes, _ = build_dataset(tmp_path)
    train, val, test = split_by_group(y, groups)
    scaler = Standardiser().fit(X[train])
    net = NeuralNetwork(X.shape[1], 16, len(classes), seed=0)
    net.fit(scaler.transform(X[train]), y[train], scaler.transform(X[val]), y[val],
            epochs=200, verbose=False)
    acc = (net.predict(scaler.transform(X[test])) == y[test]).mean()
    assert acc > 0.9


def test_save_and_load_roundtrip(tmp_path):
    make_dataset(tmp_path / "data")
    X, y, groups, classes, _ = build_dataset(tmp_path / "data")
    scaler = Standardiser().fit(X)
    net = NeuralNetwork(X.shape[1], 8, len(classes), seed=0)
    path = tmp_path / "model.npz"
    net.save(path, scaler, classes)
    net2, scaler2, classes2 = NeuralNetwork.load(path)
    assert classes2 == classes
    x = scaler.transform(X[:4])
    assert np.allclose(net.predict_proba(x), net2.predict_proba(scaler2.transform(X[:4])))


def test_api_predict(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    make_dataset(tmp_path / "data")
    X, y, groups, classes, paths = build_dataset(tmp_path / "data")
    scaler = Standardiser().fit(X)
    net = NeuralNetwork(X.shape[1], 16, len(classes), seed=0)
    net.fit(scaler.transform(X), y, epochs=150, verbose=False)
    model_path = tmp_path / "model.npz"
    net.save(model_path, scaler, classes)

    import accent_recognition.app as app_module
    monkeypatch.setattr(app_module, "MODEL_PATH", str(model_path))

    with TestClient(app_module.app) as client:
        assert client.get("/health").json()["classes"] == classes
        with open(paths[0], "rb") as f:
            resp = client.post("/predict", files={"file": ("clip.wav", f, "audio/wav")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accent"] in classes
        assert 0 <= body["confidence"] <= 1
        assert sum(body["probabilities"].values()) == pytest.approx(1.0)

        bad = client.post("/predict", files={"file": ("x.wav", b"not audio", "audio/wav")})
        assert bad.status_code == 400

def test_cross_validate_runs(tmp_path):
    from accent_recognition.cv import cross_validate

    make_dataset(tmp_path)
    X, y, groups, classes, _ = build_dataset(tmp_path)
    result = cross_validate(X, y, groups, classes, k=3, hidden=16, epochs=150, verbose=False)
    assert len(result["folds"]) == 3
    assert result["summary"]["network"]["mean"] > 0.9
    assert result["summary"]["majority"]["mean"] < 0.5


def test_limit_clips_per_group():
    from accent_recognition.data import limit_clips_per_group

    groups = np.array(["a"] * 10 + ["b"] * 5 + ["c"] * 2)
    keep = limit_clips_per_group(groups, 4, seed=1)
    counts = {g: int((groups[keep] == g).sum()) for g in "abc"}
    assert counts == {"a": 4, "b": 4, "c": 2}
    assert list(keep) == sorted(keep)
    assert list(keep) == list(limit_clips_per_group(groups, 4, seed=1))

def test_balanced_accuracy_ignores_class_size():
    from accent_recognition.evaluate import balanced_accuracy
    y_true = np.array([0] * 8 + [1] * 2)
    assert balanced_accuracy(y_true, np.zeros(10, dtype=int), 2) == pytest.approx(0.5)


def test_balanced_training_runs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(3, 1, (8, 4))])
    y = np.array([0] * 40 + [1] * 8)
    net = NeuralNetwork(4, 8, 2, seed=0)
    net.fit(X, y, epochs=30, verbose=False, balanced=True)
    assert (net.predict(X[y == 1]) == 1).mean() > 0.8

def test_final_mode_saves_loadable_model(tmp_path):
    import subprocess, sys, json
    from accent_recognition.model import NeuralNetwork
    make_dataset(tmp_path / "data")
    out, metrics = tmp_path / "m.npz", tmp_path / "m.json"
    subprocess.run([sys.executable, "scripts/train.py", "--data", str(tmp_path / "data"),
                    "--final", "--epochs", "5", "--out", str(out), "--metrics", str(metrics),
                    "--cv-metrics", str(tmp_path / "none.json")], check=True)
    net, scaler, classes = NeuralNetwork.load(out)
    assert len(classes) == 3
    assert json.loads(metrics.read_text())["mode"] == "final"