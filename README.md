## What's here

| Path                                 | Purpose                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `src/accent_recognition/features.py` | Whole-clip features: mean and std of 20 MFCCs and their deltas (80 values)                        |
| `src/accent_recognition/model.py`    | NumPy neural network: ReLU hidden layer, softmax, L2, minibatch SGD, early stopping, save/load    |
| `src/accent_recognition/data.py`     | Dataset loading and a **speaker-aware** train/val/test split                                      |
| `src/accent_recognition/evaluate.py` | Accuracy, confusion matrix, per-class precision/recall/F1, baselines                              |
| `src/accent_recognition/app.py`      | FastAPI service with `/predict` and `/health`                                                     |
| `scripts/train.py`                   | Trains, evaluates on held-out speakers, writes `artifacts/model.npz` and `artifacts/metrics.json` |
| `tests/test_pipeline.py`             | Gradient check, end-to-end training, save/load and API tests                                      |
| `docs/dataset.md`                    | Dataset source, licence and speaker selection rule                                                |

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
python scripts/train.py --data dataset --out artifacts/model.npz
MODEL_PATH=artifacts/model.npz uvicorn accent_recognition.app:app --port 8000
curl -X POST -F "file=@clip.wav" http://127.0.0.1:8000/predict
```

Or with Docker (after training):

```bash
docker build -t accent-recognition .
docker run -p 8000:8000 accent-recognition
```

Run the tests with `pytest`.
