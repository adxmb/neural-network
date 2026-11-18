# Design notes

## Goal

Classify English accents from short audio clips. The classifier is a neural network
written from scratch in NumPy. The project measures it on speakers it has never heard,
compares it with simple baselines, and serves it through an HTTP API.

## Pipeline

audio file -> features (80 values) -> standardize -> NumPy network -> softmax
probabilities -> FastAPI response

## Decisions

**Whole-clip features.**
The first version used only the first 30 MFCC frames of each clip (about 0.7 s) and
flattened them. Most of each recording was ignored. Now each clip becomes the mean
and standard deviation of 20 MFCCs and their deltas. This gives 80 values for any
clip length. The cost: summary statistics lose timing and rhythm.

**Speaker-independent evaluation.**
A random split of clips puts the same speaker in training and in test. The model can
then recognize the voice instead of the accent, and the accuracy looks too high. All
splits keep each speaker in one set only. The final numbers come from 5-fold
cross-validation. It is grouped by speaker and balanced by accent. In each fold, 20%
of the training speakers are held out for early stopping.

**Network.**
One hidden layer with ReLU, He initialization, softmax output, L2 regularization,
minibatch SGD, and early stopping on validation loss. A test compares the analytic
gradients with numerical gradients.

**Baselines.**
Each run reports the majority class and a logistic regression on the same features.
The network is only useful if it beats them.

**Model file.**
The file holds the weights, the scaler values, and the class names together. A model
cannot be used with the wrong preprocessing or the wrong label order.

**Serving.**
Training is a separate script. The API checks each upload (empty, too large, not
audio) and uses a unique temporary file for each request.

## Results

Dataset: 57 speakers, 19 for each of english, american and scottish, with 100 clips
for each speaker (see `dataset.md`). Chance for 3 classes is 0.333.

5-fold cross-validation, grouped by speaker:

| Model                            | Accuracy (mean ± std over 5 folds) |
| -------------------------------- | ---------------------------------- |
| Majority class                   | 0.333 ± 0.037                      |
| Logistic regression              | 0.602 ± 0.092                      |
| NumPy network (default settings) | 0.530 ± 0.051                      |

| Fold | Test speakers | Network | Logistic regression | Majority |
| ---- | ------------- | ------- | ------------------- | -------- |
| 1    | 12            | 0.468   | 0.493               | 0.333    |
| 2    | 11            | 0.608   | 0.688               | 0.364    |
| 3    | 11            | 0.523   | 0.675               | 0.273    |
| 4    | 12            | 0.539   | 0.640               | 0.333    |
| 5    | 11            | 0.513   | 0.514               | 0.364    |

`train.py --cv` also writes the pooled confusion matrix and the per-class precision
and recall to `artifacts/cv_metrics.json`.

### Settings tried

| Setting                                              | Network       | Logistic regression |
| ---------------------------------------------------- | ------------- | ------------------- |
| Default (l2 0.001, hidden 64, 100 clips per speaker) | 0.530 ± 0.051 | 0.602 ± 0.092       |
| l2 0.01                                              | 0.544 ± 0.067 | 0.602 ± 0.092       |
| l2 0.1                                               | 0.555 ± 0.085 | 0.602 ± 0.092       |
| hidden 16                                            | 0.537 ± 0.056 | 0.602 ± 0.092       |
| 30 clips per speaker                                 | 0.516 ± 0.080 | 0.586 ± 0.105       |
| 10 clips per speaker                                 | 0.515 ± 0.065 | 0.527 ± 0.056       |

I tried 5 settings besides the default on the same folds. The best-looking result of
several tries is slightly too good, so treat it with care.

### Findings

1. Both models are clearly above chance.
2. Logistic regression is equal to or better than the network in every fold. With 5
   folds the spread is large, so the gap is likely but not certain.
3. The network learns the training voices. In four single-split runs, the training
   loss fell to about 0.08 while the validation loss rose from the first epoch. Early
   stopping chose weights from epochs 0, 2, 0 and 5. The network has seen only about
   36 speakers in each training set.
4. Stronger L2 regularization raised the mean from 0.530 to 0.555. The change is
   smaller than the standard deviation, so it is not a clear result.
5. Fewer clips for each speaker did not help. Both models scored the same or lower.
   The number of clips is not the problem.
6. A single split is unreliable. Four single splits gave 0.357 to 0.455 for the
   network (seeds 42, 1, 2 and 3), because a test set has only 12 speakers.

The 80 summary features probably limit the accuracy. I have not yet tested other
features, so this is an inference and not a measured result.

## Limits

- The data has three accents and 57 speakers. See `dataset.md` for the accents that
  I could not include.
- The speech is read speech from a studio. The model is unlikely to work on noisy or
  telephone audio.
- Folds have 11 or 12 test speakers. Results vary by several points between folds.
- MFCC statistics do not keep timing, rhythm, or pitch movement. These carry accent
  information.
- The network does not beat logistic regression on these features. A more complex
  model does not help when the features are the limit.
