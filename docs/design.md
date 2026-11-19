# Design notes

## Goal

Classify English accents from short audio clips. The classifier is a neural network
written from scratch in NumPy. The project measures it on speakers it has never heard,
compares it with simple baselines, and serves it through a web page and an HTTP API.

## Pipeline

audio file -> mono 16 kHz -> features (80 values) -> standardise -> NumPy network ->
softmax probabilities -> FastAPI response -> web page

## Decisions

**Whole-clip features.**
The first version used only the first 30 MFCC frames of each clip (about 0.7 s) and
flattened them. Most of each recording was ignored. Now each clip becomes the mean and
standard deviation of 20 MFCCs and their deltas. This gives 80 values for any clip
length. The cost: summary statistics lose timing and rhythm.

**Speaker-independent evaluation.**
A random split of clips puts the same speaker in training and in test. The model can
then recognise the voice instead of the accent, and the accuracy looks too high. All
splits keep each speaker in one set only. The final numbers come from 5-fold
cross-validation. It is grouped by speaker and balanced by accent. In each fold, 20% of
the training speakers are held out for early stopping.

**Balanced accuracy and class weights.**
The classes have 33, 22 and 19 speakers. Plain accuracy then rewards a model that
favours English. The majority class alone scores 0.446. Balanced accuracy is the mean
recall over the classes. Guessing one class scores 0.333 for any class sizes. The
training loss gives each class the same total weight, and so does the logistic
regression baseline. A test compares the weighted gradients with numerical gradients.

**Network.**
One hidden layer with ReLU, He initialisation, softmax output, L2 regularisation,
minibatch SGD, and early stopping on the validation loss. A test compares the analytic
gradients with numerical gradients.

**Baselines.**
Each run reports the majority class and a logistic regression on the same features.
The network is only useful if it beats them.

**Final model.**
The shipped model trains on all speakers, except a random 20% of speakers that are used
only for early stopping (59 training speakers and 15 early-stopping speakers). It uses
the settings that cross-validation chose. The numbers from the early-stopping speakers
are too optimistic, because they select the weights. The cross-validation summary is
the honest estimate, and `artifacts/metrics.json` stores it next to the model.

**Model file.**
The file holds the weights, the scaler values and the class names together. A model
cannot be used with the wrong preprocessing or the wrong label order.

**Serving.**
Training is a separate script. The API checks each upload (empty, too large, not audio)
and uses a unique temporary file for each request. The web page is one HTML file that
FastAPI serves at `/`. The browser decodes the audio and converts it to 16 kHz mono WAV,
so the server needs no `ffmpeg`. After one request, the service uses about 200 MB of
memory.

**Deployment.**
A Docker image runs the service. GitHub Actions runs the tests, then builds the image,
starts it and calls `/health` and `/`. Render builds the same `Dockerfile` from the
`main` branch. The free host sleeps when idle, so the web page tells the visitor when
the server is starting.

## Results

Dataset: 74 speakers (English 33, American 22, Scottish 19) with 30 clips for each
speaker, 2,220 clips in total (see `dataset.md`). Chance for 3 classes is 0.333.

Final settings: hidden 64, learning rate 0.05, L2 0.1, batch size 32, patience 30,
class weights on, seed 42.

5-fold cross-validation, grouped by speaker (mean ± std over 5 folds):

| Model                          | Balanced accuracy | Accuracy      |
| ------------------------------ | ----------------- | ------------- |
| Majority class                 | 0.333 ± 0.000     | 0.446 ± 0.030 |
| Logistic regression            | 0.545 ± 0.081     | 0.552 ± 0.079 |
| NumPy network (final settings) | 0.536 ± 0.072     | 0.515 ± 0.077 |

Plain accuracy for each fold:

| Fold | Test speakers | Network | Logistic regression | Majority |
| ---- | ------------- | ------- | ------------------- | -------- |
| 1    | 15            | 0.529   | 0.578               | 0.467    |
| 2    | 15            | 0.520   | 0.551               | 0.467    |
| 3    | 15            | 0.384   | 0.418               | 0.467    |
| 4    | 15            | 0.567   | 0.589               | 0.400    |
| 5    | 14            | 0.576   | 0.624               | 0.429    |

`train.py --cv` also writes the pooled confusion matrix and the per-class precision and
recall to `artifacts/cv_metrics.json`.

### L2 strength

Same folds, class weights on. Balanced accuracy of the network:

| L2    | Balanced accuracy | Accuracy      |
| ----- | ----------------- | ------------- |
| 0.001 | 0.458 ± 0.055     | 0.420 ± 0.065 |
| 0.1   | 0.536 ± 0.072     | 0.515 ± 0.077 |
| 0.3   | 0.498 ± 0.079     | 0.440 ± 0.071 |
| 1.0   | 0.333 ± 0.000     | 0.257 ± 0.033 |

At L2 1.0 the network always gives one answer. The penalty is too strong.

Without class weights (L2 0.001), the network scored 0.471 ± 0.081 accuracy and the
logistic regression scored 0.558 ± 0.062. Pooled confusion matrix of that run (rows are
true accents, columns are predicted accents):

|              | American | English | Scottish |
| ------------ | -------- | ------- | -------- |
| **American** | 346      | 176     | 138      |
| **English**  | 202      | 475     | 313      |
| **Scottish** | 103      | 245     | 222      |

Recall was 0.524 for American, 0.480 for English and 0.390 for Scottish.

### Earlier experiments (57 speakers)

I explored settings first on a smaller dataset: 19 speakers for each accent and 100
clips for each speaker, with equal classes. Plain accuracy, 5 folds:

| Setting                                              | Network       | Logistic regression |
| ---------------------------------------------------- | ------------- | ------------------- |
| Default (L2 0.001, hidden 64, 100 clips per speaker) | 0.530 ± 0.051 | 0.602 ± 0.092       |
| L2 0.01                                              | 0.544 ± 0.067 | 0.602 ± 0.092       |
| L2 0.1                                               | 0.555 ± 0.085 | 0.602 ± 0.092       |
| hidden 16                                            | 0.537 ± 0.056 | 0.602 ± 0.092       |
| 30 clips per speaker                                 | 0.516 ± 0.080 | 0.586 ± 0.105       |
| 10 clips per speaker                                 | 0.515 ± 0.065 | 0.527 ± 0.056       |

These numbers are not comparable with the table above. The dataset and the classes are
different.

### Findings

1. Both models are clearly above chance. The network scores 0.536 balanced accuracy,
   and guessing scores 0.333.
2. The network and logistic regression are equal on balanced accuracy. The gap is 0.009,
   and the spread between folds is about 0.07 to 0.08. On plain accuracy, logistic
   regression is higher in all 5 folds, by 0.02 to 0.05.
3. The network learns the training voices. On the earlier dataset, the training loss
   fell to about 0.08 while the validation loss rose from the first epoch. Strong L2
   regularisation fixed part of this problem. It raised the balanced accuracy from 0.458
   to 0.536.
4. I chose L2 on the same folds that I report. The gain is therefore slightly too good.
   Treat 0.536 as an upper bound.
5. Plain accuracy hides the problem with unequal classes. Without class weights, the
   network scored 0.471. This is only 0.025 above the majority class (0.446).
6. Scottish is the hardest accent. In the run without class weights, 313 of 990 English
   clips went to Scottish.
7. Fold 3 is hard for every model. Logistic regression scores only 0.418 there. A few
   unusual speakers can move a fold.
8. Fewer clips for each speaker did not help (earlier experiments). The number of clips
   is not the problem.

The 80 summary features probably limit the accuracy. I have not tested other features,
so this is an inference and not a measured result.

## Limits

- The data has three accents and 74 speakers. See `dataset.md` for the accents that I
  could not include.
- The speech is read speech from a studio. The model is unlikely to work on noisy or
  telephone audio.
- Folds have 14 or 15 test speakers. Results vary by several points between folds.
- MFCC statistics do not keep timing, rhythm, or pitch movement. These carry accent
  information.
- The network does not beat logistic regression on these features. A more complex model
  does not help when the features are the limit.
- The free host sleeps when idle. The first request can take up to a minute.

## Next steps

- Test pretrained speech embeddings (for example wav2vec2) offline, as a reference for
  how much better features can help. The shipped model can stay small.
- Nested cross-validation for the L2 choice, so the reported number is not optimistic.
- More speakers per accent, if a larger open corpus becomes practical.
