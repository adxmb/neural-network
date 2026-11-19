# Dataset

## Source

- Corpus: CSTR VCTK Corpus, version 0.92 (Yamagishi, Veaux and MacDonald, 2019)
- Download: https://datashare.ed.ac.uk/handle/10283/3443 (DOI: 10.7488/ds/2645)
- Licence: CC BY 4.0. Attribution is required.
- Accessed: November 2026
- Citation: Yamagishi, J., Veaux, C., & MacDonald, K. (2019). CSTR VCTK Corpus:
  English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92).
  University of Edinburgh. The Centre for Speech Technology Research (CSTR).
  https://doi.org/10.7488/ds/2645

The audio is not stored in this repository. The folders `dataset/`, `dataset_all/`
and `data/` are in `.gitignore`.

## Selection

- Accent labels: the ACCENTS column of VCTK's `speaker-info.txt`.
- Accents used: English, American and Scottish.
- Accents excluded: all others. Canadian has 8 speakers. Australian, New Zealand,
  Indian, South African and the rest have 1 to 4. The classes are too small for a test
  on speakers the model has not seen.
- Speakers: every VCTK speaker with one of the three accents.
- Clips per speaker: 30, chosen at random with seed 42.
- Microphone: mic1.

| Accent   | Speakers | Clips |
| -------- | -------- | ----- |
| English  | 33       | 990   |
| American | 22       | 660   |
| Scottish | 19       | 570   |

Total: 2,220 clips from 74 speakers.

The classes have different sizes. Always answering "English" is right for 44.6% of the
clips. For this reason, the main metric is balanced accuracy, and the training loss
uses class weights. See `design.md`.

## Preparation

- Source files are VCTK's silence-trimmed 48 kHz FLAC recordings.
- The script converts each clip to mono 16 kHz, 16-bit WAV. It skips clips shorter than
  0.5 s.
- Each clip is one VCTK utterance (one read sentence). Clips are not re-cut.
- Filenames are `<speaker>_<utterance>.wav` (for example `p225_001.wav`). The speaker is
  the prefix before the first underscore. Training, validation and test data never share
  a speaker.
- `manifest.csv` in the dataset folder lists every clip with its speaker and accent.
- The build can resume after a network error. It skips clips that exist. Use `--force`
  to write into a folder that is not empty.

Rebuild (downloads only the selected clips, about 100 MB):

    python scripts/prepare_dataset.py --remote --list
    python scripts/prepare_dataset.py --remote --accents english american scottish \
        --speakers-per-accent 40 --clips-per-speaker 30 --out dataset_all

The first command prints the number of speakers for each accent.

## Known limits

- The data is read speech. Every speaker reads text. It says little about natural
  speech.
- The recordings are clean and consistent. The model has not seen noisy or
  phone-quality audio.
- Accent labels are coarse. Speakers with one label differ a lot.
- Each class has 19 to 33 speakers. In each cross-validation fold, 14 or 15 speakers are
  tested. Results change by several points from fold to fold (see `design.md`).
- Many speakers read the same sentences. Some VCTK speakers read the same text as
  others, so the model could learn the text instead of the accent. This is not tested.
