# Dataset

## Source

- Corpus: CSTR VCTK Corpus, version 0.92 (Yamagishi, Veaux & MacDonald, 2019)
- Download: https://datashare.ed.ac.uk/handle/10283/3443 (DOI: 10.7488/ds/2645)
- Licence: <confirm on the DataShare page; attribution required>
- Accessed: <date>
- Citation: Yamagishi, J., Veaux, C., & MacDonald, K. (2019). CSTR VCTK Corpus:
  English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92).
  University of Edinburgh. The Centre for Speech Technology Research (CSTR).
  https://doi.org/10.7488/ds/2645

The audio is not stored in this repository (`dataset/` and `data/` are gitignored).

## Selection

- Accent labels: the ACCENTS column of VCTK's `speaker-info.txt`.
- Accents used: <list>
- Accents excluded: <list, and why, e.g. fewer than 8 speakers>
- Speakers per accent: <accent: n, accent: n, ...> (capped at <N>, chosen at random with seed 42)
- Clips per speaker: up to <N>, chosen at random with seed 42
- Microphone: mic1

| Accent   | Speakers | Clips |
| -------- | -------- | ----- |
| <accent> | <n>      | <n>   |

Total: <N> clips from <N> speakers.

## Preparation

- Source files are VCTK's silence-trimmed 48 kHz FLAC recordings.
- Converted to mono 16 kHz, 16-bit WAV. Clips shorter than 0.5 s are skipped.
- Each clip is one VCTK utterance (a single read sentence); clips are not re-cut.
- Filenames are `<speaker>_<utterance>.wav` (e.g. `p225_001.wav`), so the speaker
  is the prefix before the first underscore. Train, validation and test sets
  never share a speaker.
- `dataset/manifest.csv` lists every clip with its speaker and accent.

Rebuild:

    python scripts/prepare_dataset.py --zip data/VCTK-Corpus-0.92.zip --download --list
    python scripts/prepare_dataset.py --zip data/VCTK-Corpus-0.92.zip \
        --accents <accents> --speakers-per-accent <N> --clips-per-speaker <N>

## Known limitations

- Read speech, with every speaker reading text, so it says little about
  spontaneous speech.
- Recordings are clean and consistent, so the model has not seen noisy or
  phone-quality audio and is unlikely to transfer to it.
- Accent labels are coarse, and speakers within one label vary widely.
- With roughly <N> speakers per accent, a single test split contains only a few
  speakers per accent, so results are noisy (see the variance note in
  design.md).
