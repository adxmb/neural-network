# Dataset

## Source

- Corpus: CSTR VCTK Corpus, version 0.92 (Yamagishi, Veaux and MacDonald, 2019).
- Download page: https://datashare.ed.ac.uk/handle/10283/3443 (DOI 10.7488/ds/2645).
- Licence: attribution licence. The Hugging Face dataset card lists CC BY 4.0.
  The DataShare page has the authoritative terms.
- Accessed: November 2026.
- Citation: Yamagishi, J., Veaux, C., and MacDonald, K. (2019). CSTR VCTK Corpus:
  English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92).
  University of Edinburgh. The Centre for Speech Technology Research (CSTR).
  https://doi.org/10.7488/ds/2645

The audio is not stored in this repository. `dataset/`, `dataset_all/` and `data/`
are in `.gitignore`.

## Why VCTK

- Each speaker has an accent label and a speaker ID.
- The file names carry the speaker ID (for example `p225_001`). The speaker-aware
  split needs this.
- All recordings come from one project. This lowers the risk that the model learns
  the microphone instead of the accent.

## Accents in VCTK and accents used

Speakers per accent in VCTK 0.92 (from `prepare_dataset.py --list`):

| Accent        | Speakers | Used |
| ------------- | -------- | ---- |
| english       | 33       | yes  |
| american      | 22       | yes  |
| scottish      | 19       | yes  |
| irish         | 9        | no   |
| canadian      | 8        | no   |
| northernirish | 6        | no   |
| southafrican  | 4        | no   |
| indian        | 3        | no   |
| australian    | 2        | no   |
| newzealand    | 1        | no   |
| welsh         | 1        | no   |
| unknown       | 1        | no   |

I used the three accents with at least 19 speakers. A test on new speakers needs
several speakers for each accent in the training set and in the test set. Accents
with 1 to 9 speakers cannot give a valid test. Australian, New Zealand, South
African and Indian English are not in the model for this reason.

## Selection

- Accent labels: the ACCENTS column of `speaker-info.txt`.
- Speakers per accent: 19. The 19 Scottish speakers are all used. The English and
  American speakers are chosen at random (seed 42).
- Clips per speaker: 100, chosen at random (seed 42).
- Microphone: mic1.
- Total: 57 speakers and 5,700 clips (19 speakers and 1,900 clips for each accent).

## Preparation

- The source files are the silence-trimmed 48 kHz FLAC recordings.
- Each clip is converted to mono, 16 kHz, 16-bit WAV.
- Clips shorter than 0.5 s are skipped. No clip was skipped in the current dataset.
- Each clip is one VCTK utterance (one read sentence). Clips are not re-cut.
- File names are `<speaker>_<utterance>.wav`. The speaker is the part before the
  first underscore.
- `dataset/manifest.csv` lists every clip with its speaker and accent.
- The script downloads only the selected clips. It reads the archive index and uses
  HTTP range requests. It does not download the full archive.

## Rebuild

```bash
python scripts/prepare_dataset.py --remote --list
python scripts/prepare_dataset.py --remote --accents english american scottish \
    --speakers-per-accent 19 --clips-per-speaker 100
```

## Limits of the data

- The speakers read text in a studio. The model has not seen spontaneous speech,
  noise, or telephone audio.
- There are few speakers. A test fold has 11 or 12 speakers. One speaker changes
  the accuracy of a fold by about 8 points.
- The 100 clips from one speaker are not independent. For accent, they are close to
  one sample. The number of speakers limits the model, not the number of clips.
- Accent labels are coarse. Speakers with the same label can sound different.
