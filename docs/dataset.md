# Dataset

## Source

- Corpus: <name, version, URL>
- Licence: <licence and any attribution requirements>
- Accessed: <date>

## Selection

- Accents used: <list>
- Speakers per accent: <counts>
- Rule for choosing speakers: <e.g. first N by ID, or all with >= K recordings>

## Preparation

- Converted to mono 16 kHz WAV
- Cut into <5-10>s clips
- Filenames: `<speaker>_<clip>.wav`
- Rebuild with: `python scripts/prepare_dataset.py ...`
