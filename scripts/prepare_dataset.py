"""Build dataset/ from the CSTR VCTK corpus (v0.92) without extracting the whole archive.

VCTK has ~110 speakers labelled by accent, all recorded with the same setup,
which is what an accent classifier needs. This script reads the official zip
directly, picks speakers per accent, converts their clips to mono 16 kHz WAV and
writes them as dataset/<accent>/<speaker>_<utterance>.wav, the layout that
accent_recognition.data expects (speaker = filename prefix before the first "_").

Typical use:

    # 1. See how many speakers each accent has (downloads the zip if missing)
    python scripts/prepare_dataset.py --zip data/VCTK-Corpus-0.92.zip --download --list

    # 2. Build the dataset from the accents you choose
    python scripts/prepare_dataset.py --zip data/VCTK-Corpus-0.92.zip \\
        --accents english scottish american --speakers-per-accent 15

Corpus: https://datashare.ed.ac.uk/handle/10283/3443  (DOI 10.7488/ds/2645)
Check the licence and citation requirements on that page before publishing results.
"""
import argparse
import csv
import io
import random
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf

from accent_recognition.features import MIN_SECONDS, SAMPLE_RATE

VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
AUDIO_RE = re.compile(r"(?:^|/)wav48_silence_trimmed/(p\d+)/(p\d+_\d+)_(mic[12])\.flac$")


def download(url, dest):
    """Stream a large file to disk, writing to .part first so a failed run leaves no half-file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\rdownloading {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    part.rename(dest)
    print()


def read_speaker_info(zf):
    """Map speaker folder id (e.g. 'p225') -> lower-cased accent label."""
    name = next((n for n in zf.namelist() if n.endswith("speaker-info.txt")), None)
    if name is None:
        raise SystemExit("speaker-info.txt not found in the archive")
    lines = zf.read(name).decode("utf-8", errors="replace").splitlines()
    header = lines[0].split()
    col = header.index("ACCENTS") if "ACCENTS" in header else 3
    speakers = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) <= col:
            continue
        sid = parts[0] if parts[0].startswith("p") else f"p{parts[0]}"
        speakers[sid] = parts[col].lower()
    return speakers


def index_audio(zf, mic):
    """Map speaker id -> sorted [(utterance id, zip member)] for one microphone."""
    index = defaultdict(list)
    for member in zf.namelist():
        m = AUDIO_RE.search(member)
        if m and m.group(3) == mic:
            index[m.group(1)].append((m.group(2), member))
    return {spk: sorted(items) for spk, items in index.items()}


def folder_name(accent):
    return re.sub(r"[^a-z0-9]+", "-", accent.lower()).strip("-")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, default=Path("data/VCTK-Corpus-0.92.zip"))
    p.add_argument("--download", action="store_true", help="fetch the zip if it is missing")
    p.add_argument("--out", type=Path, default=Path("dataset"))
    p.add_argument("--list", action="store_true", help="print speakers per accent and exit")
    p.add_argument("--accents", nargs="+", help="accents to include (default: every accent "
                                                "with at least --min-speakers speakers)")
    p.add_argument("--min-speakers", type=int, default=8)
    p.add_argument("--speakers-per-accent", type=int, default=15,
                   help="cap, so classes are balanced by speaker")
    p.add_argument("--clips-per-speaker", type=int, default=30)
    p.add_argument("--mic", choices=["mic1", "mic2"], default="mic1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="write into a non-empty --out")
    args = p.parse_args()

    if not args.zip.exists():
        if not args.download:
            raise SystemExit(f"{args.zip} not found. Re-run with --download to fetch it "
                             f"(large file), or point --zip at an existing copy.")
        print(f"downloading {VCTK_URL}")
        download(VCTK_URL, args.zip)

    with zipfile.ZipFile(args.zip) as zf:
        accent_of = read_speaker_info(zf)
        audio = index_audio(zf, args.mic)

        by_accent = defaultdict(list)
        for spk, items in audio.items():
            if spk in accent_of:
                by_accent[accent_of[spk]].append(spk)

        print(f"{'accent':<16}{'speakers':>9}")
        for accent, spks in sorted(by_accent.items(), key=lambda kv: -len(kv[1])):
            print(f"{accent:<16}{len(spks):>9}")
        if args.list:
            return

        if args.accents:
            chosen = [a.lower() for a in args.accents]
            unknown = [a for a in chosen if a not in by_accent]
            if unknown:
                raise SystemExit(f"unknown accents: {unknown}. See the table above.")
        else:
            chosen = sorted(a for a, s in by_accent.items() if len(s) >= args.min_speakers)
        too_small = [a for a in chosen if len(by_accent[a]) < args.min_speakers]
        if too_small:
            print(f"warning: fewer than {args.min_speakers} speakers for {too_small}; "
                  f"a speaker-independent test set for these will be very noisy")
        if len(chosen) < 2:
            raise SystemExit("need at least two accents to build a dataset")

        if args.out.exists() and any(args.out.iterdir()) and not args.force:
            raise SystemExit(f"{args.out} is not empty. Use --force to write into it, or "
                             f"remove the old dataset first.")

        rng = random.Random(args.seed)
        manifest, per_accent = [], {}
        for accent in chosen:
            speakers = sorted(by_accent[accent])
            if len(speakers) > args.speakers_per_accent:
                speakers = sorted(rng.sample(speakers, args.speakers_per_accent))
            out_dir = args.out / folder_name(accent)
            out_dir.mkdir(parents=True, exist_ok=True)
            n_clips = 0
            for spk in speakers:
                clips = audio[spk]
                if len(clips) > args.clips_per_speaker:
                    clips = sorted(rng.sample(clips, args.clips_per_speaker))
                for utt, member in clips:
                    y, sr = sf.read(io.BytesIO(zf.read(member)), dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    if sr != SAMPLE_RATE:
                        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
                    if len(y) < MIN_SECONDS * SAMPLE_RATE:
                        continue
                    dest = out_dir / f"{utt}.wav"
                    sf.write(dest, y, SAMPLE_RATE, subtype="PCM_16")
                    manifest.append((str(dest.relative_to(args.out)), spk, accent))
                    n_clips += 1
            per_accent[accent] = (len(speakers), n_clips)
            print(f"{accent}: {len(speakers)} speakers, {n_clips} clips")

    with open(args.out / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "speaker", "accent"])
        writer.writerows(manifest)

    total_spk = sum(s for s, _ in per_accent.values())
    total_clips = sum(c for _, c in per_accent.values())
    print(f"\nwrote {total_clips} clips from {total_spk} speakers to {args.out}/")
    print("for docs/dataset.md: VCTK 0.92, mic=%s, seed=%d, speakers/accent<=%d, clips/speaker<=%d"
          % (args.mic, args.seed, args.speakers_per_accent, args.clips_per_speaker))


if __name__ == "__main__":
    main()