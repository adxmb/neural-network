"""Build dataset/ from the CSTR VCTK corpus (v0.92) without extracting the whole archive.

VCTK has ~110 speakers labelled by accent, all recorded by one project, which is
what an accent classifier needs. This script reads the official zip, picks
speakers per accent, converts their clips to mono 16 kHz WAV and writes them as
dataset/<accent>/<speaker>_<utterance>.wav, the layout that accent_recognition.data
expects (speaker = filename prefix before the first "_").

Two ways to read the archive:

  --remote   Read the zip's index over HTTP and download ONLY the clips you
             select (a few hundred MB instead of the ~12 GB zip). Needs a server
             that supports HTTP range requests (the Edinburgh server does).
  --zip      Use a local copy of the full zip (add --download to fetch it).

Typical use:

    # 1. See how many speakers each accent has (reads only the index)
    python scripts/prepare_dataset.py --remote --list

    # 2. Build the dataset from the accents you choose
    python scripts/prepare_dataset.py --remote \\
        --accents english scottish american --speakers-per-accent 15

Corpus: https://datashare.ed.ac.uk/handle/10283/3443  (DOI 10.7488/ds/2645)
Check the licence and citation requirements on that page before publishing results.
"""
import argparse
import csv
import io
import random
import re
import struct
import time
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import soundfile as sf

from accent_recognition.features import MIN_SECONDS, SAMPLE_RATE

VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
AUDIO_RE = re.compile(r"(?:^|/)wav48_silence_trimmed/(p\d+)/(p\d+_\d+)_(mic[12])\.flac$")


# ---------------------------------------------------------------- remote zip access
class RangeClient:
    """Fetch byte ranges of a URL. Resolves redirects once, then reuses the final URL."""

    def __init__(self, url):
        self.url = url
        self._resolve()

    def _resolve(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 206:
                raise SystemExit("This server does not support range requests. Download the "
                                 "full zip instead: --zip PATH --download")
            self.final_url = resp.geturl()
            self.size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])

    def get(self, start, length):
        """Return bytes [start, start + length), clamped to the file size, with retries."""
        end = min(start + length, self.size) - 1
        want = end - start + 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.final_url,
                                             headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                    status = resp.status
                if status == 206 and len(data) == want:
                    return data
                raise IOError(f"short read: {len(data)} of {want} bytes (status {status})")
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                try:
                    self._resolve()  # the final URL may have expired
                except Exception:
                    pass


class HTTPRangeFile:
    """Minimal read-only seekable file over HTTP ranges: enough for zipfile to read the index."""

    def __init__(self, client):
        self.client = client
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.client.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.client.size - self.pos
        if n == 0 or self.pos >= self.client.size:
            return b""
        data = self.client.get(self.pos, n)
        self.pos += len(data)
        return data

    def close(self):
        pass


def fetch_member(client, info):
    """Download and decompress one zip member using only a ranged request for its bytes."""
    guess = 30 + len(info.filename.encode("utf-8")) + 512  # local header + typical extra field
    buf = client.get(info.header_offset, guess + info.compress_size)
    if struct.unpack("<I", buf[:4])[0] != 0x04034B50:
        raise IOError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", buf[26:30])
    start = 30 + name_len + extra_len
    if len(buf) < start + info.compress_size:  # unusually long extra field
        buf = client.get(info.header_offset, start + info.compress_size)
    raw = buf[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported compression method {info.compress_type}")


class Source:
    """Uniform access to the archive, whether it is a local zip or read over HTTP."""

    def __init__(self, zf, client=None):
        self.zf = zf
        self.client = client
        self.infos = {i.filename: i for i in zf.infolist()}

    def names(self):
        return list(self.infos)

    def read(self, name):
        if self.client is not None:
            return fetch_member(self.client, self.infos[name])
        return self.zf.read(name)

    def read_many(self, names, workers=8, batch=64):
        """Yield member bytes in order. Remote fetches run in parallel, in bounded batches."""
        if self.client is None:
            for name in names:
                yield self.zf.read(name)
            return
        with ThreadPoolExecutor(workers) as pool:
            for i in range(0, len(names), batch):
                yield from pool.map(self.read, names[i:i + batch])


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


def open_source(args):
    if args.remote:
        print("reading the archive index over HTTP (this can take a minute) ...")
        client = RangeClient(args.url)
        return Source(zipfile.ZipFile(HTTPRangeFile(client)), client)
    if not args.zip.exists():
        if not args.download:
            raise SystemExit(f"{args.zip} not found. Use --remote to fetch only the clips you "
                             f"need, or --download to fetch the full zip.")
        print(f"downloading {args.url}")
        download(args.url, args.zip)
    return Source(zipfile.ZipFile(args.zip))


# ---------------------------------------------------------------- corpus metadata
def read_speaker_info(src):
    """Map speaker folder id (e.g. 'p225') -> lower-cased accent label."""
    name = next((n for n in src.names() if n.endswith("speaker-info.txt")), None)
    if name is None:
        raise SystemExit("speaker-info.txt not found in the archive")
    lines = src.read(name).decode("utf-8", errors="replace").splitlines()
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


def index_audio(names, mic):
    """Map speaker id -> sorted [(utterance id, zip member)] for one microphone."""
    index = defaultdict(list)
    for member in names:
        m = AUDIO_RE.search(member)
        if m and m.group(3) == mic:
            index[m.group(1)].append((m.group(2), member))
    return {spk: sorted(items) for spk, items in index.items()}


def folder_name(accent):
    return re.sub(r"[^a-z0-9]+", "-", accent.lower()).strip("-")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote", action="store_true",
                   help="download only the selected clips via HTTP range requests")
    p.add_argument("--url", default=VCTK_URL, help="archive URL (default: Edinburgh DataShare)")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads in --remote mode")
    p.add_argument("--zip", type=Path, default=Path("data/VCTK-Corpus-0.92.zip"))
    p.add_argument("--download", action="store_true", help="fetch the full zip if it is missing")
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

    src = open_source(args)
    accent_of = read_speaker_info(src)
    audio = index_audio(src.names(), args.mic)

    by_accent = defaultdict(list)
    for spk in audio:
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

    # Decide exactly which clips to use before fetching anything.
    rng = random.Random(args.seed)
    plan, per_accent = [], {}
    for accent in chosen:
        speakers = sorted(by_accent[accent])
        if len(speakers) > args.speakers_per_accent:
            speakers = sorted(rng.sample(speakers, args.speakers_per_accent))
        for spk in speakers:
            clips = audio[spk]
            if len(clips) > args.clips_per_speaker:
                clips = sorted(rng.sample(clips, args.clips_per_speaker))
            plan.extend((accent, spk, utt, member) for utt, member in clips)
        per_accent[accent] = [len(speakers), 0]

    print(f"fetching and converting {len(plan)} clips ...")
    manifest = []
    members = [member for _, _, _, member in plan]
    for i, ((accent, spk, utt, _), data) in enumerate(
            zip(plan, src.read_many(members, args.workers)), 1):
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(y) >= MIN_SECONDS * SAMPLE_RATE:  # too-short clips are skipped
            dest = args.out / folder_name(accent) / f"{utt}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, y, SAMPLE_RATE, subtype="PCM_16")
            manifest.append((str(dest.relative_to(args.out)), spk, accent))
            per_accent[accent][1] += 1
        if i % 50 == 0 or i == len(plan):
            print(f"\r{i}/{len(plan)} clips", end="", flush=True)
    print()

    for accent, (n_spk, n_clips) in per_accent.items():
        print(f"{accent}: {n_spk} speakers, {n_clips} clips")
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