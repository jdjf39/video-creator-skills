#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy<2.3", "pyworld>=0.3.4", "soundfile>=0.12", "setuptools<81"]
# ///
"""Natural-sounding voice disguise for one speaker on a mixed recording (no API, no credits).

WORLD vocoder: a small pitch shift plus a vocal-tract (formant) warp, so the voice reads as a
different person instead of a chipmunk or a monster. Only the given time ranges change; the
seams crossfade, so the other speaker is untouched.

voice_disguise.py IN.mp4 OUT.mp4 --ranges 12.3-18.9,22.0-31.4 [--pitch 0.93 --formant 0.93]
"""
import argparse, subprocess, tempfile
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

SR = 48000


def convert(x, pitch, formant):
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, SR, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    n = sp.shape[1]
    src = np.clip(np.arange(n) / formant, 0, n - 1)  # formant < 1 = longer vocal tract (deeper timbre)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, n - 1); fr = src - lo
    sp2 = sp[:, lo] * (1 - fr) + sp[:, hi] * fr
    ap2 = ap[:, lo] * (1 - fr) + ap[:, hi] * fr
    y = pw.synthesize(f0 * pitch, np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), SR, 5.0)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    # match loudness of the original segment
    rx, ry = np.sqrt(np.mean(x ** 2) + 1e-12), np.sqrt(np.mean(y ** 2) + 1e-12)
    return y * (rx / ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--pitch", type=float, default=0.93)
    ap.add_argument("--formant", type=float, default=0.93)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    wav = tmp / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
    x, _ = sf.read(wav)
    y = x.copy()
    fade = int(0.03 * SR)
    for r in a.ranges.split(","):
        s, e = (float(v) for v in r.split("-"))
        i0, i1 = max(0, int(s * SR)), min(len(x), int(e * SR))
        if i1 - i0 < SR // 10:
            continue
        seg = convert(x[i0:i1], a.pitch, a.formant)
        w = np.ones(i1 - i0)
        k = min(fade, (i1 - i0) // 4)
        w[:k] = np.linspace(0, 1, k); w[-k:] = np.linspace(1, 0, k)
        y[i0:i1] = x[i0:i1] * (1 - w) + seg * w
    out_wav = tmp / "b.wav"
    sf.write(out_wav, y, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-i", str(out_wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ac", "2", a.out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
