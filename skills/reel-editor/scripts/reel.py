#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=11", "numpy>=2", "opencv-python-headless>=4.10"]
# ///
"""reel.py - the short-form editing engine behind the reel-editor skill.

Subcommands
  probe SRC                         what the footage is (size, fps, rotation, log/HDR)
  transcribe SRC -o DIR             word-timed transcript (whisper.cpp large-v3-turbo)
  plan DIR [--gap .35]              auto edit: drop pauses + fillers -> DIR/edl.json
  render CONFIG.json                cut, reframe 9:16, grade, captions, vocal chain, mix
  proof OUT.mp4                     contact sheet + loudness + spec check
  lut slog3 OUT.cube                build the S-Log3/S-Gamut3.Cine -> Rec.709 LUT

Everything is driven by one JSON config so an edit is reproducible and tweakable.
ffmpeg here has no libass/drawtext, so captions are drawn with Pillow and piped in
as an RGBA overlay stream.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
FONTS = SKILL / "assets" / "fonts"
LUTS = SKILL / "assets" / "luts"
WHISPER_MODEL = Path.home() / ".cache/whisper.cpp/ggml-large-v3-turbo-q5_0.bin"
W, H = 1080, 1920
FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "ah", "hmm", "mm", "mhm"}


# ----------------------------------------------------------------------------- utils
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))[:400]}\n{r.stderr[-3000:]}")
    return r


def ffprobe(src):
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(src)])
    return json.loads(r.stdout)


def probe_info(src):
    d = ffprobe(src)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    rot = 0
    for sd in (v or {}).get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = (v["width"], v["height"]) if v else (0, 0)
    if abs(rot) in (90, 270):
        w, h = h, w  # ffmpeg autorotates on decode
    num, den = (v or {}).get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den or 1)
    trc = (v or {}).get("color_transfer", "")
    sidecar = None
    p = Path(src)
    for cand in (p.with_name(p.stem + "M01.XML"), p.with_name(p.stem + "M01.xml")):
        if cand.exists():
            m = re.search(r'CaptureGammaEquation" value="([^"]+)"', cand.read_text(errors="ignore"))
            sidecar = m.group(1) if m else None
    return {
        "path": str(src), "duration": float(d["format"].get("duration", 0)),
        "width": w, "height": h, "fps": round(fps, 3), "rotation_meta": rot,
        "vcodec": (v or {}).get("codec_name"), "pix_fmt": (v or {}).get("pix_fmt"),
        "transfer": trc, "hdr": trc in ("arib-std-b67", "smpte2084"),
        "gamma_sidecar": sidecar, "audio": bool(a),
        "audio_codec": (a or {}).get("codec_name"),
    }


def snap(t, fps):
    return round(t * fps) / fps


# ----------------------------------------------------------------------------- LUT
def build_slog3_lut(out, size=33, exposure=1.0):
    """S-Log3 / S-Gamut3.Cine -> Rec.709 display, from Sony's published curve and primaries,
    with a gentle filmic shoulder so highlights roll off instead of clipping."""
    import numpy as np

    def rgb2xyz(prim, white=(0.3127, 0.3290)):
        def xyz(x, y):
            return np.array([x / y, 1.0, (1 - x - y) / y])
        P = np.stack([xyz(*c) for c in prim], axis=1)
        S = np.linalg.solve(P, xyz(*white))
        return P * S

    sg3c = rgb2xyz([(0.766, 0.275), (0.225, 0.800), (0.089, -0.087)])
    r709 = rgb2xyz([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)])
    M = np.linalg.inv(r709) @ sg3c

    g = np.linspace(0, 1, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # .cube: R fastest
    rgb = np.stack([r, gg, b], axis=-1).reshape(-1, 3)

    x = rgb * 1023.0
    lin = np.where(x >= 171.2102946929,
                   (10.0 ** ((x - 420.0) / 261.5)) * (0.18 + 0.01) - 0.01,
                   (x - 95.0) * 0.01125 / (171.2102946929 - 95.0))
    lin = lin @ M.T
    lin = np.clip(lin * exposure, 0, None)
    a_, b_, c_, d_, e_ = 2.51, 0.03, 2.43, 0.59, 0.14  # Narkowicz filmic fit
    disp = np.clip((lin * (a_ * lin + b_)) / (lin * (c_ * lin + d_) + e_), 0, 1)
    enc = np.where(disp < 0.018, 4.5 * disp, 1.099 * disp ** 0.45 - 0.099)
    with open(out, "w") as f:
        f.write(f'TITLE "SLog3 SGamut3.Cine to Rec709 (reel-editor)"\nLUT_3D_SIZE {size}\n')
        for row in enc:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    return out


def build_hlg_lut(out, size=33, white_nits=203.0, knee=0.78, sat=1.0):
    """iPhone HLG (BT.2100, BT.2020 primaries) -> SDR Rec.709: HLG inverse OETF, 1000-nit OOTF,
    2020->709 gamut, HDR reference white (203 nits) to SDR white, soft highlight shoulder."""
    import numpy as np

    def rgb2xyz(prim, white=(0.3127, 0.3290)):
        def xyz(x, y):
            return np.array([x / y, 1.0, (1 - x - y) / y])
        P = np.stack([xyz(*c) for c in prim], axis=1)
        return P * np.linalg.solve(P, xyz(*white))

    r2020 = rgb2xyz([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)])
    r709 = rgb2xyz([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)])
    M = np.linalg.inv(r709) @ r2020
    g = np.linspace(0, 1, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    E_ = np.stack([r, gg, b], axis=-1).reshape(-1, 3)
    a, bb, c = 0.17883277, 0.28466892, 0.55991073
    E = np.where(E_ <= 0.5, E_ ** 2 / 3.0, (np.exp((E_ - c) / a) + bb) / 12.0)
    Y = (E @ np.array([0.2627, 0.6780, 0.0593]))[:, None]
    nits = 1000.0 * np.power(np.maximum(Y, 1e-6), 0.2) * E           # OOTF, gamma 1.2 at 1000 nits
    x = (nits / white_nits) @ M.T
    if sat != 1.0:
        L = (x @ np.array([0.2126, 0.7152, 0.0722]))[:, None]
        x = L + (x - L) * sat
    x = np.clip(x, 0, None)
    hi = knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee)))  # shoulder into 1.0
    x = np.where(x > knee, hi, x)
    enc = np.clip(x, 0, 1) ** (1 / 2.4)                               # BT.1886 display
    with open(out, "w") as f:
        f.write(f'TITLE "HLG BT2100 to Rec709 SDR (reel-editor)"\nLUT_3D_SIZE {size}\n')
        for row in enc:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    return out


def ensure_hlg_lut():
    LUTS.mkdir(parents=True, exist_ok=True)
    p = LUTS / "hlg-bt2100-to-709.cube"
    if not p.exists():
        build_hlg_lut(p)
    return p


def ensure_slog3_lut():
    LUTS.mkdir(parents=True, exist_ok=True)
    p = LUTS / "slog3-sgamut3cine-to-709.cube"
    if not p.exists():
        build_slog3_lut(p, exposure=0.85)
    return p


# ----------------------------------------------------------------------------- transcribe
def apply_fixes(words, fixes):
    """fixes: 'Acme Co=AcmeCo,co pilot=copilot' -> whole-word, case-insensitive, punctuation kept."""
    for rule in filter(None, (fixes or "").split(",")):
        pat, rep = rule.split("=", 1)
        rx = re.compile(rf"^(?:{pat})([^\w']*)$", re.I)
        for w in words:
            m = rx.match(w["w"])
            if m:
                w["w"] = rep + m.group(1)
    return words


def refine_to_voice(words, wav):
    """Snap word edges to the real voice. Any word after a pause starts where the voice starts;
    any word before a pause ends where the voice stops. Cuts then land on speech, not air."""
    import wave
    import numpy as np
    with wave.open(str(wav)) as f:
        a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
    e = np.sqrt(np.convolve(a ** 2, np.ones(320) / 320, "same")[::160] + 1e-12)  # 10 ms frames
    db = 20 * np.log10(e)
    on = db > np.percentile(db, 8) + 14
    n = len(on)
    idx = lambda t: min(n - 1, max(0, int(t * 100)))
    # A word can't last 1+ s unless it's long: DTW sometimes anchors a word's first token seconds
    # early (seen: "it's" stretched over 4.8 s of silence). Restart it at the voiced run that ends it.
    for w in words:
        if w["e"] - w["s"] > 0.35 + 0.09 * len(w["w"]):
            i, quiet = idx(w["e"]) - 1, 0
            while i > idx(w["s"]):
                quiet = quiet + 1 if not on[i] else 0
                if quiet >= 15:
                    break
                i -= 1
            if i > idx(w["s"]):
                w["s"] = round((i + quiet) / 100, 3)
    for k, w in enumerate(words):
        prev_e = words[k - 1]["e"] if k else 0.0
        nxt_s = words[k + 1]["s"] if k + 1 < len(words) else w["e"] + 1.0
        if k == 0 or w["s"] - prev_e > 0.12:
            for i in range(idx(max(prev_e, w["s"] - 0.3)), idx(w["e"])):
                if on[i:i + 3].all():
                    w["s"] = round(min(i / 100, w["e"] - 0.05), 3)
                    break
        if nxt_s - w["e"] > 0.12 or k + 1 == len(words):
            for i in range(idx(min(nxt_s, w["e"] + 0.3)), idx(w["s"]), -1):
                if on[i]:
                    w["e"] = round(max((i + 1) / 100, w["s"] + 0.05), 3)
                    break
    return words


def transcribe(src, outdir, lang="auto", prompt=None, fixes=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not WHISPER_MODEL.exists():
        sys.exit(f"missing whisper model at {WHISPER_MODEL}")
    wav = outdir / "speech16k.wav"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])

    def whisper(tag, prm):
        base = outdir / f"whisper_{tag}"
        # -mc 0: no text carried between windows (long takes otherwise loop one line x80).
        # --dtw + -nfa: real token alignment. Segment offsets alone drift up to 0.66 s early at phrase
        # starts (measured), which put dead air at the head of cuts and captions ahead of the voice.
        # DTW silently returns nothing with flash attention on, hence -nfa.
        cmd = ["whisper-cli", "-m", str(WHISPER_MODEL), "-f", str(wav), "-ml", "1", "-sow", "-ojf",
               "-of", str(base), "-np", "-l", lang, "-mc", "0", "--dtw", "large.v3.turbo", "-nfa"]
        if prm:
            cmd += ["--prompt", prm]
        run(cmd)
        data = json.loads(Path(f"{base}.json").read_text())
        words = []
        for seg in data.get("transcription", []):
            t = seg["text"].strip()
            if not t:
                continue
            toks = [tk for tk in seg.get("tokens", []) if not tk["text"].startswith("[")]
            dt = [tk["t_dtw"] / 100 for tk in toks if tk.get("t_dtw", -1) >= 0]
            if dt:   # DTW marks each token's centre: start a touch before the first one
                s, e = max(0.0, dt[0] - 0.08), dt[-1] + 0.12
            else:
                s, e = seg["offsets"]["from"] / 1000, seg["offsets"]["to"] / 1000
            ps = [tk.get("p", 1) for tk in toks]
            words.append({"w": t, "s": round(s, 3), "e": round(max(e, s + 0.05), 3),
                          "p": round(sum(ps) / len(ps), 3) if ps else 1.0})
        for k in range(len(words) - 1):  # words never overlap
            if words[k]["e"] > words[k + 1]["s"]:
                words[k]["e"] = max(words[k]["s"] + 0.05, words[k + 1]["s"])
        return data, words

    # A prompt buys punctuation and disfluencies ("um") but can make whisper silently skip speech
    # (seen when audio opens mid-sentence). Run both; keep the prompted pass only if it lost nothing.
    data_p, words_p = whisper("prompted", prompt or "So, um, here's what happened. I mean, it works.")
    data_n, words_n = whisper("plain", None)
    lost = len(words_p) < 0.97 * len(words_n) or (words_n and words_p and words_p[0]["s"] - words_n[0]["s"] > 1.0)
    data, words = (data_n, words_n) if lost else (data_p, words_p)
    if lost:
        print(f"note: prompted pass dropped speech ({len(words_p)} vs {len(words_n)} words); using plain pass")
    words = refine_to_voice(words, wav)
    words = apply_fixes(words, fixes)
    lang_out = data.get("result", {}).get("language", lang)
    (outdir / "words.json").write_text(json.dumps({"source": str(src), "lang": lang_out, "words": words}, indent=1))
    lines = [f"[{i}] {w['s']:7.2f}-{w['e']:6.2f}  {w['w']}" for i, w in enumerate(words)]
    (outdir / "transcript.txt").write_text(" ".join(w["w"] for w in words) + "\n\n" + "\n".join(lines) + "\n")
    print(f"{len(words)} words -> {outdir/'words.json'}  (+ transcript.txt)")
    print(" ".join(w["w"] for w in words)[:1500])


# ----------------------------------------------------------------------------- plan (auto cut)
def silences(src, noise_db=-35, min_d=0.30):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-vn", "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_d}", "-f", "null", "-"],
                       capture_output=True, text=True)
    out, start = [], None
    for line in r.stderr.splitlines():
        if "silence_start" in line:
            start = float(line.split("silence_start:")[1].split()[0])
        elif "silence_end" in line and start is not None:
            out.append((max(0.0, start), float(line.split("silence_end:")[1].split()[0])))
            start = None
    return out


def spans_for(words, keep, src, duration, gap=0.35, pad_in=0.08, pad_out=0.12, noise_db=-35, sil=None):
    """Word indices (in play order) -> source time spans. A skipped word or a pause longer than
    `gap` breaks a span; padding never reaches into a neighbouring word, so cut words can't bleed in."""
    spans = []
    for i in keep:
        if spans and spans[-1][1] == i - 1 and words[i]["s"] - words[i - 1]["e"] <= gap:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    sil = silences(src, noise_db) if sil is None else sil
    out = []
    for a_i, b_i in spans:
        s, e = words[a_i]["s"], words[b_i]["e"]
        lo = words[a_i - 1]["e"] if a_i > 0 else 0.0
        hi = words[b_i + 1]["s"] if b_i + 1 < len(words) else duration
        s2, e2 = max(lo, s - pad_in), min(hi, e + pad_out)
        for a, b in sil:
            if a <= s2 < b and b < e:
                s2 = max(s2, b - pad_in)
            if a < e2 <= b and a > s:
                e2 = min(e2, a + pad_out)
        out.append([round(s2, 3), round(e2, 3)])
    merged = []
    for s, e in out:
        if merged and abs(s - merged[-1][1]) <= 0.04:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def pick(workdir, ranges, gap=0.35):
    """Print segments for word-index ranges, e.g. pick WORK 12-40 55-70 (play order = arg order)."""
    workdir = Path(workdir)
    wj = json.loads((workdir / "words.json").read_text())
    words = wj["words"]
    keep = []
    for r in ranges:
        a, _, b = r.partition("-")
        keep += [i for i in range(int(a), int(b or a) + 1)
                 if re.sub(r"[^\w']", "", words[i]["w"].lower()) not in FILLERS]
    info = probe_info(wj["source"])
    segs = spans_for(words, keep, wj["source"], info["duration"], gap)
    text = " ".join(words[i]["w"] for i in keep)
    print(json.dumps([{"src": wj["source"], "words": str(workdir / "words.json"), "s": s, "e": e} for s, e in segs]))
    print(f"# {sum(e - s for s, e in segs):.2f}s  {text}", file=sys.stderr)


def plan(workdir, gap=0.35, pad_in=0.08, pad_out=0.12, noise_db=-35, drop=None):
    workdir = Path(workdir)
    wj = json.loads((workdir / "words.json").read_text())
    src, words = wj["source"], wj["words"]
    info = probe_info(src)
    drop = set(drop or [])
    keep = []
    for i, w in enumerate(words):
        bare = re.sub(r"[^\w']", "", w["w"].lower())
        if i in drop or bare in FILLERS:
            continue
        keep.append(i)
    if not keep:
        sys.exit("nothing left to keep")
    merged = spans_for(words, keep, src, info['duration'], gap, pad_in, pad_out, noise_db)
    total = sum(e - s for s, e in merged)
    (workdir / "edl.json").write_text(json.dumps({"source": src, "segments": merged}, indent=1))
    print(f"{len(merged)} segments, {total:.1f}s kept of {info['duration']:.1f}s -> {workdir/'edl.json'}")
    cut_words(workdir, merged, words)


def cut_words(workdir, segments, words, fps=30):
    """Map source-time words onto the output timeline -> cut_words.json + cut_transcript.txt."""
    out, t0 = [], 0.0
    for s, e in segments:
        s, e = snap(s, fps), snap(e, fps)
        for w in words:
            mid = (w["s"] + w["e"]) / 2
            if s <= mid < e:
                out.append({"w": w["w"], "s": round(t0 + max(w["s"], s) - s, 3),
                            "e": round(t0 + min(w["e"], e) - s, 3)})
        t0 += e - s
    out = [w for w in out if re.sub(r"[^\w']", "", w["w"].lower()) not in FILLERS]
    Path(workdir, "cut_words.json").write_text(json.dumps(out, indent=1))
    lines = [f"{w['s']:6.2f}  {w['w']}" for w in out]
    Path(workdir, "cut_transcript.txt").write_text(f"OUTPUT TIMELINE  {t0:.2f}s\n" + "\n".join(lines) + "\n")
    return out


# ----------------------------------------------------------------------------- face / reframe
def detect_faces(src, workdir, every=0.5, rotate=0, windows=None):
    """Face boxes sampled every `every` s. `windows` [(s, e), ...] limits decoding to the parts
    the edit uses (a 4K 120 fps HEVC take is slow to decode end to end)."""
    import cv2
    fdir = Path(workdir) / "faceframes"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    vf = f"fps={1/every},scale=640:-2"
    if rotate:
        vf = rotate_filter(rotate) + "," + vf
    frames = []  # (t, path)
    if windows:
        for k, (a, b) in enumerate(windows):
            a0 = max(0.0, a - 0.5)
            run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{a0:.3f}", "-t", f"{b - a0 + 0.5:.3f}",
                 "-i", str(src), "-vf", vf, "-q:v", "4", str(fdir / f"w{k:03d}_%05d.jpg")])
            frames += [(a0 + i * every, f) for i, f in enumerate(sorted(fdir.glob(f"w{k:03d}_*.jpg")))]
    else:
        run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-i", str(src), "-vf", vf, "-q:v", "4",
             str(fdir / "%05d.jpg")])
        frames = [(i * every, f) for i, f in enumerate(sorted(fdir.glob("*.jpg")))]
    det = None
    res = []
    for t_, f in frames:
        img = cv2.imread(str(f))
        hh, ww = img.shape[:2]
        if det is None:
            det = cv2.FaceDetectorYN.create(str(SKILL / "assets/models/face_detection_yunet_2023mar.onnx"),
                                            "", (ww, hh), 0.7, 0.3, 50)
        _, faces = det.detect(img)
        faces = [] if faces is None else [r[:4] for r in faces if r[2] >= ww * 0.03]
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            res.append({"t": t_, "cx": (x + fw / 2) / ww, "cy": (y + fh / 2) / hh, "fh": fh / hh})
        else:
            res.append({"t": t_, "cx": None, "cy": None, "fh": None})
    shutil.rmtree(fdir)
    return res


def auto_rotation(src, workdir):
    """Footage shot with the camera on its side: pick the rotation where faces are found."""
    best, score = 0, -1
    for rot in (0, 90, -90):
        faces = detect_faces(src, workdir, every=2.0, rotate=rot)
        n = sum(1 for f in faces if f["cx"] is not None)
        if n > score:
            best, score = rot, n
    return best


def rotate_filter(rot):
    return {90: "transpose=1", -90: "transpose=2", 270: "transpose=2", 180: "transpose=1,transpose=1"}.get(rot, "null")


def seg_focus(faces, s, e, default=(0.5, 0.42)):
    pts = [f for f in faces if s - 0.25 <= f["t"] <= e + 0.25 and f["cx"] is not None]
    if not pts:
        return default, None
    xs = sorted(p["cx"] for p in pts)
    ys = sorted(p["cy"] for p in pts)
    fh = sorted(p["fh"] for p in pts)
    m = len(pts) // 2
    return (xs[m], ys[m]), fh[m]


def crop_for(src_w, src_h, focus, zoom=1.0, face_y=0.38, out_ar=None):
    """Output-aspect crop window over the (rotated) source, face placed at face_y of frame height."""
    out_ar = out_ar or W / H
    cw = min(src_w, src_h * out_ar) / zoom
    ch = cw / out_ar
    cx = focus[0] * src_w - cw / 2
    cy = focus[1] * src_h - ch * face_y
    cx = min(max(0, cx), src_w - cw)
    cy = min(max(0, cy), src_h - ch)
    even = lambda v: int(v) // 2 * 2
    return even(cw), even(ch), even(cx), even(cy)


# ----------------------------------------------------------------------------- audio
CHAINS = {
    # The six-stage radio/broadcast voice chain, in broadcast order.
    "radio": [
        "highpass=f=80:poles=2",                                   # 1 high-pass: rumble, handling, HVAC
        "agate=mode=downward:threshold=0.012:ratio=2:range=0.3:attack=8:release=180:knee=4",  # 2 downward expander: room tone between words
        "deesser=i=0.35:m=0.5:f=0.55:s=o",                          # 3 de-esser: harsh s / sh
        "equalizer=f=250:t=q:w=1.0:g=-2.5,equalizer=f=3200:t=q:w=1.2:g=2.5,highshelf=f=10000:g=1.5",  # 4 EQ: less mud, more presence + air
        "acompressor=threshold=0.1:ratio=3.5:attack=6:release=90:knee=3:makeup=2",  # 5 compressor: even level
    ],
    # "Spoken into a good mic" without the hollowness: no gate, no denoise (those caused the hollow/echo
    # verdict on 2026-10-16). Close-mic body, less box, presence, air, gentle leveling.
    "warm": [
        "highpass=f=70:poles=2",
        "equalizer=f=140:t=q:w=0.9:g=2.5",     # proximity warmth
        "equalizer=f=380:t=q:w=1.2:g=-2",      # boxiness / small-room tone
        "equalizer=f=3300:t=q:w=1.1:g=2",      # presence
        "highshelf=f=9500:g=1.5",              # air
        "deesser=i=0.25:m=0.5:f=0.55:s=o",
        "acompressor=threshold=0.3:ratio=1.6:attack=20:release=200:knee=6:makeup=1",  # light: more pulls the room up
    ],
    "light": ["highpass=f=70", "acompressor=threshold=0.15:ratio=2:attack=10:release=120:makeup=1.5"],
    "none": [],
}


def loudnorm_measure(path, I, tp):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                        f"loudnorm=I={I}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    return json.loads(m.group(0))


def build_audio(cfg, segs, work):
    """Cut the voice on the same frame grid as the picture, run the chain, normalise, mix."""
    a = cfg.get("audio", {})
    lufs, tp = a.get("lufs", -14), a.get("true_peak", -1.0)
    srcs = list(dict.fromkeys(src for src, _, _ in segs))
    parts, labels = [], []
    for i, (src, s, e) in enumerate(segs):
        k = srcs.index(src)
        parts.append(f"[{k}:a]atrim={s:.6f}:{e:.6f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.012,afade=t=out:st={max(0, e - s - 0.015):.6f}:d=0.015[a{i}]")
        labels.append(f"[a{i}]")
    chain = []
    if a.get("denoise"):
        chain.append("afftdn=nr=12:nf=-40:tn=1")
    chain += CHAINS[a.get("chain", "radio")]
    fc = ";".join(parts) + f";{''.join(labels)}concat=n={len(segs)}:v=0:a=1,aresample=48000"
    if chain:
        fc += "," + ",".join(chain)
    fc += "[out]"
    voice_raw = work / "voice_chain.wav"
    ins = [x for src in srcs for x in ("-i", src)]
    run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", fc, "-map", "[out]",
         "-ac", "1", "-c:a", "pcm_s24le", str(voice_raw)])
    voice = work / "voice.wav"
    if True:  # every chain ends in one static gain: loudnorm's "linear" mode silently turns dynamic when the
              # true-peak target can't be met, which lifts room tone between words (heard as hollow)
        # Untouched mic: one static gain to the loudness target. No EQ, no dynamics, nothing that
        # changes the sound (the radio chain + afftdn read as "hollow echo" on his lav).
        m = loudnorm_measure(voice_raw, lufs, tp)
        gain = lufs - float(m["input_i"])
        peak_after = float(m["input_tp"]) + gain
        af = f"volume={gain:.2f}dB,aresample=48000"
        if peak_after > tp:  # only then, a transparent ceiling
            af += f",alimiter=limit={10 ** (tp / 20):.4f}:attack=1:release=30:level=false"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(voice_raw), "-af", af, "-ac", "2", "-c:a", "pcm_s24le", str(voice)])
        print(f"  audio: chain={a.get('chain', 'radio')}, static gain {gain:+.1f} dB")
        return _mix_beds(cfg, segs, work, voice, tp)
    # Two-pass loudness so the level is exact, then a true-peak ceiling (stage 6: limiter).
    m = loudnorm_measure(voice_raw, lufs, tp)
    ln = (f"loudnorm=I={lufs}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    lim = f"alimiter=limit={10 ** (tp / 20):.4f}:attack=3:release=50:level=false"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(voice_raw), "-af", f"{ln},aresample=48000,{lim}",
         "-ac", "2", "-c:a", "pcm_s24le", str(voice)])

    return _mix_beds(cfg, segs, work, voice, tp)


def _mix_beds(cfg, segs, work, voice, tp):
    music, sfx = cfg.get("music"), cfg.get("sfx", [])
    if not music and not sfx:
        return voice
    total = sum(e - s for _, s, e in segs)
    inputs, fparts, mix = ["-i", str(voice)], [], ["[v0]"]
    fparts.append("[0:a]asplit=2[v0][vkey]")
    n = 1
    if music:
        gain = music.get("gain_db", -21)  # VO -14 LUFS -> bed ~-35; ducked further under speech
        inputs += ["-stream_loop", "-1", "-i", music["file"]]
        fade = music.get("fade_out", 1.5)
        mu = (f"[{n}:a]atrim=start={music.get('start', 0)},asetpts=PTS-STARTPTS,atrim=0:{total:.3f},"
              f"aresample=48000,volume={gain}dB,afade=t=in:d=0.4,afade=t=out:st={max(0, total - fade):.3f}:d={fade}")
        if music.get("duck", True):
            fparts.append(mu + "[mraw]")
            fparts.append("[mraw][vkey]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=350:makeup=1[m]")
        else:
            fparts.append(mu + "[m]")
            fparts.append("[vkey]anullsink")
        mix.append("[m]")
        n += 1
    else:
        fparts.append("[vkey]anullsink")
    for j, fx in enumerate(sfx):
        inputs += ["-i", fx["file"]]
        ms = int(fx["at"] * 1000)
        fparts.append(f"[{n}:a]aresample=48000,volume={fx.get('gain_db', -10)}dB,adelay={ms}|{ms}[s{j}]")
        mix.append(f"[s{j}]")
        n += 1
    fparts.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,"
                  f"alimiter=limit={10 ** (tp / 20):.4f}:level=false[out]")
    mixed = work / "mix.wav"
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fparts), "-map", "[out]",
         "-t", f"{total:.3f}", "-c:a", "pcm_s24le", str(mixed)])
    return mixed


# ----------------------------------------------------------------------------- captions / text
STYLES = {
    # Hormozi-style word pop: heavy caps, thick outline, the spoken word lights up.
    "pop":   {"font": "Montserrat-Black.ttf", "size": 92, "upper": True, "fill": "#FFFFFF", "stroke": 11,
              "stroke_fill": "#000000", "accent": "#FFE14D", "box": None, "max_words": 3, "track": 0, "shadow": True},
    # Dark rounded pill, bold sentence case (the "GitHub," look).
    "box":   {"font": "Inter-Variable.ttf", "weight": 800, "size": 98, "upper": False, "fill": "#FFFFFF", "stroke": 0,
              "accent": None, "box": (18, 18, 20, 215), "radius": 22, "pad": (34, 18), "max_words": 2, "track": -1, "shadow": False},
    # Monospace caps in a black slab, wide tracking ("BUT MOST PEOPLE", "SIX DIFFERENT SKILLS").
    "mono":  {"font": "JetBrainsMono-ExtraBold.ttf", "size": 78, "upper": True, "fill": "#FFFFFF", "stroke": 0,
              "accent": None, "box": (16, 16, 18, 235), "radius": 16, "pad": (38, 22), "max_words": 3, "track": 6, "shadow": False},
    # Full sentence, quiet, white with soft shadow (talk-show / explainer look).
    "clean": {"font": "Inter-Variable.ttf", "weight": 700, "size": 62, "upper": False, "fill": "#FFFFFF", "stroke": 0,
              "accent": None, "box": None, "max_words": 7, "max_chars": 34, "track": -1, "shadow": True},
    # What creators actually type in the IG/CapCut editor: plain white, natural case, thin outline,
    # a few words at a time along the bottom, no colour pop, no animation. (Creator feedback:
    # "make the captions better, less AI generated", captions at the bottom.)
    "native": {"font": "Inter-Variable.ttf", "weight": 650, "size": 66, "upper": False, "fill": "#FFFFFF",
               "stroke": 3, "stroke_fill": "#0A0A0A", "accent": None, "box": None, "max_words": 5,
               "max_chars": 28, "track": -1, "shadow": True, "pop": False},
    # The line at the top: same family, a touch heavier, sentence case, never a slab of caps.
    "native_hook": {"font": "Inter-Variable.ttf", "weight": 780, "size": 70, "upper": False, "fill": "#FFFFFF",
                    "stroke": 3, "stroke_fill": "#0A0A0A", "accent": None, "box": None, "max_words": 12,
                    "track": -1.5, "shadow": True, "pop": False},
}
PROPER = {"claude": "Claude", "ai": "AI", "youtube": "YouTube", "ceo": "CEO", "ceos": "CEOs",
          "roi": "ROI", "i": "I", "i'm": "I'm", "i've": "I've", "i'll": "I'll", "i'd": "I'd", "copilot": "copilot"}


def natural_case(w):
    m = re.match(r"^([\w'$,.-]*?)([.,!?;:]*)$", w)
    core, tail = (m.group(1), m.group(2)) if m else (w, "")
    low = core.lower()
    return PROPER.get(low, low) + tail


class Painter:
    def __init__(self, style, overrides=None):
        from PIL import ImageFont
        st = dict(STYLES[style])
        st.update(overrides or {})
        self.st = st
        self.font = ImageFont.truetype(str(FONTS / st["font"]), st["size"])
        if st.get("weight"):
            try:
                axes = self.font.get_variation_axes()
                vals = [st["weight"] if a.get("name", b"") in (b"Weight", "Weight") else a["default"] for a in axes]
                self.font.set_variation_by_axes(vals)
            except Exception:
                pass

    def text_w(self, s):
        tr = self.st.get("track", 0)
        return self.font.getlength(s) + tr * max(0, len(s) - 1)

    def draw_run(self, d, x, y, s, fill):
        st, tr = self.st, self.st.get("track", 0)
        kw = {}
        if st.get("stroke"):
            kw = {"stroke_width": st["stroke"], "stroke_fill": st["stroke_fill"]}
        if not tr:
            d.text((x, y), s, font=self.font, fill=fill, **kw)
            return
        for ch in s:
            d.text((x, y), ch, font=self.font, fill=fill, **kw)
            x += self.font.getlength(ch) + tr

    def render(self, words, active=None, max_width=None):
        """words: list of str; active: index lit with the accent. Returns a tight RGBA image."""
        from PIL import Image, ImageDraw, ImageFilter
        st = self.st
        max_width = max_width or int(min(W, H * 1.2) * 0.85)
        words = [w.upper() if st["upper"] else w for w in words]
        space = self.text_w(" ")
        lines, cur, cw = [], [], 0
        for i, w in enumerate(words):
            ww = self.text_w(w)
            if cur and cw + space + ww > max_width:
                lines.append((cur, cw))
                cur, cw = [], 0
            cw = cw + (space if cur else 0) + ww
            cur.append((i, w, ww))
        if cur:
            lines.append((cur, cw))
        asc, desc = self.font.getmetrics()
        lh = int((asc + desc) * 1.08)
        padx, pady = st.get("pad", (0, 0))
        sw = st.get("stroke", 0)
        bw = int(max(c for _, c in lines)) + 2 * padx + 2 * sw + 24
        bh = lh * len(lines) + 2 * pady + 2 * sw + 24
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if st.get("box"):
            d.rounded_rectangle([12, 12, bw - 12, bh - 12], radius=st.get("radius", 16), fill=tuple(st["box"]))
        y = 12 + pady + sw
        for line, lw in lines:
            x = (bw - lw) / 2
            for i, w, ww in line:
                fill = st["accent"] if (st.get("accent") and i == active) else st["fill"]
                self.draw_run(d, x, y, w, fill)
                x += ww + space
            y += lh
        if st.get("shadow"):
            a = img.split()[3].filter(ImageFilter.GaussianBlur(10))
            sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
            sh.putalpha(a.point(lambda v: int(v * 0.55)))
            base = Image.new("RGBA", img.size, (0, 0, 0, 0))
            base.alpha_composite(sh, (0, 6))
            base.alpha_composite(img)
            img = base
        return img


def caption_pages(words, style, max_words=None, max_chars=None):
    st = STYLES[style]
    mw = max_words or st["max_words"]
    mc = max_chars or st.get("max_chars", 22)
    pages, cur = [], []
    for i, w in enumerate(words):
        if cur:
            gap = w["s"] - cur[-1]["e"]
            chars = sum(len(x["w"]) + 1 for x in cur) + len(w["w"])
            if len(cur) >= mw or gap > 0.45 or chars > mc or re.search(r"[.!?]$", cur[-1]["w"]):
                pages.append(cur)
                cur = []
        cur.append(w)
    if cur:
        pages.append(cur)
    out = []
    for k, p in enumerate(pages):
        nxt = pages[k + 1][0]["s"] if k + 1 < len(pages) else p[-1]["e"] + 0.5
        out.append({"s": p[0]["s"], "e": min(nxt, p[-1]["e"] + 0.6), "words": p})
    return out


class Overlay:
    """Every on-screen text layer, rendered per frame into one RGBA stream."""

    def __init__(self, cfg, words, total, fps, face_track=None, splits=()):
        from PIL import Image
        self.fps, self.total = fps, total
        self.layers = []
        cap = cfg.get("captions")
        self.cache = {}
        if cap and cap.get("style", "pop") != "off":
            style = cap.get("style", "pop")
            over = {k: cap[k] for k in ("accent", "size", "fill", "font") if k in cap}
            if cap.get("uppercase") is not None:
                over["upper"] = cap["uppercase"]
            p = Painter(style, over)
            pages = caption_pages(words, style, cap.get("max_words"), cap.get("max_chars"))
            ycfg = cap.get("y", "auto")
            pop = cap.get("pop", STYLES[style].get("pop", style in ("pop", "box", "mono")))
            strip = cap.get("strip_punct", style != "clean")
            for pi, pg in enumerate(pages):
                mid = (pg["s"] + pg["e"]) / 2
                if any(a <= mid < b for a, b in splits):
                    y = 0.5  # split screen: captions ride the seam
                else:
                    y = ycfg if ycfg != "auto" else self.auto_y(face_track, mid)
                ws = [w["w"] for w in pg["words"]]
                if strip:
                    ws = [re.sub(r"[.,;:]+$", "", w) or w for w in ws]
                starts = [w["s"] for w in pg["words"]]
                self.layers.append({"s": pg["s"], "e": pg["e"], "y": y, "pop": pop, "painter": p,
                                    "words": ws, "starts": starts, "id": f"cap{pi}"})
        for ti, t in enumerate(([cfg["hook"]] if cfg.get("hook") else []) + cfg.get("texts", [])):
            p = Painter(t.get("style", "mono"), {k: t[k] for k in ("size", "accent", "fill") if k in t})
            self.layers.append({"s": t.get("start", 0), "e": t.get("end", 2.5), "y": t.get("y", 0.2),
                                "pop": STYLES[t.get("style", "mono")].get("pop", True),
                                "painter": p, "words": t["text"].split(), "starts": None, "id": f"txt{ti}",
                                "static": True})
        self.blank = bytes(W * H * 4)
        self.Image = Image

    @staticmethod
    def auto_y(track, t, default=0.66):
        """Keep captions off the face: just under the chin, or above the head when the face sits low.
        Always inside the Reels safe zone (y 0.12 .. 0.75)."""
        hit = next((f for f in (track or []) if f[0] <= t < f[1]), None)
        if not hit:
            return default
        _, fy, fh = hit[0], hit[2], hit[3]
        chin = fy + 0.62 * fh
        below = chin + 0.09
        if below <= 0.75:
            return max(0.58, below)
        return max(0.24, fy - 0.62 * fh - 0.08)

    def frame(self, t):
        live = [L for L in self.layers if L["s"] <= t < L["e"]]
        if not live:
            return self.blank, ("blank",)
        parts, key = [], []
        for L in live:
            active = None
            if L.get("starts"):
                active = max(i for i, s in enumerate(L["starts"]) if s <= t) if t >= L["starts"][0] else 0
            age = int((t - L["s"]) * self.fps)
            step = min(age, 4) if L["pop"] else 4
            parts.append((L, active, step))
            key.append((L["id"], active, step))
        key = tuple(key)
        if key in self.cache:
            return self.cache[key], key
        canvas = self.Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for L, active, step in parts:
            ik = (L["id"], active)
            img = self.cache.get(ik)
            if img is None:
                img = L["painter"].render(L["words"], None if L.get("static") else active)
                self.cache[ik] = img
            if step < 4:  # pop in: 0.82 -> 1.0 over 4 frames with a touch of overshoot
                sc = [0.82, 0.93, 1.04, 1.01][step]
                img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), self.Image.BICUBIC)
            x = (W - img.width) // 2
            y = int(L["y"] * H - img.height / 2)
            canvas.alpha_composite(img, (x, max(0, y)))
        b = canvas.tobytes()
        if len(self.cache) > 600:  # keep memory bounded on long edits
            for k in [k for k in self.cache if isinstance(k, tuple) and len(k) and isinstance(k[0], tuple)][:300]:
                self.cache.pop(k, None)
        self.cache[key] = b
        return b, key


# ----------------------------------------------------------------------------- render
def source_ctx(src, cfg, work, rel, windows=None):
    """Everything per source clip: probe, rotation, colour pipeline, faces, words."""
    info = probe_info(src)
    rot = cfg.get("rotate", 0)
    if rot == "auto":
        rot = auto_rotation(src, work)
        print(f"auto rotation {Path(src).name} -> {rot}")
    sw, sh = info["width"], info["height"]
    if rot in (90, -90, 270):
        sw, sh = sh, sw
    rotf = [rotate_filter(rot)] if rot else []
    pre = []
    color = cfg.get("color", "auto")
    if color == "auto":
        color = ("slog3" if (info.get("gamma_sidecar") or "").startswith("s-log3")
                 else "apple" if info["hdr"] else "none")
    if color == "slog3":
        pre.append(f"lut3d=file='{ensure_slog3_lut()}':interp=tetrahedral")
    elif color == "hlg":
        pre.append(f"lut3d=file='{ensure_hlg_lut()}':interp=tetrahedral")
    elif color.endswith(".cube"):
        pre.append(f"lut3d=file='{rel(color)}':interp=tetrahedral")
    g = cfg.get("grade")
    if g:
        pre.append(f"eq=contrast={g.get('contrast', 1)}:saturation={g.get('saturation', 1)}:"
                   f"brightness={g.get('brightness', 0)}:gamma={g.get('gamma', 1)}")
    if info["hdr"] and color not in ("hlg", "apple") and not color.endswith(".cube"):
        print("WARNING: HDR source without a tone-map; set color to 'apple' or 'hlg'.")
    faces = detect_faces(src, work, 0.5, rot, windows) if cfg.get("reframe", "face") == "face" else []
    return {"src": src, "info": info, "pre": pre, "rot": rotf, "sw": sw, "sh": sh, "faces": faces, "color": color,
            "vertical": sw / sh <= W / H + 0.01, "words": None}


def render(cfg_path):
    global W, H
    cfg_path = Path(cfg_path)
    cfg = json.loads(cfg_path.read_text())
    W, H = cfg.get("size", [1080, 1920])  # [1920, 1080] for landscape
    base = cfg_path.parent
    rel = lambda p: str((base / p).resolve()) if p and not os.path.isabs(p) else p
    if cfg.get("source"):
        cfg["source"] = rel(cfg["source"])
    for k in ("music",):
        if cfg.get(k):
            cfg[k]["file"] = rel(cfg[k]["file"])
    for fx in cfg.get("sfx", []):
        fx["file"] = rel(fx["file"])
    for ins in cfg.get("inserts", []):
        ins["file"] = rel(ins["file"])
    out = Path(rel(cfg.get("out", "out.mp4")))
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = cfg.get("fps", 30)
    work = Path(tempfile.mkdtemp(prefix="reel_", dir=cfg.get("workdir") and rel(cfg["workdir"])))

    # --- EDL items: [s, e] on cfg.source, or {"src", "words", "s", "e", "zoom"} from any clip
    edl = cfg.get("edl")
    if isinstance(edl, str):
        edl = json.loads(Path(rel(edl)).read_text())["segments"]
    if not edl:
        edl = [[0, probe_info(cfg["source"])["duration"]]]
    items = []
    for it in edl:
        if isinstance(it, dict):
            items.append({"src": rel(it.get("src") or cfg["source"]), "words": rel(it.get("words") or cfg.get("words")),
                          "s": it["s"], "e": it["e"], "zoom": it.get("zoom")})
        else:
            items.append({"src": cfg["source"], "words": rel(cfg.get("words")), "s": it[0], "e": it[1], "zoom": None})
    ctx = {}
    for it in items:
        if it["src"] not in ctx:
            wins = [(x["s"], x["e"]) for x in items if x["src"] == it["src"]]
            ctx[it["src"]] = source_ctx(it["src"], cfg, work, rel, wins)
        c = ctx[it["src"]]
        if it["words"] and c["words"] is None:
            wd = json.loads(Path(it["words"]).read_text())
            c["words"] = wd["words"] if isinstance(wd, dict) else wd
    # snapped to the output frame grid so A/V never drifts
    segs = []
    for it in items:
        c = ctx[it["src"]]
        s, e = snap(it["s"], fps), snap(min(it["e"], c["info"]["duration"]), fps)
        if e - s >= 1 / fps:
            segs.append({**it, "s": s, "e": e})
    total = sum(g["e"] - g["s"] for g in segs)

    # --- words on the output timeline
    words, t0 = [], 0.0
    for g in segs:
        for w in ctx[g["src"]]["words"] or []:
            mid = (w["s"] + w["e"]) / 2
            if g["s"] <= mid < g["e"]:
                words.append({"w": w["w"], "s": round(t0 + max(w["s"], g["s"]) - g["s"], 3),
                              "e": round(t0 + min(w["e"], g["e"]) - g["s"], 3)})
        t0 += g["e"] - g["s"]
    words = [w for w in words if re.sub(r"[^\w']", "", w["w"].lower()) not in FILLERS]
    for k, v in (cfg.get("captions", {}).get("fix") or {}).items():
        for w in words:
            w["w"] = re.sub(rf"^{k}(?=[^\w']*$)", v, w["w"], flags=re.I)
    merged_w = []
    for w in words:  # whisper splits "50,000" into "50" "000"
        if merged_w and re.fullmatch(r"0{3}[.,!?]?", w["w"]) and re.fullmatch(r"\$?\d{1,3}", merged_w[-1]["w"]):
            merged_w[-1] = {**merged_w[-1], "w": merged_w[-1]["w"] + "," + w["w"], "e": w["e"]}
        else:
            merged_w.append(w)
    words = merged_w
    natural = cfg.get("captions", {}).get("case") == "natural"
    for w in words:  # whisper's plain pass is all lowercase
        w["w"] = natural_case(w["w"]) if natural else re.sub(r"^i(?=$|'|[^\w])", "I", w["w"])
    Path(work, "cut_words.json").write_text(json.dumps(words, indent=1))

    mode = cfg.get("reframe", "face")
    punch = cfg.get("punch_in", 1.0)
    base_zoom = cfg.get("zoom", 1.0)
    face_y = cfg.get("face_y", 0.36)

    # Picture shots: every real cut, plus "beat" splits at sentence ends / pauses inside long takes
    # so the frame changes every few seconds. Beat splits are contiguous: the sound is untouched.
    beat = cfg.get("beat_every", 3.5)
    shots = []
    for g in segs:
        sw_ = ctx[g["src"]]["words"] or []
        cuts = [g["s"]]
        if punch != 1.0 and beat and g["e"] - g["s"] > beat:
            for j, w in enumerate(sw_):
                t = snap(w["e"], fps)
                nxt = sw_[j + 1] if j + 1 < len(sw_) else None
                pause = nxt is not None and nxt["s"] - w["e"] >= 0.25
                if g["s"] < t < g["e"] and (re.search(r"[.!?,]$", w["w"]) or pause) \
                        and t - cuts[-1] >= beat * 0.6 and g["e"] - t >= 1.2:
                    cuts.append(t)
        # no punctuation or pause to cut on: fall back to the word end nearest each beat
        if punch != 1.0 and beat:
            ends = [snap(w["e"], fps) for w in sw_ if g["s"] < snap(w["e"], fps) < g["e"] - 1.2]
            filled = [cuts[0]]
            for c_ in cuts[1:] + [g["e"]]:
                while c_ - filled[-1] > beat * 1.5:
                    want = filled[-1] + beat
                    cand = [t for t in ends if filled[-1] + beat * 0.6 <= t <= c_ - 1.2]
                    if not cand:
                        break
                    filled.append(min(cand, key=lambda t: abs(t - want)))
                if c_ != g["e"]:
                    filled.append(c_)
            cuts = filled
        cuts.append(g["e"])
        shots += [{**g, "s": a_, "e": b_, "seg_s": g["s"], "seg_e": g["e"]} for a_, b_ in zip(cuts, cuts[1:])]

    # --- 1) picture: one clean file per shot, then a stream copy concat
    seglist = work / "segs.txt"
    face_track, tout = [], 0.0  # (out_start, out_end, face_y_out, face_h_out) for caption placement
    proxies = {}
    pevery = max(2, int(cfg.get("punch_every", 2)))  # punch in on every Nth shot; the rest stay full frame
    with open(seglist, "w") as fl:
        for i, sh_ in enumerate(shots):
            c = ctx[sh_["src"]]
            s, e = sh_["s"], sh_["e"]
            nfr = round((e - s) * fps)
            z = (sh_.get("zoom") or base_zoom) * (punch if (punch != 1.0 and i % pevery == pevery - 1) else 1.0)
            focus, fh = seg_focus(c["faces"], s, e) if c["faces"] else ((0.5, 0.42), None)
            # cheap first: drop to the output rate, rotate, crop; colour only touches kept pixels
            vf = [f"fps={fps}"] + c["rot"]
            if mode == "fit":  # whole frame visible, over a blurred fill of itself
                vf += c["pre"] + [f"split[fa][fb];[fa]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                                  f"boxblur=40:2,eq=brightness=-0.12[bg];[fb]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2*{cfg.get('fit_y', 1.0)}"]
                cw, ch, cx, cy = c["sw"], c["sh"], 0, 0
            elif mode == "none" or (c["vertical"] and z == 1.0 and mode != "face"):
                vf += c["pre"] + [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}"]
                cw, ch, cx, cy = c["sw"], c["sh"], 0, 0
            else:
                cw, ch, cx, cy = crop_for(c["sw"], c["sh"], focus if mode == "face" else (0.5, 0.45), z, face_y)
                vf += [f"crop={cw}:{ch}:{cx}:{cy}"] + c["pre"] + [f"scale={W}:{H}:flags=lanczos"]
            if fh is not None:
                face_track.append((tout, tout + (e - s), (focus[1] * c["sh"] - cy) / ch, fh * c["sh"] / ch))
            tout += e - s
            vf += ["setsar=1", "format=yuv420p"]
            p = work / f"seg{i:04d}.mp4"
            vin, vss = sh_["src"], s
            if c["color"] == "apple":
                # AVFoundation's own HDR->SDR tone map (sample-accurate --start), cached per EDL segment
                key = (sh_["src"], sh_["seg_s"], sh_["seg_e"])
                if key not in proxies:
                    px = work / f"sdr{len(proxies):03d}.mov"
                    run(["avconvert", "--preset", "Preset3840x2160", "--source", sh_["src"], "--output", str(px),
                         "--start", f"{sh_['seg_s']:.4f}", "--duration", f"{sh_['seg_e'] - sh_['seg_s'] + 0.3:.4f}",
                         "--replace"])
                    proxies[key] = px
                vin, vss = str(proxies[key]), s - sh_["seg_s"]
            run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{vss:.6f}", "-i", vin, "-frames:v", str(nfr),
                 "-vf", ",".join(vf), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                 "-g", str(fps), str(p)])
            fl.write(f"file '{p}'\n")
            print(f"  shot {i+1}/{len(shots)}  {Path(sh_['src']).stem} {s:.2f}-{e:.2f}  zoom {z:.2f}", flush=True)
    picture = work / "picture.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(seglist), "-c", "copy", str(picture)])

    # --- 2) sound
    audio = build_audio(cfg, [(g["src"], g["s"], g["e"]) for g in segs], work)

    # --- 3) composite: picture + inserts + text overlay (piped RGBA) + sound
    inputs = ["-i", str(picture), "-i", str(audio)]
    fc, last, n = [], "[0:v]", 2
    for k, ins in enumerate(cfg.get("inserts", [])):
        at, dur, imode = ins["at"], ins["dur"], ins.get("mode", "full")
        is_img = Path(ins["file"]).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        if is_img:
            inputs += ["-loop", "1", "-t", f"{dur}", "-i", ins["file"]]
        else:
            inputs += ["-ss", f"{ins.get('src_in', 0)}", "-t", f"{dur}", "-i", ins["file"]]
        bw, bh = (W, H) if imode == "full" else (W, H // 2)
        kb = ins.get("zoom", 0.06 if is_img else 0)  # slow push-in keeps a still alive
        chain = (f"[{n}:v]scale={bw}:{bh}:force_original_aspect_ratio=increase:flags=lanczos,crop={bw}:{bh},"
                 f"fps={fps},setsar=1,format=yuva420p")
        if kb:
            chain += (f",scale=w='iw*(1+{kb}*t/{dur})':h='ih*(1+{kb}*t/{dur})':eval=frame,"
                      f"crop={bw}:{bh}")
        chain += f",setpts=PTS-STARTPTS+{at}/TB[i{k}]"
        fc.append(chain)
        if imode == "top":
            # speaker drops to the bottom half, insert takes the top half
            fc.append(f"{last}split=2[b{k}a][b{k}b]")
            hit = next((f for f in face_track if f[0] <= at + dur / 2 < f[1]), None)
            fy = hit[2] if hit else 0.4
            hy = int(min(max(0, fy * H - (H // 2) * 0.42), H // 2)) // 2 * 2
            fc.append(f"[b{k}b]crop={W}:{H//2}:0:{hy}[h{k}]")
            fc.append(f"[b{k}a][h{k}]overlay=0:{H//2}:enable='between(t,{at},{at+dur})'[c{k}]")
            fc.append(f"[c{k}][i{k}]overlay=0:0:eof_action=pass:enable='between(t,{at},{at+dur})'[v{k}]")
        else:
            fc.append(f"{last}[i{k}]overlay=0:0:eof_action=pass:enable='between(t,{at},{at+dur})'[v{k}]")
        last = f"[v{k}]"
        n += 1
    inputs += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0"]
    fc.append(f"{last}[{n}:v]overlay=0:0:format=auto,format=yuv420p[vout]")
    nframes = round(total * fps)
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc),
           "-map", "[vout]", "-map", "1:a", "-frames:v", str(nframes),
           "-c:v", "libx264", "-preset", cfg.get("preset", "slow"), "-crf", str(cfg.get("crf", 17)),
           "-profile:v", "high", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-r", str(fps), "-g", str(fps * 2),
           "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", "-shortest", str(out)]
    errlog = open(work / "ffmpeg_final.log", "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errlog)
    splits = [(i["at"], i["at"] + i["dur"]) for i in cfg.get("inserts", []) if i.get("mode") == "top"]
    ov = Overlay(cfg, words, total, fps, face_track, splits)
    try:
        for f in range(nframes):
            b, _ = ov.frame(f / fps)
            proc.stdin.write(b)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    errlog.close()
    if proc.returncode != 0:
        sys.exit("final encode failed:\n" + (work / "ffmpeg_final.log").read_text()[-3000:])
    if not cfg.get("keep_work"):
        shutil.rmtree(work, ignore_errors=True)
    print(f"DONE {out}  {total:.2f}s  {len(segs)} cuts")


# ----------------------------------------------------------------------------- proof
def proof(path, n=12):
    from PIL import Image, ImageDraw, ImageFont
    info = probe_info(path)
    dur = info["duration"]
    tmp = Path(tempfile.mkdtemp())
    thumbs = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        p = tmp / f"{i}.jpg"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
             "-vf", "scale=270:-2", str(p)])
        thumbs.append((t, Image.open(p).convert("RGB")))
    tw, th = thumbs[0][1].size
    cols = 6
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + 30) + 8), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(str(FONTS / "JetBrainsMono-ExtraBold.ttf"), 16)
    for i, (t, im) in enumerate(thumbs):
        x, y = 8 + (i % cols) * (tw + 8), 8 + (i // cols) * (th + 30)
        sheet.paste(im, (x, y + 22))
        d.text((x, y), f"{t:5.2f}s", font=f, fill=(230, 230, 230))
    sp = Path(path).with_suffix(".proof.jpg")
    sheet.save(sp, quality=88)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    I = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    TP = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r.stderr)
    ok = info["width"] == W and info["height"] == H
    print(json.dumps({"file": str(path), "duration": round(dur, 2), "size": f"{info['width']}x{info['height']}",
                      "fps": info["fps"], "spec_9x16_1080": ok, "integrated_lufs": float(I[-1]) if I else None,
                      "true_peak_dbfs": float(TP[-1]) if TP else None,
                      "mb": round(Path(path).stat().st_size / 1e6, 1), "sheet": str(sp)}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("src")
    p = sub.add_parser("transcribe"); p.add_argument("src"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--lang", default="auto"); p.add_argument("--prompt")
    p.add_argument("--fix", help="word fixes, e.g. 'acme co=AcmeCo'")
    p = sub.add_parser("plan"); p.add_argument("dir"); p.add_argument("--gap", type=float, default=0.35)
    p.add_argument("--noise", type=float, default=-35); p.add_argument("--drop", default="",
                   help="comma list of word indices to cut (retakes, flubs) from transcript.txt")
    p = sub.add_parser("pick"); p.add_argument("dir"); p.add_argument("ranges", nargs="+")
    p.add_argument("--gap", type=float, default=0.35)
    p = sub.add_parser("render"); p.add_argument("config")
    p = sub.add_parser("proof"); p.add_argument("file")
    p = sub.add_parser("lut"); p.add_argument("kind", choices=["slog3"]); p.add_argument("out")
    p.add_argument("--exposure", type=float, default=0.85)
    a = ap.parse_args()
    if a.cmd == "probe":
        print(json.dumps(probe_info(a.src), indent=1))
    elif a.cmd == "transcribe":
        transcribe(a.src, a.out, a.lang, a.prompt, a.fix)
    elif a.cmd == "plan":
        drop = []
        for part in filter(None, a.drop.split(",")):
            if "-" in part:
                x, y = part.split("-")
                drop += list(range(int(x), int(y) + 1))
            else:
                drop.append(int(part))
        plan(a.dir, a.gap, noise_db=a.noise, drop=drop)
    elif a.cmd == "pick":
        pick(a.dir, a.ranges, a.gap)
    elif a.cmd == "render":
        render(a.config)
    elif a.cmd == "proof":
        proof(a.file)
    elif a.cmd == "lut":
        build_slog3_lut(a.out, exposure=a.exposure)
        print(a.out)


if __name__ == "__main__":
    main()
