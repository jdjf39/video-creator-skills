#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["faster-whisper>=1.0", "yt-dlp>=2024.1"]
# ///
"""watch.py - let the agent WATCH and LISTEN to any video: a URL (YouTube, TikTok, Instagram,
X, Vimeo... anything yt-dlp supports) or a local file.

  watch.py SRC [-o DIR] [--model small] [--lang auto] [--frames 24] [--scene 0.3]

Writes into DIR (default ./watch/<name>):
  video.<ext>        the downloaded source (local files are linked, not copied)
  audio.wav          16 kHz mono
  transcript.txt     [mm:ss.s] line per segment
  transcript.json    segments with word timings
  frames/            stills at scene changes (falls back to even spacing), named by timestamp
  contact.jpg        one contact sheet of those stills -> open it to SEE the video
  info.json          duration, size, fps, title, uploader, url
Needs ffmpeg + ffprobe on PATH. Python deps install themselves through uv.
"""
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def ts(s):
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def fetch(src, out):
    if Path(src).expanduser().exists():
        p = Path(src).expanduser().resolve()
        dst = out / f"video{p.suffix}"
        if not dst.exists():
            dst.symlink_to(p)
        return dst, {"title": p.stem, "source": str(p)}
    import yt_dlp
    opts = {"outtmpl": str(out / "video.%(ext)s"), "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
            "merge_output_format": "mp4", "quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as y:
        meta = y.extract_info(src, download=True)
    vid = next(out.glob("video.*"))
    keep = ("title", "uploader", "channel", "upload_date", "description", "view_count", "like_count", "webpage_url")
    return vid, {k: meta.get(k) for k in keep if meta.get(k) is not None}


def probe(vid):
    j = json.loads(run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
                        "-of", "json", str(vid)]).stdout)
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), {})
    n, d = (v.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
    return {"duration": float(j["format"]["duration"]), "width": v.get("width"), "height": v.get("height"),
            "fps": round(float(n) / float(d or 1), 3) if float(d or 1) else None,
            "has_audio": any(s.get("codec_type") == "audio" for s in j["streams"])}


def transcribe(vid, out, model, lang):
    wav = out / "audio.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    import wave
    import numpy as np
    from faster_whisper import WhisperModel
    with wave.open(str(wav)) as w:  # decode ourselves: faster-whisper's PyAV path breaks across av versions
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    m = WhisperModel(model, device="auto", compute_type="int8")
    segs, info = m.transcribe(pcm, language=None if lang == "auto" else lang, word_timestamps=True, vad_filter=True)
    rows = []
    for s in segs:
        rows.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                     "words": [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for w in (s.words or [])]})
    (out / "transcript.json").write_text(json.dumps({"language": info.language, "segments": rows}, indent=1))
    (out / "transcript.txt").write_text("\n".join(f"[{ts(r['start'])}] {r['text']}" for r in rows) + "\n")
    return info.language, len(rows)


def frames(vid, out, dur, n, scene):
    fd = out / "frames"
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir()
    times = []
    try:  # scene changes first
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(vid), "-vf", f"select='gt(scene,{scene})',showinfo",
                            "-an", "-f", "null", "-"], text=True, capture_output=True)
        times = [float(t) for t in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    except Exception:
        pass
    n = max(2, min(n, int(dur / 1.5) or 2))  # no more than one still per 1.5 s
    times = sorted(set([min(0.5, dur / 2)] + times))
    if len(times) > n:  # thin evenly
        times = [times[round(i * (len(times) - 1) / (n - 1))] for i in range(n)]
    while len(times) < min(n, 8) and dur > 2:  # too few cuts: pad with even spacing
        times = sorted(set(times + [dur * (i + 0.5) / n for i in range(n)]))[:n]
    for t in times:
        t = min(t, max(dur - 0.1, 0))
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(vid), "-frames:v", "1", "-vf", "scale=640:-2",
             str(fd / f"{ts(t).replace(':', 'm')}s.jpg")])
    cols = 4 if len(times) > 9 else 3
    rows = -(-len(times) // cols)
    run(["ffmpeg", "-y", "-v", "error", "-pattern_type", "glob", "-i", str(fd / "*.jpg"), "-vf",
         f"scale=480:-2,pad=ceil(iw/2)*2:ceil(ih/2)*2,tile={cols}x{rows}:padding=6:margin=6", "-frames:v", "1",
         str(out / "contact.jpg")])
    return len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out")
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3 (bigger = slower, better)")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--scene", type=float, default=0.3, help="scene-change threshold 0-1 (lower = more frames)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found: install ffmpeg (brew install ffmpeg / apt install ffmpeg / winget install ffmpeg)")
    name = re.sub(r"[^\w-]+", "-", Path(a.src).stem if Path(a.src).expanduser().exists() else a.src.rstrip("/").split("/")[-1])[:60] or "video"
    out = Path(a.out or f"watch/{name}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    vid, meta = fetch(a.src, out)
    meta.update(probe(vid))
    (out / "info.json").write_text(json.dumps(meta, indent=1))
    print(f"video   {vid}  {meta['duration']:.1f}s  {meta['width']}x{meta['height']}  {meta['fps']} fps")
    if not a.no_audio and meta["has_audio"]:
        lang, nseg = transcribe(vid, out, a.model, a.lang)
        print(f"heard   {nseg} segments ({lang}) -> {out / 'transcript.txt'}")
    if not a.no_frames and meta["width"]:
        k = frames(vid, out, meta["duration"], a.frames, a.scene)
        print(f"saw     {k} frames -> {out / 'contact.jpg'}")
    print(f"done    {out}")


if __name__ == "__main__":
    main()
