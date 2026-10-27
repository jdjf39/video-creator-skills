#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "opencv-python-headless>=4.10"]
# ///
"""Blur one person on a recorded video call: their face (recognised, not just detected, so the
other people stay sharp) and every on-screen label with their name. Picture only; audio is copied.

anonymize.py SRC OUT --start 80 --end 540 --ref-time 100 --ref-pos top --name "alex|alexander"
  --ref-time/--ref-pos: a moment where the person's face is in that half of the frame (top/bottom/
  left/right); the other face in that frame becomes the "keep sharp" reference.
"""
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "assets" / "models"


def frames_at(src, start, dur, fps, outdir, scale=None):
    vf = f"fps={fps}" + (f",scale={scale}" if scale else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{dur}", "-i", str(src), "-vf", vf,
                    "-q:v", "3", str(outdir / "%06d.jpg")], check=True)
    return sorted(outdir.glob("*.jpg"))


class Faces:
    def __init__(self, w, h):
        self.det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (w, h), 0.6, 0.3, 200)
        self.rec = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")

    def __call__(self, img):
        _, faces = self.det.detect(img)
        out = []
        for f in (faces if faces is not None else []):
            if f[2] < 18:
                continue
            emb = self.rec.feature(self.rec.alignCrop(img, f)).flatten()
            out.append((f[:4].astype(float), emb / (np.linalg.norm(emb) + 1e-9)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--ref-time", type=float, required=True); ap.add_argument("--ref-pos", default="top")
    ap.add_argument("--name", default=None, help="regex of on-screen name labels to blur")
    ap.add_argument("--fps", type=float, default=5)
    a = ap.parse_args()
    src = Path(a.src)
    W, H = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", str(src)],
                                   capture_output=True, text=True).stdout.strip().split(","))
    tmp = Path(tempfile.mkdtemp())
    faces = Faces(W, H)

    # reference embeddings: target (to blur) and others (keep sharp)
    rd = tmp / "ref"; rd.mkdir()
    ref_img = cv2.imread(str(frames_at(src, a.ref_time, 0.2, 5, rd)[0]))
    found = faces(ref_img)
    if len(found) < 1:
        raise SystemExit("no faces at --ref-time")
    def side(b):
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        return {"top": cy < H / 2, "bottom": cy >= H / 2, "left": cx < W / 2, "right": cx >= W / 2}[a.ref_pos]
    target = [e for b, e in found if side(b)]
    others = [e for b, e in found if not side(b)]
    if not target:
        raise SystemExit("no face on the --ref-pos side at --ref-time")
    print(f"reference: {len(target)} target, {len(others)} other face(s)")

    # sample the range: faces at --fps, name labels at 1 fps
    sd = tmp / "s"; sd.mkdir()
    shots = frames_at(src, a.start, a.end - a.start, a.fps, sd)
    samples = []  # (t, [boxes])
    for i, p in enumerate(shots):
        img = cv2.imread(str(p))
        boxes = []
        for b, e in faces(img):
            st = max(float(e @ t) for t in target)
            so = max([float(e @ o) for o in others] or [-1])
            if st > 0.30 and st > so:  # SFace cosine: >0.36 is a confident match
                x, y, w, h = b
                # hair above, ears either side, chin below; tight underneath so it never spills into the next tile
                boxes.append([x - w * 0.45, y - h * 0.7, w * 1.9, h * 1.95])
                # Meet's name label sits below-left of the face in every tile size (measured: big tile, main
                # tile, the tiny tile inside a shared screen, where OCR can't read it)
                boxes.append([x - w * 1.75, y + h * 0.9, w * 1.5, h * 0.62])
        samples.append([i / a.fps, boxes])
    if a.name:
        nd = tmp / "n"; nd.mkdir()
        # 2x frames: labels on nested tiles (a shared screen showing the call) are ~10 px tall
        ns = frames_at(src, a.start, a.end - a.start, 1, nd, scale=f"{W * 2}:{H * 2}:flags=lanczos")
        r = subprocess.run([str(HERE / "ocr_boxes"), a.name, *map(str, ns)], capture_output=True, text=True)
        name_boxes = [[[v / 2 for v in b] for b in json.loads(l)["boxes"]] for l in r.stdout.splitlines() if l.strip()]
    else:
        name_boxes = []
    n_face = sum(1 for _, b in samples if b)
    print(f"{len(samples)} samples, face blurred in {n_face}, name labels in {sum(1 for b in name_boxes if b)} of {len(name_boxes)}")

    def boxes_at(t):
        k = int(t * a.fps)
        out = []
        for j in (k - 1, k, k + 1, k + 2):  # union with neighbours covers motion between samples
            if 0 <= j < len(samples):
                out += samples[j][1]
        s = int(t)
        for j in (s - 1, s, s + 1):
            if 0 <= j < len(name_boxes):
                out += [[x - 6, y - 4, w + 12, h + 8] for x, y, w, h in name_boxes[j]]
        return out

    # render: decode, blur boxes, encode; original audio copied
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{a.start}", "-t", f"{a.end - a.start}", "-i", str(src),
                            "-vf", "fps=30", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", "30", "-i", "-", "-ss", f"{a.start}", "-t", f"{a.end - a.start}", "-i", str(src),
                            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "14", "-preset", "fast",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(a.out)],
                           stdin=subprocess.PIPE)
    n, fsz = 0, W * H * 3
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        for x, y, w, h in boxes_at(n / 30):
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, int(x + w)), min(H, int(y + h))
            if x1 - x0 > 4 and y1 - y0 > 4:
                roi = img[y0:y1, x0:x1]
                bw, bh = x1 - x0, y1 - y0
                small = cv2.resize(roi, (max(2, bw // 8), max(2, bh // 8)), interpolation=cv2.INTER_AREA)
                small = cv2.GaussianBlur(small, (0, 0), 1.6)
                img[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(small, (bw, bh), interpolation=cv2.INTER_LINEAR), (0, 0), 3)
        enc.stdin.write(img.tobytes())
        n += 1
    enc.stdin.close(); enc.wait(); dec.wait()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out} ({n} frames)")


if __name__ == "__main__":
    main()
