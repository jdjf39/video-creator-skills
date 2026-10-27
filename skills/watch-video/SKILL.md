---
name: watch-video
description: >-
  Download any video (YouTube, TikTok, Instagram Reels, X, Vimeo, or a local file) and actually
  WATCH and LISTEN to it: a word-timed transcript of everything said, stills at every scene change,
  and one contact sheet to look at. Use WHENEVER the user shares a video link or file and asks what
  it says, to summarize it, analyze it, break down its editing / pacing / hook / captions, recreate
  its style, pull quotes or timestamps, or use it as a reference for an edit or motion piece.
---

# watch-video

```bash
W=~/.claude/skills/watch-video/scripts/watch.py      # uv script, Python deps install themselves
$W "https://www.youtube.com/watch?v=..."              # or a TikTok / IG / X URL, or a local path
$W clip.mp4 -o ~/videos/refs/clip --model medium      # better transcript, slower
```

Output folder (default `./watch/<name>/`):
- `transcript.txt`: `[mm:ss.s] line` per segment. **Read this to hear the video.**
- `transcript.json`: same with per-word start/end times (for captions, cuts, quotes).
- `contact.jpg`: stills at scene changes. **Open it with Read to see the video.** Single stills are
  in `frames/`, named by timestamp; open the ones that matter at full size.
- `info.json`: duration, size, fps, title, uploader, description.

Flags: `--model tiny|base|small|medium|large-v3` (default small), `--lang en` to skip detection,
`--frames 40` / `--scene 0.2` for more stills (lower scene = more sensitive), `--no-audio`, `--no-frames`.

## How to use it well

1. Run it, then read `transcript.txt` AND look at `contact.jpg` before saying anything about the video.
2. Count cuts from the scene-change frames to judge pacing (cuts per 10 s), note caption style,
   text on screen, framing, colour, and where the hook lands in the first 3 seconds.
3. Quote with timestamps from the transcript; never paraphrase speech you did not transcribe.
4. For a style breakdown, give: hook (0-3 s), structure beats with times, edit rhythm, type and
   caption treatment, colour/grade, sound (music, SFX, VO), and the one thing that makes it work.

Notes: needs `ffmpeg` on PATH. Some sites need login cookies: add them through yt-dlp's
`--cookies-from-browser chrome` by running yt-dlp directly, then pass the downloaded file.
Only download videos the user has the right to use; references are for study, not reupload.
