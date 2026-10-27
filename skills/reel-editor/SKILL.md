---
name: reel-editor
description: >-
  Edit raw talking-head or camera footage into finished Instagram Reels / TikToks / YouTube Shorts
  (1080x1920, -14 LUFS). Cuts pauses, fillers and retakes off a word-timed transcript, reframes
  16:9 or sideways footage to 9:16 on the face, auto-grades Sony S-Log3 and iPhone HDR, punches in
  every few seconds, burns in word-synced captions (native / pop / box / mono / clean), adds an
  optional hook card, runs a broadcast-style vocal chain, ducks a music bed, and drops in B-roll
  full-screen or split-screen. Use WHENEVER the user hands over videos to "edit into reels", "cut
  this into short form", "make TikToks / Shorts from this", "add captions", "clip this", "fix the
  audio like radio", or drops a folder of footage (camera MP4s, iPhone .mov, CapCut exports).
  Not for faceless motion-graphics ads or launch films (use cinetic / motionmaxxing / motion-director).
---

# reel-editor

One engine, one JSON per reel, every edit reproducible.

```bash
R=~/.claude/skills/reel-editor/scripts/reel.py     # uv script, Python deps install themselves
$R probe  SRC                     # size, fps, rotation, S-Log/HDR, audio
$R transcribe SRC -o WORK          # word-timed transcript (whisper.cpp large-v3-turbo)
$R plan WORK [--drop 27-28,40]     # auto cut: pauses > .35s, fillers, + word indices you drop
$R render reel.json                # the whole edit
$R proof OUT.mp4                   # contact sheet + LUFS + spec check -> LOOK at it
```

Setup (once): `brew install ffmpeg whisper-cpp uv` and download the model:
`mkdir -p ~/.cache/whisper.cpp && curl -L -o ~/.cache/whisper.cpp/ggml-large-v3-turbo-q5_0.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin`

Work in `~/videos/reels/<batch>/`. Deliver finished files numbered one way only
(`01 hook-name.mp4`, `02 ...`), plus a `captions.md` with suggested post copy.

## Sensible defaults (ask the user before overriding)

- **Don't over-process good audio.** If the mic was decent, use `"audio": {"chain": "none"}` (one
  static gain to -14 LUFS). The radio chain + `denoise` can make a lav sound hollow. Use `radio` for
  rough in-camera audio, `light` for already-processed exports.
- **Phone footage keeps its look.** iPhone HDR: convert to SDR (see Footage notes), no LUT.
- **One recording per reel.** Only join clips when one continues the same thought in the same session.
- **Show the frame.** Start around `"zoom": 1.0`, `"punch_in": 1.12-1.2`, `"punch_every": 3`.
- **Native-looking captions** often perform best: `"captions": {"style": "native", "case": "natural", "y": 0.72}`.
  Heavy pop / mono / box styles can read as "AI generated"; offer them, don't default to them.
- **Short and whole.** 15 to 45 s, one idea. Keep explanations whole; cut only false starts,
  stumbles and repeated takes.
- **Prove sync, don't eyeball it.** Re-transcribe the finished reel and compare caption times with the
  voice (head under 0.15 s). Word timing comes from whisper DTW plus voice-onset snapping.
- Path gotcha: a folder name with colons (`10:15:26`) makes ffmpeg read it as a protocol. Symlink it.

## The editor's job (the part the script can't do)

The script cuts and renders. **You** decide what the reel is. Per source video:

1. **Read `WORK/transcript.txt`** (every word has an index and a source timestamp).
2. **Find the reel(s) inside it.** One idea per reel. A 6-minute ramble is usually 3 to 5 reels.
   Write each as a list of source time ranges in `edl`; ranges may be **reordered**, so the
   strongest line can be pulled to the front as the hook even if it was said at minute 4.
3. **Cut the intro.** No "hey guys", no name, no "so today". Start at the sentence that would
   normally arrive at second six.
4. **Kill retakes and stumbles** with `plan --drop` (word indices) or by editing the EDL by hand.
   Keep the *last* take of a repeated line; it is almost always the good one.
5. **Check the hook** (below) on the first spoken line.
6. **End on the payoff**, then one ask. If a line echoes the hook, end on it so the loop lands.

Never invent a claim, number, client or result in a hook card or caption. On-screen text must be
something the speaker actually said or a fact the user gave you.

## Hook check (every reel, before rendering)

Answer honestly, one line each:
1. Would a total stranger stop and watch this?
2. Does the hook clearly promise what the viewer gets by the end?
3. Is it broad enough that most people would relate?
4. Does it create curiosity or tension in the first second?
5. Can the viewer immediately tell this video is for them?
6. Is it simple enough to understand in one glance or one listen?

Any "no": pull a better line forward, or add an on-screen hook card that sharpens the spoken one
(more specific than the speech, never different in meaning).

## reel.json

```jsonc
{
  "source": "/path/C0106.MP4",
  "edl": [[12.4, 18.9], [3.1, 9.7], [40.2, 51.0]],   // or "WORK/edl.json" from plan
  "words": "WORK/words.json",          // source-time words; remapped through the edit for you
  "rotate": "auto",                    // 0 | 90 | -90 | 180 | "auto" (camera on its side)
  "color": "auto",                     // auto = S-Log3 LUT if the Sony XML says so; "apple" for iPhone HDR; or a .cube
  "grade": {"contrast": 1.04, "saturation": 1.06},   // optional eq on top
  "reframe": "face",                   // face | center | none
  "punch_in": 1.12,                    // alternate shots zoom in; long takes split at sentence ends
  "beat_every": 3.5,                   // max seconds before the frame changes
  "captions": {"style": "native", "y": "auto"},  // native | pop | box | mono | clean | off
  "hook": {"text": "5 HOURS -> 20 MIN", "start": 0, "end": 2.8, "style": "native_hook", "y": 0.17},
  "texts": [{"text": "STEP 2", "start": 9.1, "end": 11, "style": "box", "y": 0.18}],
  "inserts": [                         // output-timeline seconds
    {"file": "broll.mp4", "at": 4.2, "dur": 2.5, "mode": "top", "src_in": 0},   // split screen
    {"file": "shot.png",  "at": 12,  "dur": 2,   "mode": "full"}                // cutaway, slow push
  ],
  "audio": {"chain": "radio", "lufs": -14, "denoise": false},   // radio | light | none
  "music": {"file": "bed.wav", "gain_db": -20, "duck": true},
  "sfx":   [{"file": "whoosh.wav", "at": 4.2, "gain_db": -12}],
  "out": "out/01-hook.mp4"
}
```

## Styles

| look | config | when |
| --- | --- | --- |
| Plain white, natural case, thin outline | `captions.style: "native"` | default; looks hand-made |
| Word pop (yellow active word, heavy outline) | `"pop"` | high-energy talking head |
| Dark pill, sentence case | `"box"` | tech / explainer, calmer |
| Mono caps slab | `"mono"` | builder / AI content, split screens |
| Full sentence, quiet | `"clean"` | when a graphic carries the frame |
| Split screen: B-roll top, speaker bottom | insert `mode: "top"` | explaining a thing you can show |
| Full-screen screen recording / diagram | insert `mode: "full"` | proof moments, "look at this" |
| Animated graphic card (list, diagram, node map) | build in HyperFrames, insert it | lists, processes, comparisons |

Mix styles across a batch, not inside a reel: one caption style per reel.

**Safe zone** (the app UI covers the rest): keep text between y 0.12 and 0.75 and out of the right
230 px. `y: "auto"` already does this.

## Audio: the radio chain

`chain: "radio"` runs the six stages broadcast puts on a microphone: high-pass 80 Hz → downward
expander (room tone between words) → de-esser → EQ (-2.5 dB @250 mud, +2.5 dB @3.2k presence, air
shelf) → compressor 3.5:1 → two-pass loudnorm to -14 LUFS → true-peak limiter at -1 dBTP. Add
`"denoise": true` for fans, AC or wind.

Mix balance that works: voice around -16 LUFS, SFX around -24, music around -27 before the master.
Music `gain_db` -18 to -22 with `duck: true` sits the bed under the voice. Audition several tracks
in a numbered reel before picking one; use a different track per posted reel.

## Generating what the footage doesn't have

- **B-roll / stills:** any image or video generator the user has (Higgsfield, Runway, Kling, Veo...),
  9:16 for full inserts, 1:1 cropped to 1080x960 for `top` inserts. Check cost before generating.
- **Motion graphics** (lists, diagrams, stats): build a 1080x960 or 1080x1920 HyperFrames composition
  (`/hyperframes`), render it, insert it. Free and local.
- **Voice:** only clone a voice with that person's consent, from a clean 10 to 30 s clip.

## Extra tools in `scripts/`

- `anonymize.py`: blur one person's face (recognised, not just detected) and their on-screen name
  labels on a recorded video call.
- `voice_disguise.py`: natural-sounding pitch + formant shift on chosen time ranges of one speaker.

## Footage notes

- **Sony (`C0xxx.MP4` + `C0xxxM01.XML`):** often S-Log3 / S-Gamut3.Cine and sometimes sideways.
  `color: "auto"` reads the XML and applies `assets/luts/slog3-sgamut3cine-to-709.cube`;
  `rotate: "auto"` picks the rotation where faces are found.
- **iPhone HDR (HLG):** on macOS use `"color": "apple"` (AVFoundation HDR to SDR via `avconvert`), or
  `assets/luts/hlg-bt2100-to-709.cube`.
- **CapCut exports:** often already cut tight, so `plan` finds little; do the editorial pass by hand.

## Done means

For every reel: `proof` shows 1080x1920, 30 fps, -14 ±0.5 LUFS, true peak ≤ -1; you LOOKED at the
contact sheet (no caption on a face, no black frames, hook legible at frame 1); the hook passed the
checklist. **Never post.** The user posts.
