# Video Creator Skills for Claude Code

A ready-to-install set of [Claude Code](https://claude.com/claude-code) skills for video editing and
motion design. Hand Claude a link or a folder of footage and it can watch it, listen to it, cut it,
caption it, and build motion graphics around it.

## Install

```bash
git clone https://github.com/jdjf39/video-creator-skills.git
cd video-creator-skills && ./install.sh
```

Then restart Claude Code. You also want `ffmpeg`, `uv` and Node 22+ (macOS: `brew install ffmpeg uv node whisper-cpp`).

## What's inside

| Skill | What it does | Try saying |
| --- | --- | --- |
| **watch-video** | Downloads any video (YouTube, TikTok, IG, X, local file), transcribes every word with timestamps, grabs a still at every scene change, and builds a contact sheet so Claude can actually see and hear it. | "Watch this and break down why it works: <link>" |
| **reel-editor** | Turns raw talking-head footage into finished 9:16 Reels / TikToks / Shorts: cuts pauses and retakes from the transcript, reframes on the face, grades log/HDR, word-synced captions, broadcast vocal chain, music ducking, B-roll inserts. | "Cut this folder into 30-second reels with captions" |
| **cinetic** | Cinematic motion design from code: launch films, product videos, logo stings, kinetic type, UI walkthroughs, with a synced soundtrack and a self-critique loop. | "Make a 20 s launch video for my app" |
| **motionmaxxing** | Senior-motion-designer direction for promos, feature reveals, brand stings and social ads, so output doesn't read as AI slop. | "Make a brand sting for <name>" |
| **motion-director** | Storyboards, scores and animates promos, explainers, case studies and social reels from a URL, screenshots, a deck or a brief. | "Turn this website into a 30 s promo" |

`install.sh` also adds two open-source packs these build on:
- **HyperFrames** by HeyGen ([heygen-com/hyperframes](https://github.com/heygen-com/hyperframes)): HTML-to-video rendering engine plus skills for compositions, audio, captions and product launch videos.
- **GSAP skills** by GreenSock ([greensock/gsap-skills](https://github.com/greensock/gsap-skills)): the animation library most of the motion work uses.

## A good first session

1. `Watch this reference and tell me the hook, pacing, caption style and music: <link>`
2. `Here's my footage in ~/Movies/shoot. Make 3 reels in that style.`
3. `Add an animated title card at the start, built in HyperFrames.`

## Credits and licenses

- `cinetic`: MIT, © Leonxlnx (see `skills/cinetic/LICENSE.txt`)
- `motionmaxxing`: Apache 2.0 (see `skills/motionmaxxing/LICENSE` and `NOTICE`)
- `motion-director`: MIT, © Allen "AJ" Bactad ([systembuiltbyaj/motion-director](https://github.com/systembuiltbyaj/motion-director))
- `reel-editor`, `watch-video`: MIT (this repo). Bundled fonts are SIL OFL; face models are OpenCV Zoo (Apache 2.0).

Only download and edit videos you have the rights to use.
