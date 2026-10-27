#!/usr/bin/env bash
# Installs every skill in this repo into ~/.claude/skills, plus the open-source video skill packs
# they work with (HyperFrames by HeyGen, GSAP by GreenSock).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
mkdir -p "$dest"
for s in "$here"/skills/*/; do
  n="$(basename "$s")"
  rm -rf "$dest/$n" && cp -R "$s" "$dest/$n" && echo "installed $n"
done
chmod +x "$dest"/watch-video/scripts/*.py "$dest"/reel-editor/scripts/*.py 2>/dev/null || true

if command -v npx >/dev/null; then
  echo "adding HyperFrames + GSAP skill packs..."
  npx -y skills add heygen-com/hyperframes -g -y || echo "  (skipped: run 'npx skills add heygen-com/hyperframes' yourself)"
  npx -y skills add greensock/gsap-skills -g -y   || echo "  (skipped: run 'npx skills add greensock/gsap-skills' yourself)"
else
  echo "Node not found: install Node 22+, then run: npx skills add heygen-com/hyperframes && npx skills add greensock/gsap-skills"
fi

missing=""
for t in ffmpeg uv; do command -v $t >/dev/null || missing="$missing $t"; done
[ -n "$missing" ] && echo "still needed:$missing   (macOS: brew install ffmpeg uv whisper-cpp)"
echo "done. Restart Claude Code so it picks up the new skills."
