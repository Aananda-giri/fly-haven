"""Assemble the rendered frame sequence into the final captioned film.

    uv run python -m heaven.overlay runs/fly-heaven/final/edl.json runs/fly-heaven/final/frames runs/fly-heaven/fly-heaven.mp4

Adds a title card and burns in one caption per clip (from `heaven.director`'s
EDL, so each caption's timing is the real clip boundary, not retyped by hand).
"""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1920, 1080, 24
TITLE_SECONDS = 3.5
BG = (18, 26, 20)
INK = (223, 236, 208)
ACCENT = (196, 248, 106)  # experiments/fly-wirehead's own lime accent


def _font(size):
    for path in ["/usr/share/fonts/noto/NotoSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_title_card(out_path, width=WIDTH, height=HEIGHT):
    scale = height / HEIGHT
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title, subtitle, credit = "FLY HEAVEN", "a MaleCNS connectome decides how it lives", "eat · groom · bask · court · mate"
    for text, size, color, dy in [(title, 96, ACCENT, -70), (subtitle, 34, INK, 30), (credit, 26, INK, 90)]:
        font = _font(round(size * scale))
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, height / 2 + dy * scale), text, font=font, fill=color)
    img.save(out_path)


def make_srt(edl, out_path):
    lines = []
    t = 0.0
    for i, (clip, rng) in enumerate(zip(edl["clips"], edl["ranges"]), start=1):
        a, b = rng[0], rng[1]
        step = rng[2] if len(rng) > 2 else 1
        n_frames = len(range(a, b + 1, step))
        start, end = t, t + n_frames / FPS
        t = end

        def ts(s):
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            ms = round((s % 1) * 1000)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

        lines.append(f"{i}\n{ts(start)} --> {ts(end)}\n{clip['label']}\n")
    Path(out_path).write_text("\n".join(lines))


def run(*args):
    subprocess.run(list(args), check=True)


def assemble(edl_path, frames_dir, out_path, width=WIDTH, height=HEIGHT):
    edl = json.loads(Path(edl_path).read_text())
    if edl.get("mode") == "continuous":
        # No title insertion, retiming, or edit: frame zero through the final frame.
        expected = edl["total_output_frames"]
        missing = next((i for i in range(expected) if not (Path(frames_dir) / f"frame_{i:06d}.png").exists()), None)
        if missing is not None:
            raise ValueError(f"Incomplete render: missing frame {missing}")
        run("ffmpeg", "-y", "-framerate", str(edl["fps"]), "-i", str(Path(frames_dir) / "frame_%06d.png"),
            "-frames:v", str(expected), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out_path))
        print(f"wrote {out_path}")
        return
    work = Path(frames_dir).parent
    title_png, srt_path = work / "title.png", work / "captions.srt"
    make_title_card(title_png, width, height)
    make_srt(edl, srt_path)

    title_mp4, content_mp4 = work / "title.mp4", work / "content.mp4"
    run(
        "ffmpeg", "-y", "-loop", "1", "-i", str(title_png), "-t", str(TITLE_SECONDS),
        "-vf", f"scale={width}:{height},fps={FPS},format=yuv420p",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(title_mp4),
    )
    run(
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(Path(frames_dir) / "frame_%06d.png"),
        "-vf", f"subtitles={srt_path}:force_style='FontName=Noto Sans,FontSize=20,PrimaryColour=&H6AF8C4&,BorderStyle=3,Outline=1,Shadow=0'",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(content_mp4),
    )
    concat_txt = work / "concat.txt"
    concat_txt.write_text(f"file '{title_mp4.resolve()}'\nfile '{content_mp4.resolve()}'\n")
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("edl_path")
    p.add_argument("frames_dir")
    p.add_argument("out_path")
    args = p.parse_args()
    assemble(args.edl_path, args.frames_dir, args.out_path)
