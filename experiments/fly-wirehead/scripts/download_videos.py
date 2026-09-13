"""Download portrait insect Shorts and prepare 9:16 browser-compatible MP4s.

Requires yt-dlp, ffmpeg, and ffprobe on PATH. Media stays outside Git.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    return next(s for s in probe["streams"] if s["codec_type"] == "video"), float(probe["format"]["duration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="Use videos already in the download cache")
    args = parser.parse_args()
    for tool in (["ffmpeg", "ffprobe"] if args.skip_download else ["yt-dlp", "ffmpeg", "ffprobe"]):
        if not shutil.which(tool):
            raise SystemExit(f"Missing {tool}. On macOS: brew install yt-dlp ffmpeg")
    cache, output = ROOT / "runs/video-downloads", ROOT / "dist/media"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "video-sources.json").read_text())
    ids = sources["ids"]
    trims = sources.get("trim_end_seconds", {})
    clips = []
    for video_id in ids:
        if not re.fullmatch(r"[\w-]{11}", video_id):
            raise ValueError("Invalid YouTube video ID")
        source = f"https://www.youtube.com/shorts/{video_id}"
        raw = cache / f"{video_id}.mp4"
        metadata = cache / f"{video_id}.info.json"
        if not args.skip_download and not (raw.exists() and metadata.exists()):
            subprocess.run(["yt-dlp", "--no-playlist", "--no-progress", "--write-info-json", "--merge-output-format", "mp4",
                            "--match-filters", "duration <= 90 & !is_live", "-f",
                            "bv*[height<=1280][vcodec^=avc1]+ba[ext=m4a]/b[height<=1280][ext=mp4]/bv*[height<=1280]+ba/b",
                            "-o", str(cache / "%(id)s.%(ext)s"), source], check=True)
        if not raw.exists() or not metadata.exists():
            raise SystemExit(f"Missing {video_id}; run again without --skip-download")
        data = json.loads(metadata.read_text())
        original, source_duration = inspect(raw)
        if original["width"] >= original["height"] or not 3 <= source_duration <= 91:
            raise ValueError(f"Expected a native portrait Short lasting 3–90 seconds: {video_id}")
        trim_end = trims.get(video_id)
        if trim_end is not None and (not isinstance(trim_end, (int, float)) or not math.isfinite(trim_end) or not 3 <= trim_end <= source_duration):
            raise ValueError(f"Invalid trim endpoint: {video_id}")
        expected_duration = trim_end if trim_end is not None else source_duration
        target = output / f"{video_id}.mp4"
        prepared, prepared_duration = inspect(target) if target.exists() else ({}, 0)
        if prepared.get("width") != 360 or prepared.get("height") != 640 or prepared.get("codec_name") != "h264" or abs(prepared_duration - expected_duration) > .1:
            temporary = output / f"{video_id}.partial.mp4"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                            *(["-t", str(trim_end)] if trim_end is not None else []),
                            "-map", "0:v:0", "-map", "0:a?", "-vf", "scale=360:640:force_original_aspect_ratio=increase,crop=360:640,setsar=1",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-threads", "2",
                            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(temporary)], check=True)
            temporary.replace(target)
        video, duration = inspect(target)
        if video["codec_name"] != "h264" or (video["width"], video["height"]) != (360, 640) or not 3 <= duration <= 91:
            raise ValueError(f"Invalid prepared video: {video_id}")
        if abs(duration - expected_duration) > .1:
            raise ValueError(f"Prepared duration does not match the requested trim: {video_id}")
        clips.append({"id": video_id, "file": target.name, "title": data["title"], "channel": data.get("channel") or data.get("uploader", ""),
                      "source": source, "duration": round(duration, 3), "width": video["width"], "height": video["height"],
                      "source_width": original["width"], "source_height": original["height"],
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        if trim_end is not None:
            clips[-1]["trim_end_seconds"] = trim_end
        print(f'{len(clips):02d} · {duration:.1f}s · {data["title"]}', flush=True)
    temporary = output / "playlist.json.partial"
    temporary.write_text(json.dumps({"version": 1, "clips": clips}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "playlist.json")
    print(f'{len(clips)} portrait Shorts ready, {len(clips) * 3} seconds per loop at 3 seconds per Short.', flush=True)


if __name__ == "__main__":
    main()
