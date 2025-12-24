#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge silent NeRF video with audio using ffmpeg, with configurable codecs.

Example:
    # 最常用、兼容性最好的 H.264 + AAC：
    python add_audio.py --video result-Obama.avi --audio aud-Zhao.wav --output TestResult-Obama_with_audio.mp4 --vcodec libx264 --crf 18
    python add_audio.py --video result-Obama1.avi --audio reply_1766419453546.wav --output ChatResult-Obama1_with_audio.mp4 --vcodec libx264 --crf 18
    
    # 如果你真的想保持原视频编码（不改码）：
    # python add_audio.py --vcodec copy
"""

import argparse
import subprocess
import sys


def merge_video_audio(video_path: str,
                      audio_path: str,
                      output_path: str,
                      vcodec: str = "libx264",
                      acodec: str = "aac",
                      crf: int = 18) -> None:
    """
    Use ffmpeg to mux video and audio into a single mp4 file.

    Parameters
    ----------
    video_path : str
        Path to input silent video.
    audio_path : str
        Path to input audio file.
    output_path : str
        Path to output video file.
    vcodec : str
        Video codec. "libx264" is recommended. Use "copy" to avoid re-encoding.
    acodec : str
        Audio codec, default "aac".
    crf : int
        Constant Rate Factor for libx264/others (lower = higher quality).
    """
    cmd = [
        "ffmpeg",
        "-y",                # overwrite output file without asking
        "-i", video_path,
        "-i", audio_path,
    ]

    if vcodec == "copy":
        # 保持原视频编码（可能就是你现在看到的 mp4v）
        cmd += ["-c:v", "copy"]
    else:
        # 重新编码为指定编码器（推荐 libx264）
        cmd += [
            "-c:v", vcodec,
            "-crf", str(crf),        # 18≈高质量，20~23 体积更小
            "-pix_fmt", "yuv420p",   # 提高播放器兼容性
        ]

    cmd += [
        "-c:a", acodec,
        "-shortest",
        output_path,
    ]

    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Attach audio to NeRF rendered video using ffmpeg."
    )
    parser.add_argument(
        "--video", type=str, default="result.avi",
        help="Path to input silent video (default: result.avi)"
    )
    parser.add_argument(
        "--audio", type=str, default="dataset/Zhao/aud.wav",
        help="Path to input audio file (default: dataset/Zhao/aud.wav)"
    )
    parser.add_argument(
        "--output", type=str, default="result_with_audio.mp4",
        help="Path to output video file (default: result_with_audio.mp4)"
    )
    parser.add_argument(
        "--vcodec", type=str, default="libx264",
        help='Video codec, e.g. "libx264", "libx265", or "copy" (default: libx264)'
    )
    parser.add_argument(
        "--acodec", type=str, default="aac",
        help='Audio codec, e.g. "aac" (default: aac)'
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="CRF for video quality when re-encoding (lower = better, default: 18)"
    )

    args = parser.parse_args()

    merge_video_audio(
        video_path=args.video,
        audio_path=args.audio,
        output_path=args.output,
        vcodec=args.vcodec,
        acodec=args.acodec,
        crf=args.crf,
    )


if __name__ == "__main__":
    main()
