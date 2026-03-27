"""Video utilities for GRABETTE dataset generation."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def mux_grpc_video(episode_dir: Path) -> None:
    """Assemble gRPC JPEG frames from an episode directory into grpc_video.mp4."""
    camera_dir = episode_dir / "grpc_camera_frames"
    if not camera_dir.is_dir():
        return

    frames = sorted(camera_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        logger.info("gRPC: not enough frames to create MP4 (%d)", len(frames))
        return

    # Parse timestamps from filenames: frame_{timestamp_ms:016d}_{n:06d}.jpg
    try:
        ts_list = [int(f.stem.split("_")[1]) for f in frames]
    except (IndexError, ValueError):
        logger.warning("gRPC: could not parse timestamps from frame filenames")
        return

    duration_ms = ts_list[-1] - ts_list[0]
    fps = (len(ts_list) - 1) / (duration_ms / 1000.0) if duration_ms > 0 else 30.0

    concat_path = camera_dir / "frames.txt"
    lines = []
    for i, f in enumerate(frames):
        lines.append(f"file '{f.name}'")
        duration_s = (ts_list[i + 1] - ts_list[i]) / 1000.0 if i + 1 < len(ts_list) else 1.0 / fps
        lines.append(f"duration {duration_s:.6f}")
    concat_path.write_text("\n".join(lines))

    output_path = episode_dir / "grpc_video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-vf", "format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        concat_path.unlink(missing_ok=True)
        logger.warning("gRPC: ffmpeg not found — install ffmpeg to mux Quest frames into video")
        return
    concat_path.unlink(missing_ok=True)
    if result.returncode != 0:
        logger.warning("gRPC: ffmpeg failed: %s", result.stderr[-300:])
    else:
        logger.info("gRPC: video saved → %s (%.1f fps, %d frames)", output_path, fps, len(frames))
