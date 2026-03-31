"""LeRobot v3 dataset builder for GRABETTE.

Converts trajectory + capture data into LeRobot v3 format (Parquet + MP4).
Supports two camera sources:
  - RPi fisheye camera (raw_video.mp4)
  - Quest POV camera (grpc_camera_frames/*.jpg), optional
"""

import json
import os
from pathlib import Path

import av
import cv2
import numpy as np

from grabette_data.trajectory import (
    load_trajectory_csv,
    trajectory_to_poses,
    interpolate_angles,
    load_gravity,
)

# Feature schema for the GRABETTE dataset
FEATURES_BASE = {
    "observation.images.cam0": {
        "dtype": "video",
        "shape": (3, 720, 960),  # C, H, W — LeRobot convention
        "names": ["channels", "height", "width"],
    },
    "observation.pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "ax", "ay", "az"],
    },
    "observation.joints": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["proximal", "distal"],
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["proximal", "distal"],
    },
}


def _iter_video_frames(video_path: Path, size: tuple[int, int],
                       frame_skip: int = 1):
    """Yield (H, W, 3) uint8 BGR frames resized to (h, w) = size.

    When frame_skip > 1, yields every Nth frame to match decimated trajectories.
    """
    h, w = size
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if frame_skip > 1 and i % frame_skip != 0:
                continue
            img = frame.to_ndarray(format="bgr24")
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            yield img


def _load_quest_frames(episode_dir: Path) -> list[tuple[float, Path]] | None:
    """Load Quest camera frame paths with timestamps.

    Returns list of (timestamp_ms, path) sorted by timestamp, or None if
    no Quest frames found.
    """
    quest_dir = episode_dir / "grpc_camera_frames"
    if not quest_dir.is_dir():
        return None

    frames = []
    for f in sorted(quest_dir.iterdir()):
        if not f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            continue
        # Filename format: frame_{timestamp}_{seqnum}.ext
        parts = f.stem.split('_')
        if len(parts) >= 2:
            ts_ms = int(parts[1])
            frames.append((ts_ms, f))

    return frames if frames else None


def _get_nearest_frame(quest_frames: list[tuple[float, Path]],
                       target_ts_ms: float) -> Path | None:
    """Find the Quest frame nearest to target timestamp (in ms)."""
    # Binary search
    timestamps = [f[0] for f in quest_frames]
    idx = np.searchsorted(timestamps, target_ts_ms)
    # Check neighbors
    best_idx = idx
    if idx > 0 and idx < len(timestamps):
        if abs(timestamps[idx - 1] - target_ts_ms) < abs(timestamps[idx] - target_ts_ms):
            best_idx = idx - 1
    elif idx >= len(timestamps):
        best_idx = len(timestamps) - 1
    return quest_frames[best_idx][1]


def build_dataset(
    repo_id: str,
    episode_dirs: list[Path],
    task: str,
    fps: float = 46.0,
    image_size: tuple[int, int] = (720, 960),
    quest_image_size: tuple[int, int] | None = None,
    root: Path | None = None,
    include_quest_camera: bool = False,
):
    """Build LeRobot v3 dataset from processed episode directories.

    Each episode directory must contain:
        - raw_video.mp4
        - imu_data.json (raw, with ANGL stream)
        - camera_trajectory.csv (or mapping_camera_trajectory.csv)

    Optionally:
        - grpc_camera_frames/ (Quest POV images, added as observation.images.cam1)

    Args:
        repo_id: dataset identifier (e.g. "steve/grabette-demo")
        episode_dirs: list of episode directory paths
        task: task description string
        fps: dataset frame rate
        image_size: (height, width) for RPi camera output frames
        quest_image_size: (height, width) for Quest camera output frames
            (default: same as image_size)
        root: local storage path (default: HF cache)
        include_quest_camera: include Quest POV camera as observation.images.cam1
    """
    # Lazy import — lerobot is a heavy dependency
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if quest_image_size is None:
        quest_image_size = image_size

    # Build feature schema
    features = FEATURES_BASE.copy()
    h, w = image_size
    features["observation.images.cam0"] = {
        **features["observation.images.cam0"],
        "shape": (3, h, w),
    }
    if include_quest_camera:
        qh, qw = quest_image_size
        features["observation.images.cam1"] = {
            "dtype": "video",
            "shape": (3, qh, qw),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        root=root,
        robot_type="grabette",
        use_videos=True,
        vcodec="h264",
    )

    for ep_dir in episode_dirs:
        ep_dir = Path(ep_dir).absolute()
        print(f"\nProcessing {ep_dir.name}...")

        # Find trajectory file
        traj_path = ep_dir / "camera_trajectory.csv"
        if not traj_path.is_file():
            traj_path = ep_dir / "mapping_camera_trajectory.csv"
        if not traj_path.is_file():
            print(f"  Skipping: no trajectory CSV found")
            continue

        # Load trajectory and convert to 6D poses
        df = load_trajectory_csv(traj_path)
        poses = trajectory_to_poses(df)
        timestamps = df['timestamp'].values
        n_frames = len(df)

        # Load joint angles from raw IMU (ANGL stream)
        imu_path = ep_dir / "imu_data.json"
        if imu_path.is_file():
            joints = interpolate_angles(imu_path, timestamps)
        else:
            print(f"  Warning: no imu_data.json, joints will be zeros")
            joints = np.zeros((n_frames, 2), dtype=np.float32)

        # Compute actions: action[t] = joints[t+1] (next-step angle)
        actions = np.zeros_like(joints)
        actions[:-1] = joints[1:]
        actions[-1] = joints[-1]

        # Read metadata to determine trajectory source
        slam_meta_path = ep_dir / "slam_metadata.json"
        frame_skip = 1
        method = "slam"
        if slam_meta_path.is_file():
            with open(slam_meta_path) as f:
                meta = json.load(f)
            frame_skip = meta.get("frame_skip", 1)
            method = meta.get("method", "slam")

        # Load Quest camera frames if requested
        quest_frames = None
        quest_traj_ts_ms = None
        if include_quest_camera:
            quest_frames = _load_quest_frames(ep_dir)
            if quest_frames:
                # For Quest-based trajectories, the trajectory timestamps align
                # with Quest frame timestamps. Load Quest trajectory for timestamp mapping.
                quest_traj_path = ep_dir / "r_hand_traj.json"
                if quest_traj_path.is_file() and method == "quest":
                    with open(quest_traj_path) as f:
                        quest_data = json.load(f)
                    quest_traj_ts_ms = [e["timestamp_ms"] for e in quest_data]
                print(f"  Quest camera: {len(quest_frames)} frames")
            else:
                print(f"  Warning: no Quest camera frames found")

        # Iterate video frames
        video_path = ep_dir / "raw_video.mp4"
        if method == "quest":
            # Quest-based trajectory: video frames need to be matched by timestamp
            # Open video and get frame timestamps
            cap = cv2.VideoCapture(str(video_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # RPi video timestamps (zero-based)
            rpi_frame_ts = np.arange(total_video_frames) / video_fps
            frame_iter = _iter_video_frames(video_path, image_size, frame_skip=1)
            all_rpi_frames = list(frame_iter)
        else:
            # SLAM-based trajectory: frame_skip handles alignment
            frame_iter = _iter_video_frames(video_path, image_size, frame_skip)

        qh, qw = quest_image_size

        for i in range(n_frames):
            if method == "quest":
                # Find nearest RPi frame for this trajectory timestamp
                traj_t = timestamps[i]
                nearest_idx = int(np.argmin(np.abs(rpi_frame_ts - traj_t)))
                if nearest_idx < len(all_rpi_frames):
                    img = all_rpi_frames[nearest_idx]
                else:
                    print(f"  Warning: no RPi frame for t={traj_t:.3f}s")
                    break
            else:
                try:
                    img = next(frame_iter)
                except StopIteration:
                    print(f"  Warning: video ended at frame {i}/{n_frames}")
                    break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frame_data = {
                "task": task,
                "observation.images.cam0": img_rgb,
                "observation.pose": poses[i],
                "observation.joints": joints[i],
                "action": actions[i],
            }

            # Add Quest camera image if available
            if include_quest_camera and quest_frames:
                quest_img = None
                if quest_traj_ts_ms and i < len(quest_traj_ts_ms):
                    # Direct timestamp match for Quest-based trajectories
                    quest_path = _get_nearest_frame(quest_frames, quest_traj_ts_ms[i])
                else:
                    # Fallback: match by trajectory timestamp (needs absolute Quest timestamps)
                    quest_path = quest_frames[min(i, len(quest_frames) - 1)][1]

                if quest_path:
                    quest_img_bgr = cv2.imread(str(quest_path))
                    if quest_img_bgr is not None:
                        if quest_img_bgr.shape[0] != qh or quest_img_bgr.shape[1] != qw:
                            quest_img_bgr = cv2.resize(quest_img_bgr, (qw, qh))
                        quest_img = cv2.cvtColor(quest_img_bgr, cv2.COLOR_BGR2RGB)

                if quest_img is not None:
                    frame_data["observation.images.cam1"] = quest_img
                else:
                    # Black frame if quest image missing
                    frame_data["observation.images.cam1"] = np.zeros((qh, qw, 3), dtype=np.uint8)

            dataset.add_frame(frame_data)

        dataset.save_episode()
        print(f"  Saved episode: {n_frames} frames")

    dataset.finalize()
    print(f"\nDataset complete: {repo_id}")
    if root:
        print(f"  Location: {root}")

    return dataset
