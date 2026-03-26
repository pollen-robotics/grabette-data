"""LeRobot v3 dataset builder for GRABETTE.

Converts SLAM output + raw capture data into LeRobot v3 format
(Parquet + MP4). No Zarr intermediate.
"""

import json
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
FEATURES = {
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


def build_dataset(
    repo_id: str,
    episode_dirs: list[Path],
    task: str,
    fps: float = 46.0,
    image_size: tuple[int, int] = (720, 960),
    root: Path | None = None,
):
    """Build LeRobot v3 dataset from processed episode directories.

    Each episode directory must contain:
        - raw_video.mp4
        - imu_data.json (raw, with ANGL stream)
        - camera_trajectory.csv (or mapping_camera_trajectory.csv)
        - gravity.csv (optional, stored as episode metadata)

    Args:
        repo_id: dataset identifier (e.g. "steve/grabette-demo")
        episode_dirs: list of episode directory paths
        task: task description string
        fps: video frame rate
        image_size: (height, width) for output frames
        root: local storage path (default: HF cache)
    """
    # Lazy import — lerobot is a heavy dependency
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Override feature shapes to match requested image size
    features = FEATURES.copy()
    h, w = image_size
    features["observation.images.cam0"] = {
        **features["observation.images.cam0"],
        "shape": (3, h, w),
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
        # Last frame action = same as last joints (hold position)
        actions = np.zeros_like(joints)
        actions[:-1] = joints[1:]
        actions[-1] = joints[-1]

        # Read frame_skip from SLAM metadata (default 1 for old data)
        slam_meta_path = ep_dir / "slam_metadata.json"
        frame_skip = 1
        if slam_meta_path.is_file():
            with open(slam_meta_path) as f:
                frame_skip = json.load(f).get("frame_skip", 1)

        # Iterate video frames and add to dataset
        video_path = ep_dir / "raw_video.mp4"
        frame_iter = _iter_video_frames(video_path, image_size, frame_skip)

        for i in range(n_frames):
            try:
                img = next(frame_iter)
            except StopIteration:
                print(f"  Warning: video ended at frame {i}/{n_frames}")
                break

            # Convert BGR -> RGB for LeRobot
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            dataset.add_frame({
                "task": task,
                "observation.images.cam0": img_rgb,
                "observation.pose": poses[i],
                "observation.joints": joints[i],
                "action": actions[i],
            })

        dataset.save_episode()
        print(f"  Saved episode: {n_frames} frames")

    dataset.finalize()
    print(f"\nDataset complete: {repo_id}")
    if root:
        print(f"  Location: {root}")

    return dataset
