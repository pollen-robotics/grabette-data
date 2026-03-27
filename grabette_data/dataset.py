"""LeRobot v3 dataset builder for GRABETTE.

Converts SLAM output + raw capture data into LeRobot v3 format
(Parquet + MP4). No Zarr intermediate.
"""

from pathlib import Path

import av
import cv2
import numpy as np

from grabette_data.video import mux_grpc_video
from grabette_data.trajectory import (
    load_trajectory_csv,
    trajectory_to_poses,
    interpolate_angles,
    load_gravity,
    get_grpc_timestamps_ms,
    load_hand_trajectory,
    interpolate_hand_poses,
)

# Feature schema for the GRABETTE dataset
FEATURES = {
    "observation.images.cam0": {
        "dtype": "video",
        "shape": (3, 720, 960),  # C, H, W — LeRobot convention
        "names": ["channels", "height", "width"],
    },
    "observation.images.quest_cam": {
        "dtype": "video",
        "shape": (3, 720, 960),  # C, H, W — LeRobot convention
        "names": ["channels", "height", "width"],
    },
    "observation.pose": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["x", "y", "z", "ax", "ay", "az"],
    },
    "observation.pose_quest": {
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


def _iter_video_frames(video_path: Path, size: tuple[int, int]):
    """Yield (H, W, 3) uint8 BGR frames resized to (h, w) = size."""
    h, w = size
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="bgr24")
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            yield img


class _GrpcFrameReader:
    """Stream grpc_video.mp4 on demand by timestamp (non-decreasing access)."""

    def __init__(self, video_path: Path, size: tuple[int, int]):
        self.h, self.w = size
        self._container = av.open(str(video_path))
        stream = self._container.streams.video[0]
        self._fps = float(stream.average_rate or stream.guessed_rate or 30)
        self._iter = self._container.decode(stream)
        self._current_idx = -1
        self._current_img: np.ndarray | None = None

    def get(self, target_ts: float) -> np.ndarray:
        """Return the video frame closest to target_ts (seconds from video start).

        target_ts must be non-decreasing across successive calls.
        """
        target_idx = int(round(target_ts * self._fps))
        while self._current_idx < target_idx:
            try:
                frame = next(self._iter)
                self._current_idx += 1
                img = frame.to_ndarray(format="bgr24")
                if img.shape[0] != self.h or img.shape[1] != self.w:
                    img = cv2.resize(img, (self.w, self.h))
                self._current_img = img
            except StopIteration:
                break
        if self._current_img is None:
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
        return self._current_img

    def close(self):
        self._container.close()


def build_dataset(
    repo_id: str,
    episode_dirs: list[Path],
    task: str,
    fps: float = 46.0,
    image_size: tuple[int, int] = (720, 960),
    root: Path | None = None,
    normalize_quest: bool = False,
):
    """Build LeRobot v3 dataset from processed episode directories.

    Each episode directory must contain:
        - raw_video.mp4
        - imu_data.json (raw, with ANGL stream)
        - camera_trajectory.csv (or mapping_camera_trajectory.csv)
        - grpc_video.mp4 + grpc_camera_frames/ + r_hand_traj.json (optional)

    When the quest sources are present, observation.images.quest_cam is filled
    from grpc_video.mp4 and observation.pose_quest from r_hand_traj.json,
    both synchronised to the camera trajectory timeline.  If any source is
    shorter than the others, the episode is truncated to the shortest duration.

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
    features["observation.images.quest_cam"] = {
        **features["observation.images.quest_cam"],
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
        timestamps = df['timestamp'].values  # seconds, relative to recording start
        poses = trajectory_to_poses(df)

        # Load joint angles from raw IMU (ANGL stream)
        imu_path = ep_dir / "imu_data.json"
        if imu_path.is_file():
            joints = interpolate_angles(imu_path, timestamps)
        else:
            print(f"  Warning: no imu_data.json, joints will be zeros")
            joints = np.zeros((len(timestamps), 2), dtype=np.float32)

        # --- Quest cam and hand pose ---
        # grpc_camera_frames timestamps are absolute ms on the Quest clock.
        # We zero-base them to the first grpc frame to get relative timestamps,
        # then align with the trajectory (assuming simultaneous recording start).

        # Generate grpc_video.mp4 from raw JPEG frames if not already present.
        grpc_video_path = ep_dir / "grpc_video.mp4"
        if not grpc_video_path.is_file() and (ep_dir / "grpc_camera_frames").is_dir():
            print(f"  Muxing gRPC frames to video...")
            mux_grpc_video(ep_dir)

        grpc_ts_ms = get_grpc_timestamps_ms(ep_dir)
        hand_traj_path = ep_dir / "r_hand_traj.json"

        has_quest = (
            grpc_ts_ms is not None
            and grpc_video_path.is_file()
            and hand_traj_path.is_file()
        )

        if has_quest:
            grpc_start_ms = int(grpc_ts_ms[0])
            grpc_ts_s = (grpc_ts_ms - grpc_start_ms) / 1000.0  # relative seconds

            hand_ts_s, hand_poses = load_hand_trajectory(hand_traj_path, grpc_start_ms, normalize=normalize_quest)

            # Clip trajectory to the range covered by all sources
            t_max = min(timestamps[-1], grpc_ts_s[-1], hand_ts_s[-1])
            n_frames = int(np.searchsorted(timestamps, t_max, side='right'))
            if n_frames < len(timestamps):
                print(f"  Clipping to {n_frames}/{len(timestamps)} frames "
                      f"(shortest source ends at {t_max:.2f}s)")
        else:
            missing = []
            if grpc_ts_ms is None:
                missing.append("grpc_camera_frames")
            if not grpc_video_path.is_file():
                missing.append("grpc_video.mp4")
            if not hand_traj_path.is_file():
                missing.append("r_hand_traj.json")
            print(f"  Warning: quest data missing ({', '.join(missing)}), "
                  f"quest_cam and pose_quest will be zeros")
            n_frames = len(timestamps)

        # Slice to n_frames
        timestamps = timestamps[:n_frames]
        poses = poses[:n_frames]
        joints = joints[:n_frames]

        if has_quest:
            quest_poses = interpolate_hand_poses(hand_ts_s, hand_poses, timestamps)
            # Map each trajectory timestamp to the nearest Quest-relative timestamp.
            # grpc_ts_s is relative to the first JPEG frame; the video shares the same origin.
            grpc_indices = np.searchsorted(grpc_ts_s, timestamps, side='left')
            grpc_indices = np.clip(grpc_indices, 0, len(grpc_ts_s) - 1)
            # Resolve ties: pick the index with the closer timestamp
            mask = grpc_indices > 0
            prev_dist = np.abs(timestamps[mask] - grpc_ts_s[grpc_indices[mask] - 1])
            curr_dist = np.abs(timestamps[mask] - grpc_ts_s[grpc_indices[mask]])
            grpc_indices[mask] -= (prev_dist < curr_dist).astype(int)
            # Timestamps (seconds) to seek in the video — avoids assuming 1:1 JPEG/frame mapping
            grpc_target_ts = grpc_ts_s[grpc_indices]
        else:
            quest_poses = np.zeros((n_frames, 6), dtype=np.float32)

        # Compute actions: action[t] = joints[t+1] (next-step angle)
        # Last frame action = same as last joints (hold position)
        actions = np.zeros_like(joints)
        actions[:-1] = joints[1:]
        actions[-1] = joints[-1]

        # Iterate video frames and add to dataset
        raw_video_path = ep_dir / "raw_video.mp4"
        raw_iter = _iter_video_frames(raw_video_path, image_size)
        grpc_reader = _GrpcFrameReader(grpc_video_path, image_size) if has_quest else None

        for i in range(n_frames):
            try:
                img = next(raw_iter)
            except StopIteration:
                print(f"  Warning: raw_video ended at frame {i}/{n_frames}")
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if grpc_reader is not None:
                quest_img = grpc_reader.get(float(grpc_target_ts[i]))
                quest_img_rgb = cv2.cvtColor(quest_img, cv2.COLOR_BGR2RGB)
            else:
                quest_img_rgb = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

            dataset.add_frame({
                "task": task,
                "observation.images.cam0": img_rgb,
                "observation.images.quest_cam": quest_img_rgb,
                "observation.pose": poses[i],
                "observation.pose_quest": quest_poses[i],
                "observation.joints": joints[i],
                "action": actions[i],
            })

        if grpc_reader is not None:
            grpc_reader.close()

        dataset.save_episode()
        print(f"  Saved episode: {n_frames} frames")

    dataset.finalize()
    print(f"\nDataset complete: {repo_id}")
    if root:
        print(f"  Location: {root}")

    return dataset
