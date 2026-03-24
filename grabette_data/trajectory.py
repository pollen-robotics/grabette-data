"""Trajectory CSV parsing, quaternion conversion, and joint angle interpolation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def get_grpc_timestamps_ms(ep_dir: Path) -> np.ndarray | None:
    """Parse grpc_camera_frames filenames to get absolute timestamps in ms.

    Filenames follow the pattern: frame_<timestamp_ms>_<frame_number>.jpg

    Returns:
        (N,) int64 array of absolute timestamps in ms, sorted by frame number.
        None if the directory doesn't exist or is empty.
    """
    frames_dir = ep_dir / "grpc_camera_frames"
    if not frames_dir.is_dir():
        return None
    files = sorted(frames_dir.glob("frame_*.jpg"))
    if not files:
        return None
    return np.array([int(f.stem.split('_')[1]) for f in files], dtype=np.int64)


# Axis remapping: current z -> -y, current y -> z, x unchanged.
# new_x = old_x,  new_y = -old_z,  new_z = old_y
_AXIS_REMAP = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=np.float64)


def _remap_pose(mat: np.ndarray) -> tuple[np.ndarray, Rotation]:
    """Apply _AXIS_REMAP to a 4x4 pose matrix.

    Returns:
        pos: (3,) remapped position
        rot: Rotation in remapped frame
    """
    pos = _AXIS_REMAP @ mat[:3, 3]
    rotvec = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    rotvec[1] *= -1
    rotvec[2] *= -1
    rot = Rotation.from_rotvec(rotvec)
    return pos, rot


def load_hand_trajectory(path: Path, grpc_start_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Load r_hand_traj.json and convert to relative timestamps + 6D poses.

    The 4×4 transformation matrix in each entry is decomposed into
    [x, y, z, ax, ay, az] (position + rotation vector) to match the
    observation.pose convention.

    Args:
        path: path to r_hand_traj.json
        grpc_start_ms: absolute timestamp (ms) of the first grpc frame,
            used to zero-base hand timestamps to the recording start.

    Returns:
        timestamps_s: (N,) float64 array in seconds (relative to grpc start)
        poses: (N, 6) float32 array [x, y, z, ax, ay, az]
    """
    with open(path) as f:
        data = json.load(f)
    timestamps_s = np.array(
        [(e['timestamp_ms'] - grpc_start_ms) / 1000.0 for e in data],
        dtype=np.float64,
    )
    poses = np.zeros((len(data), 6), dtype=np.float32)
    for i, entry in enumerate(data):
        mat = np.array(entry['pose'], dtype=np.float64).reshape(4, 4)
        pos, rot = _remap_pose(mat)
        poses[i, :3] = pos
        poses[i, 3:] = rot.as_rotvec()
    return timestamps_s, poses

def load_hand_trajectory_quat(path: Path, grpc_start_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Load r_hand_traj.json and convert to relative timestamps + 6D poses.

    The 4×4 transformation matrix in each entry is decomposed into
    [x, y, z, qx, qy, qz, qw] (position + quaternion) to match the
    observation.pose convention.

    Args:
        path: path to r_hand_traj.json
        grpc_start_ms: absolute timestamp (ms) of the first grpc frame,
            used to zero-base hand timestamps to the recording start.

    Returns:
        timestamps_s: (N,) float64 array in seconds (relative to grpc start)
        poses: (N, 6) float32 array [x, y, z, qx, qy, qz, qw]
    """
    with open(path) as f:
        data = json.load(f)
    timestamps_s = np.array(
        [(e['timestamp_ms'] - grpc_start_ms) / 1000.0 for e in data],
        dtype=np.float64,
    )
    poses = np.zeros((len(data), 7), dtype=np.float32)
    for i, entry in enumerate(data):
        mat = np.array(entry['pose'], dtype=np.float64).reshape(4, 4)
        pos, rot = _remap_pose(mat)
        poses[i, :3] = pos
        poses[i, 3:] = rot.as_quat()
    return timestamps_s, poses


def interpolate_hand_poses(
    hand_timestamps: np.ndarray,
    hand_poses: np.ndarray,
    video_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate 6D hand poses to video frame timestamps.

    Args:
        hand_timestamps: (N,) timestamps in seconds
        hand_poses: (N, 6) poses [x, y, z, ax, ay, az]
        video_timestamps: (M,) target timestamps in seconds

    Returns:
        (M, 6) float32 interpolated poses
    """
    result = np.zeros((len(video_timestamps), hand_poses.shape[1]), dtype=np.float32)
    for i in range(hand_poses.shape[1]):
        result[:, i] = np.interp(video_timestamps, hand_timestamps, hand_poses[:, i])
    return result


def load_trajectory_csv(path: Path) -> pd.DataFrame:
    """Load SLAM trajectory CSV.

    Columns: frame_idx, timestamp, state, is_lost, is_keyframe,
             x, y, z, q_x, q_y, q_z, q_w
    """
    return pd.read_csv(path)


def quaternion_to_axis_angle(qx: np.ndarray, qy: np.ndarray,
                             qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Convert quaternions to compact axis-angle (rotation vector).

    Args:
        qx, qy, qz, qw: arrays of shape (N,)

    Returns:
        (N, 3) rotation vectors (axis * angle in radians)
    """
    quats = np.stack([qx, qy, qz, qw], axis=-1)
    return Rotation.from_quat(quats, scalar_first=False).as_rotvec()


def trajectory_to_poses(df: pd.DataFrame) -> np.ndarray:
    """Convert trajectory DataFrame to (N, 6) pose array [x, y, z, ax, ay, az].

    Lost frames get all zeros.

    Args:
        df: trajectory DataFrame from load_trajectory_csv()

    Returns:
        (N, 6) float32 array: position + axis-angle
    """
    n = len(df)
    poses = np.zeros((n, 6), dtype=np.float32)

    tracked = ~df['is_lost'].astype(bool)
    if tracked.any():
        pos = df.loc[tracked, ['x', 'y', 'z']].values
        rotvec = quaternion_to_axis_angle(
            df.loc[tracked, 'q_x'].values,
            df.loc[tracked, 'q_y'].values,
            df.loc[tracked, 'q_z'].values,
            df.loc[tracked, 'q_w'].values,
        )
        poses[tracked, :3] = pos
        poses[tracked, 3:] = rotvec

    return poses


def interpolate_angles(imu_json_path: Path,
                       video_timestamps: np.ndarray) -> np.ndarray:
    """Interpolate ANGL stream (100Hz) to video frame timestamps.

    The raw ANGL stream stores value=[distal, proximal]. This function swaps
    them to return [proximal, distal] which matches the kinematic chain order.

    Args:
        imu_json_path: path to raw imu_data.json (not resampled — ANGL is stripped there)
        video_timestamps: (N,) array of video frame timestamps in seconds

    Returns:
        (N, 2) float32 array: [proximal, distal] in radians
    """
    with open(imu_json_path) as f:
        data = json.load(f)

    angl_samples = data['1']['streams']['ANGL']['samples']

    # ANGL timestamps are in ms, convert to seconds
    angl_cts = np.array([s['cts'] for s in angl_samples]) * 1e-3
    angl_vals = np.array([s['value'] for s in angl_samples])  # [distal, proximal]

    # Zero-base ANGL timestamps to match video timestamps (same as CORI in LoadTelemetry)
    # Video timestamps are frame_idx/fps, starting at 0
    # ANGL cts are absolute; we need to align them.
    # The ACCL stream's first timestamp is subtracted from IMU timestamps in LoadTelemetry.
    # ANGL timestamps use the same clock, so subtract the same offset.
    # We load ACCL to find the offset.
    accl_samples = data['1']['streams']['ACCL']['samples']
    imu_start_t = accl_samples[0]['cts'] * 1e-3
    angl_cts = angl_cts - imu_start_t

    n = len(video_timestamps)
    angles = np.zeros((n, 2), dtype=np.float32)

    # Interpolate each axis, then swap distal/proximal -> proximal/distal
    for i, axis in enumerate([1, 0]):  # proximal=index1, distal=index0
        angles[:, i] = np.interp(video_timestamps, angl_cts, angl_vals[:, axis])

    return angles


def load_gravity(path: Path) -> np.ndarray:
    """Load 3x3 gravity rotation matrix from CSV.

    Returns:
        (3, 3) float64 array
    """
    return np.loadtxt(path, delimiter=',')
