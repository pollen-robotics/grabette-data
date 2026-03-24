#!/usr/bin/env python3
"""Create SLAM map from a single mapping video (two-pass)."""

import click
from pathlib import Path
import numpy as np

from grabette_data.trajectory import (
    load_trajectory_csv,
    get_grpc_timestamps_ms,
    load_hand_trajectory_quat,
    interpolate_hand_poses,
)

@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Directory containing raw_video.mp4 and imu_data.json")
@click.option("--normalize", is_flag=True, default=False,
              help="Express Quest poses relative to the first pose (origin = frame 0)")
def main(input_dir, normalize):
    # Find trajectory file
    traj_path = Path(input_dir).absolute() / "camera_trajectory.csv"
    if not traj_path.is_file():
        traj_path = Path(input_dir).absolute() / "mapping_camera_trajectory.csv"
    if not traj_path.is_file():
        print(f"  Skipping: no trajectory CSV found")

    # Load trajectory and convert to 6D poses
    df = load_trajectory_csv(traj_path)
    timestamps = df['timestamp'].values  # seconds, relative to recording start


    # --- Quest cam and hand pose ---
    # grpc_camera_frames timestamps are absolute ms on the Quest clock.
    # We zero-base them to the first grpc frame to get relative timestamps,
    # then align with the trajectory (assuming simultaneous recording start).
    grpc_ts_ms = get_grpc_timestamps_ms(Path(input_dir).absolute())
    grpc_video_path = Path(input_dir).absolute() / "grpc_video.mp4"
    hand_traj_path = Path(input_dir).absolute() / "r_hand_traj.json"

    has_quest = (
        grpc_ts_ms is not None
        and grpc_video_path.is_file()
        and hand_traj_path.is_file()
    )

    if has_quest:
        grpc_start_ms = int(grpc_ts_ms[0])
        grpc_ts_s = (grpc_ts_ms - grpc_start_ms) / 1000.0  # relative seconds

        hand_ts_s, hand_poses = load_hand_trajectory_quat(hand_traj_path, grpc_start_ms, normalize=normalize)

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

    if has_quest:
        quest_poses = interpolate_hand_poses(hand_ts_s, hand_poses, timestamps)
    else:
        quest_poses = np.zeros((n_frames, 7), dtype=np.float32)

    print(quest_poses)
    traj = np.column_stack((timestamps, quest_poses))
    frames_ids = np.arange(len(traj), dtype=int)
    poses = np.column_stack((frames_ids, traj))
    formats = ["%d"] + ["%.8f"] * (poses.shape[1] - 1)
    np.savetxt(Path(input_dir).absolute() / "quest_traj.csv", poses, delimiter=",", fmt=formats, header="frame_idx,timestamp,x,y,z,q_x,q_y,q_z,q_w", comments="")


if __name__ == "__main__":
    main()
