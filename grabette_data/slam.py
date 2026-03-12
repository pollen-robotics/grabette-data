"""Docker-based SLAM orchestration.

Runs grabette_slam inside Docker for mapping (pass 1 + pass 2) and
batch localization against an existing map.
"""

import concurrent.futures
import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from grabette_data.imu import prepare_imu_for_slam
from grabette_data.mask import generate_mask

# Default Docker image name (local build, no hub pull)
DEFAULT_DOCKER_IMAGE = "pollenrobotics/orbslam3-headless"

# Default SLAM settings — shipped with this package
_PACKAGE_DIR = Path(__file__).parent
DEFAULT_SETTINGS = _PACKAGE_DIR.parent / "config" / "rpi_bmi088_slam_settings.yaml"


@dataclass
class SlamResult:
    """Result from a single SLAM run."""
    returncode: int
    total_frames: int
    tracked_frames: int
    trajectory_path: Path | None

    @property
    def tracking_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return 100.0 * self.tracked_frames / self.total_frames


def _ensure_imu_resampled(video_dir: Path) -> str:
    """Resample IMU if not already done. Returns IMU filename to use."""
    resampled = video_dir / "imu_data_resampled.json"
    if not resampled.is_file():
        raw = video_dir / "imu_data.json"
        if not raw.is_file():
            raise FileNotFoundError(f"No imu_data.json in {video_dir}")
        print(f"Resampling IMU to uniform 200Hz...")
        prepare_imu_for_slam(raw, resampled)
    return "imu_data_resampled.json"


def _ensure_mask(video_dir: Path, video_path: Path) -> str:
    """Generate SLAM mask if not already present. Returns mask filename."""
    mask_path = video_dir / "slam_mask.png"
    if not mask_path.is_file():
        # Get video resolution for mask
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            w, h = stream.width, stream.height
        mask = generate_mask(w, h)
        cv2.imwrite(str(mask_path), mask)
    return "slam_mask.png"


def _parse_tracking_rate(trajectory_path: Path) -> tuple[int, int]:
    """Parse trajectory CSV and return (total_frames, tracked_frames)."""
    if not trajectory_path.is_file():
        return 0, 0
    df = pd.read_csv(trajectory_path)
    total = len(df)
    tracked = total - int(df['is_lost'].sum())
    return total, tracked


def _build_docker_cmd(
    video_dir: Path,
    *,
    imu_filename: str,
    output_csv: str,
    settings_path: Path,
    save_map: Path | None = None,
    load_map: Path | None = None,
    mask: bool = True,
    output_gravity: str | None = None,
    output_biases: str | None = None,
    max_lost_frames: int = -1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
) -> list[str]:
    """Build the docker run command list."""
    # Docker mount points
    data_mount = "/data"
    settings_mount = "/settings"

    cmd = [
        "docker", "run", "--rm",
        "--volume", f"{video_dir}:{data_mount}",
    ]

    # Mount map directory if saving or loading
    if save_map is not None:
        cmd.extend(["--volume", f"{save_map.parent}:/map"])
    if load_map is not None:
        cmd.extend(["--volume", f"{load_map.parent}:/map"])

    # Mount settings
    cmd.extend(["--volume", f"{settings_path.parent}:{settings_mount}"])

    # Image + binary
    cmd.extend([
        docker_image,
        "/ORB_SLAM3/Examples/Monocular-Inertial/grabette_slam",
        "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting", f"{settings_mount}/{settings_path.name}",
        "--input_video", f"{data_mount}/raw_video.mp4",
        "--input_imu_json", f"{data_mount}/{imu_filename}",
        "--output_trajectory_csv", f"{data_mount}/{output_csv}",
    ])

    if save_map is not None:
        cmd.extend(["--save_map", f"/map/{save_map.name}"])
    if load_map is not None:
        cmd.extend(["--load_map", f"/map/{load_map.name}"])
    if mask:
        cmd.extend(["--mask_img", f"{data_mount}/slam_mask.png"])
    if output_gravity:
        cmd.extend(["--output_gravity", f"{data_mount}/{output_gravity}"])
    if output_biases:
        cmd.extend(["--output_biases", f"{data_mount}/{output_biases}"])
    if max_lost_frames > 0:
        cmd.extend(["--max_lost_frames", str(max_lost_frames)])

    return cmd


def run_slam(
    video_dir: Path,
    *,
    output_csv: str = "camera_trajectory.csv",
    save_map: Path | None = None,
    load_map: Path | None = None,
    output_gravity: bool = False,
    output_biases: bool = False,
    mask: bool = True,
    max_lost_frames: int = -1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
    timeout_s: float | None = None,
) -> SlamResult:
    """Run grabette_slam in Docker on a single video directory.

    Args:
        video_dir: directory containing raw_video.mp4 and imu_data.json
        output_csv: trajectory output filename (inside video_dir)
        save_map: if set, save map atlas to this path
        load_map: if set, load map atlas from this path
        output_gravity: write gravity.csv
        output_biases: write biases.csv
        mask: generate and apply device mask
        max_lost_frames: terminate after N lost frames (-1 = disabled)
        docker_image: Docker image name
        settings_path: SLAM settings YAML path
        timeout_s: subprocess timeout in seconds

    Returns:
        SlamResult with tracking statistics
    """
    video_dir = Path(video_dir).absolute()
    settings_path = Path(settings_path).absolute()

    # Ensure prerequisites
    imu_filename = _ensure_imu_resampled(video_dir)
    if mask:
        _ensure_mask(video_dir, video_dir / "raw_video.mp4")

    cmd = _build_docker_cmd(
        video_dir,
        imu_filename=imu_filename,
        output_csv=output_csv,
        settings_path=settings_path,
        save_map=save_map,
        load_map=load_map,
        mask=mask,
        output_gravity="gravity.csv" if output_gravity else None,
        output_biases="biases.csv" if output_biases else None,
        max_lost_frames=max_lost_frames,
        docker_image=docker_image,
    )

    stdout_path = video_dir / "slam_stdout.txt"
    stderr_path = video_dir / "slam_stderr.txt"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(video_dir),
            stdout=stdout_path.open("w"),
            stderr=stderr_path.open("w"),
            timeout=timeout_s,
        )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        print(f"  SLAM timed out after {timeout_s:.0f}s")
        returncode = -1

    traj_path = video_dir / output_csv
    total, tracked = _parse_tracking_rate(traj_path)

    return SlamResult(
        returncode=returncode,
        total_frames=total,
        tracked_frames=tracked,
        trajectory_path=traj_path if traj_path.is_file() else None,
    )


def _copy_file(src: Path, dst: Path):
    """Copy file handling root-owned Docker output files."""
    if src.is_file():
        data = src.read_bytes()
        if dst.is_file():
            os.remove(str(dst))
        dst.write_bytes(data)


def create_map(
    video_dir: Path,
    *,
    retries: int = 3,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
) -> Path:
    """Two-pass mapping: pass 1 (save_map + gravity/biases), pass 2 (load_map).

    Retries pass 1 up to N times, keeps best by tracking rate.
    Pass 2 re-localizes against the map to recover initialization frames.

    Args:
        video_dir: directory containing raw_video.mp4 and imu_data.json
        retries: max retry attempts for pass 1
        docker_image: Docker image name
        settings_path: SLAM settings YAML path

    Returns:
        Path to map_atlas.osa
    """
    video_dir = Path(video_dir).absolute()
    map_dir = video_dir / "map"
    map_dir.mkdir(exist_ok=True)
    map_path = map_dir / "map_atlas.osa"

    total_attempts = 1 + retries
    best_pct = -1.0
    best_attempt = 0

    # --- Pass 1: Mapping ---
    for attempt in range(1, total_attempts + 1):
        if total_attempts > 1:
            print(f"\n--- Pass 1, attempt {attempt}/{total_attempts} ---")
        else:
            print("Running SLAM mapping (pass 1)...")

        result = run_slam(
            video_dir,
            output_csv="mapping_camera_trajectory.csv",
            save_map=map_path,
            output_gravity=True,
            output_biases=True,
            docker_image=docker_image,
            settings_path=settings_path,
        )

        if result.returncode != 0 or result.trajectory_path is None:
            print(f"  SLAM failed (return code {result.returncode})")
            continue

        pct = result.tracking_pct
        print(f"  Tracking: {result.tracked_frames}/{result.total_frames} ({pct:.1f}%)")

        if pct > best_pct:
            best_pct = pct
            best_attempt = attempt
            # Save best results for restoration
            if total_attempts > 1:
                for src_name, dst_name in [
                    ("mapping_camera_trajectory.csv", "mapping_camera_trajectory_best.csv"),
                    ("slam_stdout.txt", "slam_stdout_best.txt"),
                    ("gravity.csv", "gravity_best.csv"),
                    ("biases.csv", "biases_best.csv"),
                ]:
                    _copy_file(video_dir / src_name, video_dir / dst_name)
                _copy_file(map_path, map_dir / "map_atlas_best.osa")

        if pct >= 90:
            if total_attempts > 1:
                print(f"  >= 90% tracking, stopping early")
            break

    # Restore best result if last attempt wasn't the best
    if total_attempts > 1 and best_pct >= 0 and best_attempt != attempt:
        _copy_file(video_dir / "mapping_camera_trajectory_best.csv",
                   video_dir / "mapping_camera_trajectory.csv")
        _copy_file(map_dir / "map_atlas_best.osa", map_path)
        _copy_file(video_dir / "gravity_best.csv", video_dir / "gravity.csv")
        _copy_file(video_dir / "biases_best.csv", video_dir / "biases.csv")

    if total_attempts > 1:
        print(f"\nBest result: attempt {best_attempt}/{total_attempts} ({best_pct:.1f}% tracking)")

    if best_pct <= 0 or not map_path.is_file():
        raise RuntimeError(f"All {total_attempts} SLAM attempts failed")

    # --- Pass 2: Re-localization ---
    print("\nRunning pass 2 (re-localization to recover init frames)...")

    # Compute timeout from video duration
    with av.open(str(video_dir / "raw_video.mp4")) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
    pass2_timeout = max(duration * 10, 120)

    result2 = run_slam(
        video_dir,
        output_csv="mapping_camera_trajectory_pass2.csv",
        load_map=map_path,
        docker_image=docker_image,
        settings_path=settings_path,
        timeout_s=pass2_timeout,
    )

    if result2.returncode == 0 and result2.trajectory_path is not None:
        pct2 = result2.tracking_pct
        print(f"  Pass 2: {result2.tracked_frames}/{result2.total_frames} ({pct2:.1f}%)")

        if pct2 > best_pct:
            _copy_file(
                video_dir / "mapping_camera_trajectory_pass2.csv",
                video_dir / "mapping_camera_trajectory.csv",
            )
            print(f"  Pass 2 improved tracking: {best_pct:.1f}% -> {pct2:.1f}%")
        else:
            print(f"  Pass 2 did not improve ({pct2:.1f}% vs {best_pct:.1f}%), keeping pass 1")
    else:
        print(f"  Pass 2 failed, keeping pass 1")

    return map_path


def _run_slam_worker(
    cmd: list[str],
    cwd: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
) -> subprocess.CompletedProcess | subprocess.TimeoutExpired:
    """Worker for batch SLAM — runs in a thread."""
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=stdout_path.open("w"),
            stderr=stderr_path.open("w"),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return e


def batch_slam(
    video_dirs: list[Path],
    map_path: Path,
    *,
    num_workers: int | None = None,
    max_lost_frames: int = 60,
    timeout_multiple: float = 16,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
):
    """Localize multiple videos against a shared map.

    Args:
        video_dirs: list of directories, each with raw_video.mp4 + imu_data.json
        map_path: path to map_atlas.osa
        num_workers: parallel Docker containers (default: cpu_count // 2)
        max_lost_frames: terminate individual runs after N lost frames
        timeout_multiple: timeout = video_duration * this
        docker_image: Docker image name
        settings_path: SLAM settings YAML path
    """
    map_path = Path(map_path).absolute()
    settings_path = Path(settings_path).absolute()
    if not map_path.is_file():
        raise FileNotFoundError(f"Map not found: {map_path}")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    # Prepare all directories (IMU resample + mask) before launching threads
    jobs = []
    for vdir in video_dirs:
        vdir = Path(vdir).absolute()
        if (vdir / "camera_trajectory.csv").is_file():
            print(f"  Skipping {vdir.name} (camera_trajectory.csv exists)")
            continue

        imu_filename = _ensure_imu_resampled(vdir)
        _ensure_mask(vdir, vdir / "raw_video.mp4")

        # Get video duration for timeout
        with av.open(str(vdir / "raw_video.mp4")) as container:
            stream = container.streams.video[0]
            duration = float(stream.duration * stream.time_base)
        timeout = duration * timeout_multiple

        cmd = _build_docker_cmd(
            vdir,
            imu_filename=imu_filename,
            output_csv="camera_trajectory.csv",
            settings_path=settings_path,
            load_map=map_path,
            mask=True,
            max_lost_frames=max_lost_frames,
            docker_image=docker_image,
        )
        jobs.append((cmd, vdir, timeout))

    print(f"Running {len(jobs)} SLAM jobs with {num_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for cmd, vdir, timeout in jobs:
            fut = executor.submit(
                _run_slam_worker,
                cmd,
                str(vdir),
                vdir / "slam_stdout.txt",
                vdir / "slam_stderr.txt",
                timeout,
            )
            futures[fut] = vdir

        for fut in concurrent.futures.as_completed(futures):
            vdir = futures[fut]
            result = fut.result()
            if isinstance(result, subprocess.TimeoutExpired):
                print(f"  {vdir.name}: TIMEOUT")
            elif result.returncode != 0:
                print(f"  {vdir.name}: FAILED (rc={result.returncode})")
            else:
                total, tracked = _parse_tracking_rate(vdir / "camera_trajectory.csv")
                pct = 100.0 * tracked / total if total > 0 else 0
                print(f"  {vdir.name}: {tracked}/{total} ({pct:.1f}%)")
