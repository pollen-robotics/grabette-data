"""Docker-based SLAM orchestration.

Runs grabette_slam inside Docker for mapping (pass 1 + pass 2) and
batch localization against an existing map.
"""

import concurrent.futures
import multiprocessing
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

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
) -> tuple[list[str], str]:
    """Build the docker run command list. Returns (cmd, container_name)."""
    # Docker mount points
    data_mount = "/data"
    settings_mount = "/settings"

    # Unique container name so we can docker-kill it on timeout
    container_name = f"grabette-slam-{uuid.uuid4().hex[:8]}"

    cmd = [
        "docker", "run", "--rm", "-t",
        "--name", container_name,
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

    return cmd, container_name


def _read_slam_pipe(pipe, stdout_path: Path, show_progress: bool):
    """Reader thread: drain pipe line-by-line, write to log, optionally show progress."""
    pbar = None
    total_frames = None
    n_lost = 0

    with open(stdout_path, "w") as f_out:
        for raw_line in pipe:
            line = raw_line.rstrip("\r\n")
            f_out.write(line + "\n")
            f_out.flush()

            if not show_progress:
                continue

            if total_frames is None:
                m = re.search(r"There are (\d+) frames in total", line)
                if m:
                    total_frames = int(m.group(1))
                    pbar = tqdm(total=total_frames, unit="fr",
                                desc="  SLAM", leave=True)

            if pbar is not None and "Video FPS:" in line:
                pbar.update(100 - (pbar.n % 100) if pbar.n % 100 else 100)

            if "n_lost_frames=" in line:
                m = re.search(r"n_lost_frames=(\d+)", line)
                if m:
                    n_lost = int(m.group(1))
                    if pbar is not None:
                        pbar.set_postfix(lost=n_lost)

    # Pipe closed (process exited or was killed)
    if pbar is not None:
        if total_frames:
            pbar.n = total_frames
            pbar.refresh()
        pbar.close()


def run_slam(
    video_dir: Path,
    *,
    output_csv: str = "camera_trajectory.csv",
    save_map: Path | None = None,
    load_map: Path | None = None,
    output_gravity: str | None = None,
    output_biases: str | None = None,
    mask: bool = True,
    max_lost_frames: int = -1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
    timeout_s: float | None = None,
    log_prefix: str = "slam",
    show_progress: bool = True,
) -> SlamResult:
    """Run grabette_slam in Docker on a single video directory.

    Args:
        video_dir: directory containing raw_video.mp4 and imu_data.json
        output_csv: trajectory output filename (inside video_dir)
        save_map: if set, save map atlas to this path
        load_map: if set, load map atlas from this path
        output_gravity: gravity output filename (inside video_dir), or None
        output_biases: biases output filename (inside video_dir), or None
        mask: generate and apply device mask
        max_lost_frames: terminate after N lost frames (-1 = disabled)
        docker_image: Docker image name
        settings_path: SLAM settings YAML path
        timeout_s: subprocess timeout in seconds
        log_prefix: prefix for stdout/stderr log files
        show_progress: show tqdm progress bar

    Returns:
        SlamResult with tracking statistics
    """
    video_dir = Path(video_dir).absolute()
    settings_path = Path(settings_path).absolute()

    # Ensure prerequisites
    imu_filename = _ensure_imu_resampled(video_dir)
    if mask:
        _ensure_mask(video_dir, video_dir / "raw_video.mp4")

    cmd, container_name = _build_docker_cmd(
        video_dir,
        imu_filename=imu_filename,
        output_csv=output_csv,
        settings_path=settings_path,
        save_map=save_map,
        load_map=load_map,
        mask=mask,
        output_gravity=output_gravity,
        output_biases=output_biases,
        max_lost_frames=max_lost_frames,
        docker_image=docker_image,
    )

    stdout_path = video_dir / f"{log_prefix}_stdout.txt"
    stderr_path = video_dir / f"{log_prefix}_stderr.txt"

    # Pipe stdout through a reader thread for real-time progress.
    # Docker -t flag gives line-buffered output inside the container,
    # pipe gives low-latency delivery to the reader thread.
    # Stderr goes to a file (no need for real-time monitoring).
    returncode = -1
    reader = None
    try:
        with open(stderr_path, "w") as f_err:
            proc = subprocess.Popen(
                cmd,
                cwd=str(video_dir),
                stdout=subprocess.PIPE,
                stderr=f_err,
                text=True,
            )
            reader = threading.Thread(
                target=_read_slam_pipe,
                args=(proc.stdout, stdout_path, show_progress),
                daemon=True,
            )
            reader.start()

            try:
                proc.wait(timeout=timeout_s)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                # Kill the Docker container directly (more reliable than killing docker CLI)
                subprocess.run(
                    ["docker", "kill", container_name],
                    capture_output=True, timeout=10,
                )
                proc.wait(timeout=10)
                print(f"\n  SLAM timed out after {timeout_s:.0f}s")
                returncode = -1

    except Exception as e:
        print(f"  SLAM error: {e}")
        returncode = -1
    finally:
        if reader and reader.is_alive():
            reader.join(timeout=5)

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


def _run_attempt(
    attempt: int,
    video_dir: Path,
    map_dir: Path,
    *,
    docker_image: str,
    settings_path: Path,
    show_progress: bool = True,
) -> tuple[int, SlamResult]:
    """Run a single pass-1 mapping attempt. Returns (attempt_number, result)."""
    result = run_slam(
        video_dir,
        output_csv=f"mapping_traj_attempt{attempt}.csv",
        save_map=map_dir / f"map_atlas_attempt{attempt}.osa",
        output_gravity=f"gravity_attempt{attempt}.csv",
        output_biases=f"biases_attempt{attempt}.csv",
        docker_image=docker_image,
        settings_path=settings_path,
        log_prefix=f"slam_attempt{attempt}",
        show_progress=show_progress,
    )
    return attempt, result


def create_map(
    video_dir: Path,
    *,
    retries: int = 3,
    parallel: int = 1,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    settings_path: Path = DEFAULT_SETTINGS,
) -> Path:
    """Two-pass mapping: pass 1 (save_map + gravity/biases), pass 2 (load_map).

    Pass 1 runs 1+retries attempts (sequentially or in parallel), keeps best by
    tracking rate. Pass 2 re-localizes against the best map.

    Args:
        video_dir: directory containing raw_video.mp4 and imu_data.json
        retries: number of extra attempts for pass 1 (total = 1 + retries)
        parallel: number of pass-1 attempts to run simultaneously
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

    # Ensure prerequisites once before launching attempts
    _ensure_imu_resampled(video_dir)
    _ensure_mask(video_dir, video_dir / "raw_video.mp4")

    # --- Pass 1: Mapping ---
    if parallel <= 1:
        # Sequential mode: run one at a time with progress bars, early-stop at 90%
        results = _pass1_sequential(
            video_dir, map_dir, total_attempts,
            docker_image=docker_image, settings_path=settings_path,
        )
    else:
        # Parallel mode: launch all attempts at once
        results = _pass1_parallel(
            video_dir, map_dir, total_attempts, parallel,
            docker_image=docker_image, settings_path=settings_path,
        )

    # Pick the best attempt
    best_attempt, best_result = max(
        ((a, r) for a, r in results if r.trajectory_path is not None),
        key=lambda x: x[1].tracking_pct,
        default=(0, None),
    )

    if best_result is None or best_result.tracking_pct <= 0:
        raise RuntimeError(f"All {total_attempts} SLAM attempts failed")

    best_pct = best_result.tracking_pct
    print(f"\nBest result: attempt {best_attempt}/{total_attempts} "
          f"({best_result.tracked_frames}/{best_result.total_frames}, {best_pct:.1f}%)")

    # Copy best attempt's output files to canonical names
    _copy_file(video_dir / f"mapping_traj_attempt{best_attempt}.csv",
               video_dir / "mapping_camera_trajectory.csv")
    _copy_file(map_dir / f"map_atlas_attempt{best_attempt}.osa", map_path)
    _copy_file(video_dir / f"gravity_attempt{best_attempt}.csv",
               video_dir / "gravity.csv")
    _copy_file(video_dir / f"biases_attempt{best_attempt}.csv",
               video_dir / "biases.csv")

    # --- Pass 2: Re-localization ---
    print("\nRunning pass 2 (re-localization to recover init frames)...")

    # Timeout: pass 2 should be faster than pass 1 (no mapping, just localization).
    with av.open(str(video_dir / "raw_video.mp4")) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)
    pass2_timeout = max(duration * 5, 180)

    result2 = run_slam(
        video_dir,
        output_csv="mapping_camera_trajectory_pass2.csv",
        load_map=map_path,
        docker_image=docker_image,
        settings_path=settings_path,
        timeout_s=pass2_timeout,
        log_prefix="slam_pass2",
    )

    # Use pass 2 result if trajectory was written — even if the process timed out
    # (the binary may hang during shutdown after successfully writing the trajectory)
    if result2.trajectory_path is not None and result2.tracked_frames > 0:
        pct2 = result2.tracking_pct
        if result2.returncode != 0:
            print(f"  Pass 2: process exited with rc={result2.returncode} "
                  f"but trajectory exists: {result2.tracked_frames}/{result2.total_frames} ({pct2:.1f}%)")
        else:
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


def _pass1_sequential(
    video_dir: Path,
    map_dir: Path,
    total_attempts: int,
    *,
    docker_image: str,
    settings_path: Path,
) -> list[tuple[int, SlamResult]]:
    """Run pass-1 attempts sequentially with progress bars. Early-stops at >=90%."""
    results = []
    for attempt in range(1, total_attempts + 1):
        if total_attempts > 1:
            print(f"\n--- Pass 1, attempt {attempt}/{total_attempts} ---")
        else:
            print("Running SLAM mapping (pass 1)...")

        attempt_num, result = _run_attempt(
            attempt, video_dir, map_dir,
            docker_image=docker_image, settings_path=settings_path,
            show_progress=True,
        )
        results.append((attempt_num, result))

        if result.trajectory_path is None or result.tracked_frames == 0:
            print(f"  SLAM failed (return code {result.returncode})")
            continue

        pct = result.tracking_pct
        print(f"  Tracking: {result.tracked_frames}/{result.total_frames} ({pct:.1f}%)")

        if pct >= 90:
            if total_attempts > 1:
                print(f"  >= 90% tracking, stopping early")
            break

    return results


def _pass1_parallel(
    video_dir: Path,
    map_dir: Path,
    total_attempts: int,
    parallel: int,
    *,
    docker_image: str,
    settings_path: Path,
) -> list[tuple[int, SlamResult]]:
    """Run pass-1 attempts in parallel (no per-attempt progress bars)."""
    n_workers = min(parallel, total_attempts)
    print(f"\nRunning {total_attempts} pass-1 attempts ({n_workers} in parallel)...")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for attempt in range(1, total_attempts + 1):
            fut = executor.submit(
                _run_attempt,
                attempt, video_dir, map_dir,
                docker_image=docker_image, settings_path=settings_path,
                show_progress=False,
            )
            futures[fut] = attempt

        for fut in concurrent.futures.as_completed(futures):
            attempt_num, result = fut.result()
            results.append((attempt_num, result))

            if result.trajectory_path is None or result.tracked_frames == 0:
                print(f"  Attempt {attempt_num}: FAILED (rc={result.returncode})")
            else:
                pct = result.tracking_pct
                print(f"  Attempt {attempt_num}: "
                      f"{result.tracked_frames}/{result.total_frames} ({pct:.1f}%)")

    return results


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

        cmd, _ = _build_docker_cmd(
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
