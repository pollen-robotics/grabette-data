#!/usr/bin/env python3
"""Batch localization: run SLAM on multiple videos against an existing map."""

import click
from pathlib import Path

from grabette_data.slam import batch_slam, DEFAULT_DOCKER_IMAGE, DEFAULT_SETTINGS


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Parent directory containing episode subdirectories")
@click.option("-m", "--map_path", required=True, type=click.Path(exists=True),
              help="Path to map_atlas.osa")
@click.option("-n", "--num_workers", type=int, default=None,
              help="Parallel Docker containers (default: cpu_count // 2)")
@click.option("--max_lost_frames", type=int, default=60,
              help="Terminate after N lost frames per video")
@click.option("--timeout_multiple", type=float, default=16,
              help="timeout = video_duration * this")
@click.option("-d", "--docker_image", default=DEFAULT_DOCKER_IMAGE)
@click.option("-s", "--settings", default=str(DEFAULT_SETTINGS), type=click.Path(exists=True))
@click.option("--deterministic", is_flag=True, default=False,
              help="Run in deterministic mode (slower, reproducible)")
@click.option("--min_tracking_pct", type=float, default=50.0,
              help="Min tracking %% before retry in mapping mode")
@click.option("--no-retry", is_flag=True, default=False,
              help="Disable mapping retry for failed episodes")
def main(input_dir, map_path, num_workers, max_lost_frames, timeout_multiple,
         docker_image, settings, deterministic, min_tracking_pct, no_retry):
    input_dir = Path(input_dir).expanduser().absolute()

    # Find all episode directories with raw_video.mp4
    video_dirs = sorted([
        p.parent for p in input_dir.glob("*/raw_video.mp4")
    ])
    print(f"Found {len(video_dirs)} video directories")

    if not video_dirs:
        raise click.ClickException(f"No raw_video.mp4 found under {input_dir}")

    batch_slam(
        video_dirs,
        Path(map_path),
        num_workers=num_workers,
        max_lost_frames=max_lost_frames,
        timeout_multiple=timeout_multiple,
        deterministic=deterministic,
        min_tracking_pct=min_tracking_pct,
        retry_mapping=not no_retry,
        docker_image=docker_image,
        settings_path=Path(settings),
    )


if __name__ == "__main__":
    main()
