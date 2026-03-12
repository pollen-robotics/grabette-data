#!/usr/bin/env python3
"""Create SLAM map from a single mapping video (two-pass)."""

import click
from pathlib import Path

from grabette_data.slam import create_map, DEFAULT_DOCKER_IMAGE, DEFAULT_SETTINGS


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Directory containing raw_video.mp4 and imu_data.json")
@click.option("--retries", type=int, default=3,
              help="Retry pass 1 up to N times, keeping best result")
@click.option("-d", "--docker_image", default=DEFAULT_DOCKER_IMAGE,
              help="Docker image name")
@click.option("-s", "--settings", default=str(DEFAULT_SETTINGS), type=click.Path(exists=True),
              help="SLAM settings YAML")
def main(input_dir, retries, docker_image, settings):
    video_dir = Path(input_dir).expanduser().absolute()
    for fn in ["raw_video.mp4", "imu_data.json"]:
        if not (video_dir / fn).is_file():
            raise click.ClickException(f"Missing {fn} in {video_dir}")

    map_path = create_map(
        video_dir,
        retries=retries,
        docker_image=docker_image,
        settings_path=Path(settings),
    )
    print(f"\nMap saved to: {map_path}")


if __name__ == "__main__":
    main()
