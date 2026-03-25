# grabette-data

SLAM orchestration and LeRobot dataset generation for the GRABETTE project.

Takes raw episode recordings (video + IMU) from [Grabette](https://github.com/SteveNguyen/grabette), runs ORB-SLAM3 in Docker to produce camera trajectories, and converts everything into a [LeRobot v3](https://huggingface.co/docs/lerobot) dataset (Parquet + MP4) ready for policy training.

## Data flow

```
Episode directory (from Grabette)
├── raw_video.mp4          1296x972 @ ~46fps
├── imu_data.json          ACCL 200Hz, GYRO 200Hz, ANGL 100Hz
└── metadata.json

    │  create_map.py / batch_slam.py
    ▼

├── imu_data_resampled.json     uniform 200Hz, ANGL stripped
├── slam_mask.png               device body mask
├── slam_metadata.json          SLAM run info (method, tracking %, map file...)
├── map/map_atlas.osa           SLAM map (pass 1, mapping video only)
├── mapping_camera_trajectory.csv   full trajectory (pass 2, mapping video only)
├── camera_trajectory.csv       trajectory (episodes)
├── gravity.csv                 3x3 rotation matrix
└── biases.csv                  IMU biases

    │  generate_dataset.py
    ▼

LeRobot v3 dataset/
├── meta/info.json, stats.json, tasks.parquet, episodes/
├── data/chunk-NNN/file-NNN.parquet
└── videos/observation.images.cam0/chunk-NNN/file-NNN.mp4
```

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd grabette-data
uv sync
```

The SLAM step requires Docker with the pre-built ORB-SLAM3 image:

```bash
docker pull pollenrobotics/orbslam3-headless
```

## Quick start

Standard workflow for processing a new dataset:

```bash
# 0. (Optional) Check calibration on a new device
uv run python scripts/check_calibration.py \
  --video /path/to/charuco_video.mp4

# 1. Create map from the mapping video
uv run python scripts/create_map.py \
  -i ~/data/dataset/mapping \
  --retries 3

# 2. Batch localize all episodes (with automatic mapping retry)
uv run python scripts/batch_slam.py \
  -i ~/data/dataset \
  -m ~/data/dataset/mapping/map/map_atlas.osa \
  -n 4

# 3. (Optional) Visualize a trajectory
uv run python scripts/visualize_trajectory.py ~/data/dataset/some_episode

# 4. Generate LeRobot dataset
uv run python scripts/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id user/dataset-name \
  --task "task description" \
  --root ~/lerobot_datasets

# 5. Push to HuggingFace Hub
uv run python scripts/push_to_hub.py \
  --repo_id user/dataset-name \
  --root ~/lerobot_datasets
```

## Usage details

### 1. Check calibration (optional)

Verify that existing camera intrinsics fit a new device by detecting ChArUco corners and computing fisheye reprojection error. Useful when assembling a new device with the same lens model.

```bash
uv run python scripts/check_calibration.py \
  --video /path/to/charuco_video.mp4
```

Thresholds: median < 1.0px = good, 1.0–2.0px = marginal, > 2.0px = recalibrate.

### 2. Create map from a mapping video

Runs two-pass SLAM: pass 1 builds the map (with retries), pass 2 re-localizes against it to recover initialization frames.

```bash
uv run python scripts/create_map.py \
  -i ~/data/dataset/mapping \
  --retries 3
```

Outputs in the episode directory: `map/map_atlas.osa`, `mapping_camera_trajectory.csv`, `gravity.csv`, `biases.csv`.

Multiple retries are recommended (default 3) since ORB-SLAM3 is non-deterministic — each attempt explores a different optimization path, and the best map is kept. More retries = better chance of a high-quality map.

### 3. Batch localization

Localize multiple episode videos against an existing map. Runs parallel Docker containers.

```bash
uv run python scripts/batch_slam.py \
  -i ~/data/dataset \
  -m ~/data/dataset/mapping/map/map_atlas.osa \
  -n 4
```

Batch SLAM runs in two phases:

1. **Localization** — each episode localizes against the shared map. If SLAM loses tracking and resets the map (unrecoverable in localization-only mode), the container is killed immediately instead of hanging until timeout.

2. **Mapping retry** — episodes that failed localization or tracked below `--min_tracking_pct` (default 50%) are retried in full mapping mode. SLAM builds its own map from scratch for these episodes. The trajectory will be in its own coordinate frame (not the shared map's), but still usable for per-episode pose data.

Each episode gets a `slam_metadata.json` recording the method used, tracking stats, map file, and retry history.

Options:
- `--min_tracking_pct 50.0` — retry threshold (default 50%)
- `--no-retry` — disable mapping retry (localization only)
- `--max_lost_frames 60` — terminate localization after N consecutive lost frames

### Deterministic mode

ORB-SLAM3's LocalMapping thread runs asynchronously, which means keyframe decisions can vary between runs due to race conditions. The `--deterministic` flag forces tracking to wait for LocalMapping to finish after every frame, making results fully reproducible.

```bash
# Deterministic map creation (single attempt, no retries needed)
uv run python scripts/create_map.py \
  -i ~/data/dataset/mapping \
  --deterministic

# Deterministic batch localization
uv run python scripts/batch_slam.py \
  -i ~/data/dataset \
  -m ~/data/dataset/mapping/map/map_atlas.osa \
  --deterministic
```

Trade-offs:
- **Slower**: tracking blocks on LocalMapping after each frame instead of running in parallel
- **Reproducible**: identical results across runs — no need for retries since results don't vary
- **Simpler**: `create_map` with `--deterministic` forces `retries=0` and `parallel=1`

Deterministic mode gives LocalMapping unlimited time to optimize each keyframe before the next frame arrives, so map quality should be equal or better than normal mode. Normal mode with retries compensates for its randomness by running multiple attempts and keeping the best — deterministic mode doesn't need this since it converges to the same result every time.

### 4. Generate LeRobot dataset

Converts SLAM trajectories + raw data into a LeRobot v3 dataset.

```bash
uv run python scripts/generate_dataset.py \
  --input_dir ~/data/dataset \
  --repo_id myuser/grabette-demo \
  --task "cup manipulation" \
  --root ~/lerobot_datasets
```

#### Dataset features

| Feature | dtype | shape | Source |
|---------|-------|-------|--------|
| `observation.images.cam0` | video | (3, 720, 960) | raw_video.mp4 (resized) |
| `observation.pose` | float32 | (6,) | trajectory: [x, y, z, ax, ay, az] |
| `observation.joints` | float32 | (2,) | ANGL stream: [proximal, distal] rad |
| `action` | float32 | (2,) | joints[t+1] (next-step angle) |

Poses are gravity-aligned (Z-up) directly from ORB-SLAM3's IMU initialization. The 6D pose is position + axis-angle rotation.

### 5. Push dataset to Hugging Face Hub

```bash
# Login (one-time)
huggingface-cli login

# Push
uv run python scripts/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets

# Or as private
uv run python scripts/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets \
  --private
```

### 6. Visualize trajectory

Interactive 3D visualization with [Rerun](https://rerun.io/): trajectory, camera frustum, video overlay, and IMU time series.

```bash
uv run python scripts/visualize_trajectory.py ~/data/dataset/mapping_session
uv run python scripts/visualize_trajectory.py ~/data/dataset/mapping_session --video-skip 1
```

## Project structure

```
grabette-data/
├── pyproject.toml
├── config/
│   └── rpi_bmi088_slam_settings.yaml   # SLAM settings for RPi camera + BMI088
├── grabette_data/
│   ├── imu.py           # IMU deduplication + resampling to uniform 200Hz
│   ├── mask.py          # Device body mask polygon (auto-scales to resolution)
│   ├── slam.py          # Docker SLAM orchestration (run_slam, create_map, batch_slam)
│   ├── trajectory.py    # CSV parsing, quaternion→axis-angle, ANGL interpolation
│   └── dataset.py       # LeRobot v3 dataset builder
├── scripts/
│   ├── check_calibration.py      # CLI: verify intrinsics on new device
│   ├── create_map.py             # CLI: two-pass mapping
│   ├── batch_slam.py             # CLI: batch localization + mapping retry
│   ├── generate_dataset.py       # CLI: SLAM outputs → LeRobot v3
│   ├── push_to_hub.py            # CLI: upload dataset to Hugging Face Hub
│   └── visualize_trajectory.py   # CLI: Rerun 3D visualization
└── test_data/
    └── grabette9/                # Test episode (raw_video.mp4 + imu_data.json)
```

## Hardware

- **Camera**: Raspberry Pi camera module, 1296x972 @ 50fps, fisheye lens (KannalaBrandt8)
- **IMU**: BMI088 (Bosch), 200Hz, raw gyro + accel
- **Mounting**: back-to-back, camera and IMU centers aligned along z-axis, 11.15mm apart
- **Angle sensors**: two joint encoders at 100Hz (proximal + distal)
