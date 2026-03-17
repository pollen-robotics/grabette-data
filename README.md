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
├── map/map_atlas.osa           SLAM map (pass 1)
├── mapping_camera_trajectory.csv   full trajectory (pass 2, 100% tracking)
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

## Usage

### 1. Create map from a mapping video

Runs two-pass SLAM: pass 1 builds the map (with retries), pass 2 re-localizes against it to recover initialization frames.

```bash
uv run python scripts/create_map.py \
  -i ~/data/episodes/mapping_session \
  --retries 3
```

Outputs in the episode directory: `map/map_atlas.osa`, `mapping_camera_trajectory.csv`, `gravity.csv`, `biases.csv`.

### 2. Batch localization

Localize multiple episode videos against an existing map. Runs parallel Docker containers.

```bash
uv run python scripts/batch_slam.py \
  -i ~/data/episodes \
  -m ~/data/episodes/mapping_session/map/map_atlas.osa \
  -n 4
```

Outputs `camera_trajectory.csv` in each episode directory.

### Deterministic mode

ORB-SLAM3's LocalMapping thread runs asynchronously, which means keyframe decisions can vary between runs due to race conditions. The `--deterministic` flag forces tracking to wait for LocalMapping to finish after every frame, making results fully reproducible.

```bash
# Deterministic map creation (single attempt, no retries needed)
uv run python scripts/create_map.py \
  -i ~/data/episodes/mapping_session \
  --deterministic

# Deterministic batch localization
uv run python scripts/batch_slam.py \
  -i ~/data/episodes \
  -m ~/data/episodes/mapping_session/map/map_atlas.osa \
  --deterministic
```

Trade-offs:
- **Slower**: tracking blocks on LocalMapping after each frame instead of running in parallel
- **Reproducible**: identical results across runs — useful for debugging and validation
- **Simpler**: `create_map` with `--deterministic` forces `retries=0` and `parallel=1` (single pass, no retry logic needed since results don't vary)

### 3. Generate LeRobot dataset

Converts SLAM trajectories + raw data into a LeRobot v3 dataset.

```bash
uv run python scripts/generate_dataset.py \
  --input_dir ~/data/episodes \
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

### 4. Push dataset to Hugging Face Hub

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

### 5. Visualize trajectory

Interactive 3D visualization with [Rerun](https://rerun.io/): trajectory, camera frustum, video overlay, and IMU time series.

```bash
uv run python scripts/visualize_trajectory.py ~/data/episodes/mapping_session
uv run python scripts/visualize_trajectory.py ~/data/episodes/mapping_session --video-skip 1
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
│   ├── create_map.py           # CLI: two-pass mapping
│   ├── batch_slam.py           # CLI: batch localization
│   ├── generate_dataset.py     # CLI: SLAM outputs → LeRobot v3
│   ├── push_to_hub.py          # CLI: upload dataset to Hugging Face Hub
│   └── visualize_trajectory.py # CLI: Rerun 3D visualization
└── test_data/
    └── grabette9/              # Test episode (raw_video.mp4 + imu_data.json)
```

## Hardware

- **Camera**: Raspberry Pi camera module, 1296x972 @ 50fps, fisheye lens (KannalaBrandt8)
- **IMU**: BMI088 (Bosch), 200Hz, raw gyro + accel
- **Mounting**: back-to-back, camera and IMU centers aligned along z-axis, 11.15mm apart
- **Angle sensors**: two joint encoders at 100Hz (proximal + distal)
