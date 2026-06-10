<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,2,3&height=200&section=header&text=TE3003B%20Final%20Project&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Robotics%20and%20Intelligent%20Systems%20Implementation&descAlignY=58&descColor=89b4fa" alt="banner" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/ROS2-Humble-00599C?style=for-the-badge&logo=ros&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Final%20Project-4CAF50?style=for-the-badge&logo=check&logoColor=white" />
  <img src="https://img.shields.io/badge/Robot-Puzzlebot-FF6B35?style=for-the-badge&logo=robot&logoColor=white" />
  <img src="https://img.shields.io/badge/Environment-Real%20World-2196F3?style=for-the-badge&logo=warehouse&logoColor=white" />
</p>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-nodes">Nodes</a> •
  <a href="#-launch-files">Launch Files</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a>
</p>

</div>

---

## 🏭 Overview

**TE3003B Final Project** is a complete ROS2-based autonomy stack for a **Puzzlebot** (Manchester Robotics) differential-drive robot performing autonomous pallet transport in a known warehouse environment. The system implements a full perception-localization-planning-control-manipulation pipeline operating on real hardware.

> **Objective:** Deliver a production-quality autonomous warehouse robot stack on real Puzzlebot hardware — detecting pallets via QR codes, collecting them with a FPGA-controlled lift, navigating to the correct delivery truck identified by YOLO, and depositing the pallet. Integrates EKF odometry, SLAM, MCL, ArUco landmark anchoring, VFH+ obstacle avoidance, A\* path planning, a hierarchical FSM mission manager, and voice control — all on ROS 2 Humble without Nav2 at runtime.

---

## 🧠 Architecture

The system is organized around a layered architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PERCEPTION                                     │
│  aruco_detector ──► /aruco/*       │  yolo_vision ──► /yolo/*              │
│  (4X4_50 + 6X6_50 dual-dict)       │  (YOLOv8, custom weights)             │
│  qr_detector    ──► /qr/*          │  truck_pos   ──► /yolo/world_pos      │
│  (fisheye undistort + solvePnP)    │  (YOLO world-frame projection)        │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                          LOCALIZATION / MAPPING                             │
│  slam_node       ──► /slam_map + /icp/pose  (ICP scan matching + grid)     │
│  mcl_node        ──► /mcl/pose              (SIR particle filter)          │
│  aruco_localizer ──► /aruco/pose            (landmark anchoring)           │
│                    ▼                                                         │
│  puzzlebotOdometry ──► /odom + TF  (EKF: dead-reckoning + fusion)          │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                              PLANNING                                       │
│  astar_planner   ──► /goal          (A*, dynamic map, obstacle inflation)  │
│  mission_manager ──► /astar/goal    (HFSM: 20 states + 4 recovery states)  │
│  mission_planner ──► /astar/goal    (legacy YAML waypoint sequencer)       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                         COLLECTION / DELIVERY                               │
│  qr_aligner_node ──► /cmd_vel       (FSM: fisheye align to QR + solvePnP) │
│  qr_zone_checker ──► /collect/trigger (conveyor vs rack classifier)        │
│  truck_aligener  ──► /truck_align/done (9-state delivery FSM + YOLO)       │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                              CONTROL                                        │
│  puzzlebotGoToGoal ──► /cmd_raw   (dual PID v+ω, align-and-drive)          │
│  vfh_plus          ──► /cmd_vel   (VFH+ obstacle avoidance, LiDAR)         │
│  puzzlebotController ──► /VelocitySetR, /VelocitySetL                      │
│  (dual PID wheel velocity, inverse kinematics, anti-windup)                │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────────┐
│                            MANIPULATION                                     │
│  spi_servo_node  ──► /lift_done, /lift_state   (Tang Nano 20K via SPI)     │
│  (10-state FPGA SM: IDLE→TO_N1/N2→AT_N1/N2→LIFTING→HOLD→LOWERING)        │
│  voice_action_node ──► /cmd_vel, /lift_auto    (HMM + VQ speech control)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### EKF Source Priority

The odometry node implements an automatic source-switching state machine:

| Priority | Source | State |
|---|---|---|
| 1 (highest) | `/mcl/pose` | `MCL_ACTIVE` / `MCL_PRIORITY` |
| 2 | `/aruco/pose` | `ARUCO_ACTIVE` / `ARUCO_PRIORITY` |
| 3 | `/icp/pose` | `ICP_ACTIVE` |
| 4 (fallback) | Dead-reckoning only | `PREDICT_ONLY` |

### Mission Manager HFSM

The mission manager implements a hierarchical FSM with 20 operational states organized in four super-states, plus 4 recovery states accessible from any mission state:

| Super-state | States |
|---|---|
| **System** | `INIT`, `IDLE`, `NAVSELECT`, `TELEOP`, `VOICE_CMD`, `MAPPING` |
| **Autonomous Mission** | `AUTONAV_INIT`, `ASTAR_EXPLORE`, `QR_ALIGN` |
| **Collect** | `COLLECT_APPROACH`, `COLLECT_INSERT_ACQUIRE` |
| **Delivery** | `GO2GOAL`, `TRUCK_ALIGN`, `DROP_PALLET`, `MISSION_DONE` |
| **Recovery** | `QR_RECOVERY`, `MANIP_RECOVERY`, `NAV_RECOVERY`, `YOLO_RECOVERY`, `E_STOP` |

---

## 📁 Project Structure

```
mcrxe80-final-challenge/
├── Camera Calibration/
│   ├── calib_imgs/                   # Checkerboard calibration images
│   ├── calibrar.py                   # Fisheye calibration script (Zhang's method)
│   └── save_imgs.py                  # Calibration image capture
│
├── Tang20K/
│   ├── slave_lift.v                  # Verilog FSM for Tang Nano 20K FPGA (dual servo PWM + SPI slave)
│   └── lift.cst                      # Pin constraints file
│
├── jetson/
│   ├── spi_servo_node.py             # Jetson → Tang Nano 20K SPI bridge (lift driver)
│   └── camera.py                     # Camera node for Jetson
│
├── rviz_bonito.rviz                  # RViz configuration for full-stack visualization
├── comandos puzzle                   # Quick-reference startup commands (SSH, launch, rosbridge)
│
└── src/
    ├── iolair/                        # Base robot package (actuation + odometry)
    │   ├── iolair/
    │   │   ├── puzzlebotController.py   # PID wheel velocity controller
    │   │   ├── puzzlebotGoToGoal.py     # Dual-PID goal navigation → /cmd_raw
    │   │   ├── puzzlebotOdometry.py     # EKF odometry (MCL/ICP/ArUco fusion)
    │   │   └── puzzlebotTeleop.py       # Keyboard teleoperation
    │   ├── launch/
    │   │   ├── manchester.launch.py     # Full autonomous stack (primary entry point)
    │   │   ├── odometry.launch.py       # Odometry + Controller only
    │   │   └── qr_align_tryout.launch.py # QR alignment integration test
    │   ├── maps/
    │   │   ├── SLAM_map.pgm             # Saved occupancy map
    │   │   ├── SLAM_map.yaml            # Map metadata
    │   │   └── map_pista.pgm            # Arena pre-built map
    │   ├── configs/
    │   │   ├── aruco_landmarks.yaml     # Global ArUco landmark positions
    │   │   └── waypoints.yaml           # Delivery waypoints (trucks + detection WPs)
    │   └── setup.py
    │
    ├── Navigation/                    # Planning + mission management
    │   ├── Navigation/
    │   │   ├── astar_planner.py         # A* path planner on occupancy grid
    │   │   ├── goal_bridge.py           # RViz /clicked_point → /astar/goal bridge
    │   │   ├── mission_manager.py       # Hierarchical FSM (20 states, 4 recovery)
    │   │   ├── mission_planner.py       # Legacy YAML waypoint sequencer
    │   │   ├── qr_aligner_node.py       # QR aligner (also registered here for Navigation launch)
    │   │   ├── truck_aligener.py        # 9-state truck delivery FSM (YOLO + geometry)
    │   │   └── vfh_plus.py              # VFH+ obstacle avoidance layer
    │   ├── configs/
    │   │   └── exploration_waypoints.yaml
    │   ├── maps/
    │   │   ├── SLAM_map.pgm
    │   │   └── map_pista.pgm
    │   └── setup.py
    │
    ├── Vision/                        # Perception nodes
    │   ├── Vision/
    │   │   ├── aruco_detector.py        # Dual-dict ArUco detector (4X4_50 + 6X6_50)
    │   │   ├── yolo_vision.py           # YOLOv8 inference (custom weights)
    │   │   ├── qr_detector.py           # Fisheye-corrected QR detection + pose
    │   │   ├── qr_aligner_node.py       # FSM: visual alignment to pallet via QR
    │   │   ├── qr_zone_checker.py       # Conveyor vs rack zone classifier
    │   │   ├── truck_pos.py             # YOLO world-frame position with lateral offset
    │   │   ├── fisheye_params.npz       # Camera calibration (fisheye, numpy format)
    │   │   └── fisheye_params.json      # Camera calibration (fisheye, JSON format)
    │   └── setup.py
    │
    ├── LocalizationMapping/           # SLAM + localization
    │   ├── LocalizationMapping/
    │   │   ├── aruco_localizer.py       # ArUco landmark anchoring EKF correction
    │   │   ├── aruco_map_publisher.py   # Landmark RViz visualization
    │   │   ├── mapping_node.py          # Map merger (max-occupancy accumulator)
    │   │   ├── mcl_node.py              # SIR particle filter localization
    │   │   └── slam_node.py             # Online SLAM (ICP + occupancy grid)
    │   ├── configs/
    │   │   └── aruco_landmarks.yaml
    │   ├── maps/
    │   │   ├── SLAM_map.pgm
    │   │   └── map_pista.pgm
    │   └── setup.py
    │
    └── voice_hmm_ros/                 # Voice command package
        ├── voice_hmm_ros/
        │   ├── voice_action_node.py     # HMM + VQ speech recognition → robot actions
        │   ├── hmm_from_scratch.py      # Baum-Welch HMM trainer/recognizer
        │   ├── run_hmm.py               # MFCC feature extraction + VQ codebook
        │   ├── grab_audio.py            # Microphone capture utility
        │   └── resultados_hmm_bw_tol05/
        │       └── models/              # Pre-trained HMM models (10 voice commands)
        └── setup.py
```

---

## 🔧 Nodes

### `iolair` package

| Node | Executable | Description |
|---|---|---|
| **puzzlebotController** | `controller` | Closed-loop PID wheel velocity controller at 50 Hz. Implements inverse differential kinematics, per-wheel PID with anti-windup, and integral reset on direction change. |
| **puzzlebotGoToGoal** | `go_to_goal` | Dual-PID navigation to a `Pose2D` goal. Separates linear and angular PID loops. Uses align-and-drive logic: rotates to face goal before advancing. Publishes to `/cmd_raw` (before the VFH+ safety layer). |
| **puzzlebotOdometry** | `odometry` | Full EKF odometry fusing encoder dead-reckoning with up to three external pose sources (MCL, ICP, ArUco). Joseph-form covariance updates, innovation gating, and automatic source-switching with configurable timeouts. |
| **puzzlebotTeleop** | `teleop` | Keyboard teleoperation with velocity ramping at 50 Hz. W/S/A/D keys, Space for emergency stop. Thread-safe with guaranteed terminal restoration. |

### `Navigation` package

| Node | Executable | Description |
|---|---|---|
| **astar_planner** | `astar_planner` | A\* path planner on occupancy grids. Dynamically selects between `/slam_map` and `/map` at runtime, inflates obstacles, applies line-of-sight shortcutting, and feeds waypoints to GoToGoal. Publishes `/astar/status` (PLANNING / MOVING / GOAL_REACHED / NO_PATH). |
| **mission_manager** | `mission_manager` | Hierarchical FSM central node. Orchestrates the full pallet-transport mission across 20 states and 4 recovery states. Handles QR detection, lift control handshake (via `/lift_done` confirmation), truck alignment delegation, and a `RecoveryManager` with per-state retry counters and anti-infinite-loop guards. Supports checkpoint persistence and E_STOP from `/emergency_stop Bool`. |
| **truck_aligener** | *(launched directly)* | 9-state delivery FSM: `GOTO_DETECTION_WP → SEARCH_TRUCK → ALIGNING → APPROACH_FINAL → ADVANCING → LOWERING → BACK_AWAY → DONE / ABORT`. Uses YOLO to identify the correct truck logo, computes a lateral-offset delivery goal in the world frame, and uses GoToGoal for final approach. Bypasses VFH+ via `/align/active` during insertion. |
| **vfh_plus** | `vfh_plus` | VFH+ obstacle avoidance layer sitting between GoToGoal (`/cmd_raw`) and the controller (`/cmd_vel`). Builds a polar obstacle histogram from `/scan`, steers through free sectors, and emits `PASS / BRAKE / VFH_STEER / REFLEX_STOP` on `/reflex_status`. Bypassed automatically during pallet alignment via `/align/active`. |
| **mission_planner** | `mission_planner` | Legacy YAML waypoint sequencer. Reads `exploration_waypoints.yaml` and publishes goals to `/astar/goal` sequentially with configurable per-goal timeout. Used for pre-mission scouting or standalone waypoint demos. |
| **goal_bridge** | `rviz_goal_bridge` | Converts RViz `/clicked_point` (PointStamped) into `/astar/goal` (Pose2D), enabling click-to-navigate from RViz. |
| **qr_aligner_node** | `qr_align_node` | Also registered in the Navigation package to allow use from Navigation-only launches. See Vision package for full description. |

### `Vision` package

| Node | Executable | Description |
|---|---|---|
| **aruco_detector** | `aruco_detector` | Dual-dictionary ArUco detector (`4X4_50` IDs 0–10 for internal/external WPs, `6X6_50` IDs 0–6 for named waypoints). Estimates 6-DOF pose via `solvePnP`. Publishes ID, distance, angle, label, and annotated image. |
| **yolo_vision** | `yolo_vision` | YOLOv8 inference node using custom weights. Publishes annotated image and JSON detections (`class`, `conf`, `bbox`, `bbox_cx`) on `/yolo/detecciones`. |
| **qr_detector** | `qr_detector` | Fisheye-corrected QR code detector. Undistorts frames using calibrated `fisheye_params.npz` before running `cv2.QRCodeDetector`. Publishes decoded payload (`/qr/data`), distance (`/qr/distance`), and horizontal angle (`/qr/angle`). |
| **qr_aligner_node** | `qr_align_node` | Visual P-controller that aligns the robot to a detected QR code using `solvePnP` on the fisheye-undistorted frame. Computes linear and angular errors from the 3D translation vector, applies a 0.5 s safety stop if no QR is visible. Used during `COLLECT_APPROACH` for precise fork positioning. |
| **qr_zone_checker** | `qr_zone_checker` | Classifies a detected QR position as `conveyor` or `rack` by projecting the QR world position (from `/qr/distance` + `/qr/angle` + `/odom`) against hardcoded zone polygons. Publishes `/collect/trigger` (String) and `/qr/world_pos` (PointStamped). |
| **truck_pos** | `yolo_world_pos` | Projects YOLO detections into the world frame using `/yolo/distance`, `/yolo/angle`, and `/odom`. Applies a configurable lateral offset (default 20 cm right) to compute the truck entry point. Publishes `/yolo/world_pos` (PointStamped). |

### `LocalizationMapping` package

| Node | Executable | Description |
|---|---|---|
| **aruco_localizer** | `aruco_localizer` | Drift correction by landmark anchoring against `aruco_landmarks.yaml`. Unknown markers are anchored dynamically on first observation. Adaptive noise covariance scales with distance². |
| **aruco_map_publisher** | `aruco_map_publisher` | Publishes ArUco landmark positions as a `MarkerArray` on `/aruco/markers` for RViz visualization. |
| **mcl_node** | `mcl` | SIR Particle Filter localization. Vectorised likelihood-field sensor model with precomputed EDT. Fuses EKF odometry (motion model) with LiDAR scans. Supports `/initialpose` from RViz. |
| **slam_node** | `slam` | Online SLAM building an occupancy grid with log-odds updates and Bresenham ray-casting. ICP scan matching corrects drift between keyframes. Publishes `/icp/pose` for EKF fusion and `/slam_map` for A\*. |
| **mapping_node** | `mapping_node` | Map merger that accumulates SLAM snapshots using a max-occupancy strategy. Occupied cells are never freed. Expands automatically as the map grows. |

### `voice_hmm_ros` package

| Node | Executable | Description |
|---|---|---|
| **voice_action_node** | `voice_action_node` | Integrated voice control node. Records microphone audio, runs HMM + VQ recognition, and executes robot actions at 20 Hz. Supports 10 commands: `avanza`, `atras`, `izquierda`, `derecha`, `gira`, `detente`, `toma` (lift n1 → hold), `arriba` (lift n2 → hold), `suelta` / `abajo` (lift down). Thread-safe with automatic lift hold sequencing. |

### Jetson (standalone, not a ROS package)

| File | Description |
|---|---|
| `jetson/spi_servo_node.py` | ROS 2 node running on the Jetson Nano. Bridges `/lift_auto` (String) and `/lift_trigger` (Int8) topics to the Tang Nano 20K FPGA via SPI. Implements a 10-state FPGA state machine mirror, anti-glitch MISO reading, poll-silence window around timer expirations, and a retry/timeout watchdog (200 ms retry, 1 s timeout) that publishes `<label>_TIMEOUT` on `/lift_done` to unblock the pipeline. Valid auto transitions: `IDLE→n1/n2`, `AT_N1/AT_N2→hold`, `HOLD→down`. |
| `jetson/camera.py` | Camera node for Jetson Nano. Captures and publishes frames from the onboard fisheye camera. |

### Tang Nano 20K FPGA (`Tang20K/`)

| File | Description |
|---|---|
| `Tang20K/slave_lift.v` | Verilog implementation of the lift state machine running on the Tang Nano 20K (GW2AR-18C). Implements dual 360° servo PWM control with SPI slave interface and MISO feedback. Includes fixes for timer race conditions (timer_fired flag), robust bit counter (5-bit), and MOSI/SCK alignment. States: `IDLE → TO_N1/TO_N2 → AT_N1/AT_N2 → LIFTING → HOLD → LOWERING`. |
| `Tang20K/lift.cst` | Physical pin constraint file for the Tang Nano 20K. |

---

## 📡 Key Topics

| Topic | Type | Description |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command (after VFH+ safety layer) |
| `/cmd_raw` | `geometry_msgs/Twist` | Velocity command (before safety layer, from GoToGoal) |
| `/odom` | `nav_msgs/Odometry` | EKF-fused odometry + covariance |
| `/slam_map` | `nav_msgs/OccupancyGrid` | Live SLAM map (latched) |
| `/map` | `nav_msgs/OccupancyGrid` | Static map served by nav2_map_server |
| `/mcl/pose` | `PoseWithCovarianceStamped` | MCL corrected pose |
| `/icp/pose` | `PoseWithCovarianceStamped` | ICP-corrected pose from SLAM node |
| `/aruco/pose` | `PoseWithCovarianceStamped` | ArUco landmark correction |
| `/ekf/active_source` | `std_msgs/String` | Current EKF fusion source |
| `/astar/goal` | `geometry_msgs/Pose2D` | Goal input to A\* planner |
| `/astar/status` | `std_msgs/String` | PLANNING / MOVING / GOAL_REACHED / NO_PATH |
| `/astar/cancel` | `std_msgs/String` | Cancel active path (used by E_STOP) |
| `/goal` | `geometry_msgs/Pose2D` | A\* waypoint output to GoToGoal |
| `/mission/mode` | `std_msgs/String` | HMI command: `auto / stop / estop / teleop / voice / resume / reset` |
| `/mission/state` | `std_msgs/String` | Current FSM state → HMI |
| `/mission/context` | `std_msgs/String` | JSON mission context (checkpoint data) |
| `/recovery/active` | `std_msgs/String` | Active recovery state (diagnostics) |
| `/emergency_stop` | `std_msgs/Bool` | Hardware E-stop input (bypasses tick loop) |
| `/lift_auto` | `std_msgs/String` | Lift command: `n1 / n2 / hold / down` |
| `/lift_trigger` | `std_msgs/Int8` | Manual lift: `+1=up / -1=down / 0=stop` |
| `/lift_done` | `std_msgs/String` | `AT_N1 / AT_N2 / HOLD / DOWN / <label>_TIMEOUT` |
| `/lift_state` | `std_msgs/String` | Current FPGA state name (continuous) |
| `/align/active` | `std_msgs/Bool` | VFH+ bypass flag (True during pallet/truck alignment) |
| `/reflex_status` | `std_msgs/String` | VFH+ status: `PASS / BRAKE / VFH_STEER / REFLEX_STOP` |
| `/truck_align/cmd` | `std_msgs/String` | Truck alignment trigger: `align:<client_name>` |
| `/truck_align/done` | `std_msgs/String` | `SUCCESS / ABORT` |
| `/collect/trigger` | `std_msgs/String` | Zone type for collection: `conveyor / rack` |
| `/collect/qr_payload` | `std_msgs/String` | QR payload pre-load for truck_aligener |
| `/qr/data` | `std_msgs/String` | Decoded QR payload |
| `/qr/distance` | `std_msgs/Float32` | Distance to QR code [m] |
| `/qr/angle` | `std_msgs/Float32` | Horizontal angle to QR [rad] |
| `/qr/world_pos` | `geometry_msgs/PointStamped` | QR global position |
| `/yolo/detecciones` | `std_msgs/String` | JSON detections `[{class, conf, bbox, bbox_cx}]` |
| `/yolo/world_pos` | `geometry_msgs/PointStamped` | YOLO detection world position with lateral offset |
| `/aruco/id` | `std_msgs/Int32` | Detected ArUco ID |
| `/aruco/angle` | `std_msgs/Float32` | Horizontal angle to marker [rad] |
| `/aruco/distance` | `std_msgs/Float32` | Distance to marker [m] |
| `/aruco/markers` | `visualization_msgs/MarkerArray` | Landmark spheres for RViz |

---

## 🚀 Launch Files

| Launch file | Package | What it starts |
|---|---|---|
| `manchester.launch.py` | `iolair` | **Full autonomous stack**: Map server + ArUco detector + QR detector + QR zone checker + Odometry (EKF) + ArUco localizer + MCL + SLAM + ArUco map publisher + A\* planner + GoToGoal + **VFH+** + QR aligner FSM + Controller + RViz goal bridge |
| `qr_align_tryout.launch.py` | `iolair` | **QR alignment integration test**: Map server + ArUco detector + QR detector + QR zone checker + Odometry (EKF) + ArUco localizer + MCL + SLAM + A\* planner + GoToGoal + **VFH+** + QR aligner FSM + Controller + RViz goal bridge. Accepts `zone:=rack\|conveyor` argument. |
| `odometry.launch.py` | `iolair` | Legacy: Odometry + Controller only |

> **Primary entry point for competition use:** `manchester.launch.py`
>
> The `mission_manager` and `truck_aligener` are launched separately (or added to `manchester.launch.py`) — they are not included in the default launch to allow independent testing.
>
> The included `rviz_bonito.rviz` config provides a ready-to-use RViz layout for the full stack (map, TF, LiDAR, ArUco markers, MCL particles).

---

## 📋 Requirements

- **ROS 2 Humble** on Ubuntu 22.04
- **Python 3.8+** with `numpy`, `scipy`, `pyyaml`, `sounddevice`
- **OpenCV** with fisheye + ArUco module (`opencv-contrib-python`)
- **Ultralytics YOLOv8** (`pip install ultralytics`)
- **nav2_map_server** + **nav2_lifecycle_manager** (only needed for `manchester.launch.py`)
- **Physical Puzzlebot** hardware: RPLiDAR, fisheye camera, Tang Nano 20K FPGA lift controller
- **Jetson Nano** for SPI → FPGA communication (`jetson/spi_servo_node.py`)
- Camera calibration files: `src/Vision/Vision/fisheye_params.npz` and `fisheye_params.json` (both included)
- Custom YOLOv8 weights: not included in repo — place `best.pt` under `src/Vision/Vision/weights/`
- **Gowin EDA** (or compatible toolchain) to synthesize `Tang20K/slave_lift.v` if reflashing the FPGA

---

## 🛠️ Installation

### 1. Set up ROS 2 workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Cook131/mcrxe80-final-challenge.git
cd ~/ros2_ws
```

### 2. Install dependencies

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
pip install ultralytics opencv-contrib-python sounddevice --break-system-packages
```

### 3. Build

```bash
colcon build --packages-select iolair Navigation Vision LocalizationMapping voice_hmm_ros
source install/setup.bash
```

---

## 🎮 Usage

### Full autonomous navigation (`manchester.launch.py`)

Primary competition launch file. Starts the complete perception, localization, planning, avoidance, and control stack.

```bash
ros2 launch iolair manchester.launch.py

# Override the map file
ros2 launch iolair manchester.launch.py map_yaml:=/path/to/your_map.yaml

# Load the included RViz config for visualization
rviz2 -d rviz_bonito.rviz
```

To navigate:
- **RViz click**: activate *Publish Point* and click on the map — `rviz_goal_bridge` forwards it to A\*.
- **CLI**: publish a goal manually:

```bash
ros2 topic pub --once /astar/goal geometry_msgs/Pose2D "{x: 1.5, y: 0.0, theta: 0.0}"
```

### Autonomous pallet mission (mission_manager)

```bash
# Start the full stack first, then in a second terminal:
ros2 run Navigation mission_manager

# Send mode commands via HMI topic
ros2 topic pub --once /mission/mode std_msgs/String "data: auto"
ros2 topic pub --once /mission/mode std_msgs/String "data: stop"

# Emergency stop (bypasses the FSM tick loop)
ros2 topic pub --once /emergency_stop std_msgs/Bool "data: true"
```

### QR alignment integration test

```bash
ros2 launch iolair qr_align_tryout.launch.py zone:=rack

# Manual trigger from another terminal
ros2 topic pub --once /collect/trigger std_msgs/String "data: rack"
```

### Voice control

```bash
# List available audio devices first
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Run the voice action node with the correct device index
ros2 run voice_hmm_ros voice_action_node --ros-args --device 1

# Recognized commands (10 total, trained with Baum-Welch HMM + VQ):
# avanza / atras / izquierda / derecha / gira / detente
# toma   → lift to n1, then auto-hold
# arriba → lift to n2, then auto-hold
# suelta / abajo → lower lift
```

### Lift manual control (Jetson)

```bash
# Run on the Jetson Nano (requires spidev)
python3 jetson/spi_servo_node.py --test status
python3 jetson/spi_servo_node.py --test cycle      # full n1→hold→down→n2→hold→down cycle
python3 jetson/spi_servo_node.py --test manual     # interactive 1/-1/0 control

# Or as a ROS 2 node
ros2 run iolair spi_servo   # if registered in setup.py
```

### SLAM mapping

```bash
ros2 launch iolair manchester.launch.py
# Drive manually in a second terminal:
ros2 run iolair teleop
# Map auto-saves to maps/SLAM_map when slam_node shuts down
```

### Connecting to the Puzzlebot

```bash
# SSH into the Puzzlebot onboard computer
ssh 10.42.0.1   # password: Puzzlebot72

# On the robot — start LiDAR and micro-ROS agent (separate terminals)
ros2 launch sllidar_ros2 sllidar_a1_launch.py serial_port:=/dev/ttyUSB1
ros2 launch puzzlebot_ros micro_ros_agent.launch.py

# Start camera (from iolair_ws on the robot)
python3 camera.py
```

### Camera fisheye calibration

```bash
python3 "Camera Calibration/save_imgs.py"   # capture checkerboard frames
python3 "Camera Calibration/calibrar.py"    # compute calibration → fisheye_params.npz
```

### Individual nodes

```bash
# iolair
ros2 run iolair controller
ros2 run iolair odometry
ros2 run iolair go_to_goal
ros2 run iolair teleop

# Navigation
ros2 run Navigation astar_planner
ros2 run Navigation mission_manager
ros2 run Navigation mission_planner
ros2 run Navigation vfh_plus
ros2 run Navigation rviz_goal_bridge
ros2 run Navigation qr_align_node

# Vision
ros2 run Vision aruco_detector
ros2 run Vision yolo_vision
ros2 run Vision qr_detector
ros2 run Vision qr_align_node
ros2 run Vision qr_zone_checker
ros2 run Vision yolo_world_pos

# LocalizationMapping
ros2 run LocalizationMapping slam
ros2 run LocalizationMapping mcl
ros2 run LocalizationMapping aruco_localizer
ros2 run LocalizationMapping aruco_map_publisher
ros2 run LocalizationMapping mapping_node

# Voice
ros2 run voice_hmm_ros voice_action_node
```

---

## 📄 License

Developed as part of the TE3003B course curriculum at Tecnológico de Monterrey. See individual files for licensing details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,2,3&height=100&section=footer" width="100%"/>

*TE3003B Final Project — Robotics and Intelligent Systems Implementation*

</div>
