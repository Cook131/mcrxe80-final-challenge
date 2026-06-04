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

**TE3003B Final Project** is a complete ROS2-based autonomy stack for a **Puzzlebot** (Manchester Robotics) differential-drive robot. The system implements a full perception-localization-planning-control pipeline capable of operating in known environments, correcting odometry drift via ArUco landmark anchoring and MCL, and navigating autonomously to a sequence of waypoints.

> **Objective:** Deliver a production-quality autonomous navigation stack on real Puzzlebot hardware, integrating EKF odometry, SLAM, MCL, ArUco landmark anchoring, YOLO object detection, A\* path planning, and a reactive safety layer — all running on ROS 2 Humble without Nav2 at runtime.

---

## 🧠 Architecture

The system is organized around a layered architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION                                  │
│  aruco_detector ──► /aruco/*   │  yolo_vision ──► /yolo/*          │
│  (dual-dict 4X4_50 + 6X6_50)   │  (YOLOv8, custom weights)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                       LOCALIZATION / MAPPING                        │
│  slam_node  ──► /slam_map + /icp/pose  (ICP scan matching + grid)  │
│  mcl_node   ──► /mcl/pose              (SIR particle filter)        │
│  aruco_localizer ──► /aruco/pose       (landmark anchoring)         │
│                   ▼                                                  │
│  puzzlebotOdometry ──► /odom + TF odom→base_link                    │
│  (EKF: dead-reckoning + MCL/ICP/ArUco fusion, source switching)     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                          PLANNING                                   │
│  astar_planner ──► /astar/path + /goal                              │
│  (dynamic map source, obstacle inflation, line-of-sight shortcut)   │
│  mission_planner ──► /astar/goal   (YAML waypoint sequencer)        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                          CONTROL                                    │
│  puzzlebotGoToGoal ──► /cmd_raw  (dual PID v+ω, align-and-drive)   │
│  bug_IBA ──► /cmd_vel            (safety reflex: BUG2 wall-follow)  │
│  puzzlebotController ──► /VelocitySetR, /VelocitySetL               │
│  (dual PID wheel velocity, inverse kinematics, anti-windup)         │
└─────────────────────────────────────────────────────────────────────┘
```

### EKF Source Priority

The odometry node implements an automatic source-switching state machine:

| Priority | Source | State |
|---|---|---|
| 1 (highest) | `/mcl/pose` | `MCL_ACTIVE` / `MCL_PRIORITY` |
| 2 | `/aruco/pose` | `ARUCO_ACTIVE` / `ARUCO_PRIORITY` |
| 3 | `/icp/pose` | `ICP_ACTIVE` |
| 4 (fallback) | Dead-reckoning only | `PREDICT_ONLY` |

---

## 📁 Project Structure

```
mcrxe80-final-challenge/
├── calibrar.py                      # Camera calibration script (checkerboard)
├── calib_imgs/                      # Calibration images (30 frames)
│
└── src/
    ├── iolair/                        # Main ROS 2 package
    │   ├── iolair/
    │   │   ├── puzzlebotController.py   # PID wheel velocity controller
    │   │   ├── puzzlebotGoToGoal.py     # Dual-PID goal navigation
    │   │   ├── puzzlebotOdometry.py     # EKF odometry (MCL/ICP/ArUco fusion)
    │   │   ├── puzzlebotTeleop.py       # Keyboard teleoperation
    │   │   ├── aruco_localizer.py       # ArUco landmark anchoring
    │   │   ├── aruco_map_publisher.py   # ArUco landmark RViz visualizer
    │   │   ├── astar_planner.py         # A* path planner
    │   │   ├── bug_IBA.py               # Reactive safety layer (BUG2)
    │   │   ├── goal_bridge.py           # RViz /clicked_point → /astar/goal bridge
    │   │   ├── mapping_node.py          # Map merger (max-occupancy accumulator)
    │   │   ├── mcl_node.py              # Monte Carlo Localization
    │   │   ├── mission_planner.py       # YAML waypoint sequencer
    │   │   └── slam_node.py             # Online SLAM (ICP + occupancy grid)
    │   ├── launch/
    │   │   ├── manchester.launch.py     # Full autonomous stack (primary entry point)
    │   │   ├── astar_slam_bug.launch.py # A* + SLAM + BUG2 (no ArUco/MCL)
    │   │   └── odometry.launch.py       # Odometry + Controller only (legacy)
    │   ├── maps/
    │   │   ├── SLAM_map.pgm             # Saved occupancy map
    │   │   └── SLAM_map.yaml            # Map metadata
    │   ├── configs/
    │   │   ├── aruco_landmarks.yaml     # Global ArUco landmark positions
    │   │   └── exploration_waypoints.yaml # Mission waypoints for mission_planner
    │   └── setup.py
    │
    └── puzzlebot/                     # Perception package
        └── puzzlebot/
            ├── aruco_detector.py        # Dual-dict ArUco detector (4X4_50 + 6X6_50)
            ├── yolo_vision.py           # YOLOv8 object detector
            ├── camera_params.npz        # Camera calibration (numpy)
            ├── camera_params.json       # Camera calibration (JSON)
            └── weights/
                └── best.pt              # Custom YOLOv8 weights
```

---

## 🔧 Nodes

### `iolair` package

| Node | Executable | Description |
|---|---|---|
| **puzzlebotController** | `controller` | Closed-loop PID wheel velocity controller at 50 Hz. Implements inverse differential kinematics, per-wheel PID with anti-windup, and integral reset on direction change. |
| **puzzlebotGoToGoal** | `go_to_goal` | Dual-PID navigation to a `Pose2D` goal. Separates linear and angular PID loops. Uses align-and-drive logic: rotates to face goal before advancing. Publishes to `/cmd_raw`. |
| **puzzlebotOdometry** | `odometry` | Full EKF odometry fusing encoder dead-reckoning with up to three external pose sources (MCL, ICP, ArUco). Joseph-form covariance updates, innovation gating, and automatic source-switching with configurable timeouts. |
| **puzzlebotTeleop** | `teleop` | Keyboard teleoperation with velocity ramping at 50 Hz. W/S/A/D keys, Space for emergency stop. Thread-safe with guaranteed terminal restoration. |
| **aruco_localizer** | `aruco_localizer` | Drift correction by landmark anchoring against `aruco_landmarks.yaml`. Unknown markers are anchored dynamically on first observation. Adaptive noise covariance scales with distance². |
| **aruco_map_publisher** | `aruco_map_publisher` | Publishes ArUco landmark positions from `aruco_landmarks.yaml` as a `MarkerArray` on `/aruco/markers` for RViz visualization. |
| **astar_planner** | `astar_planner` | A\* path planner on occupancy grids. Dynamically selects between `/slam_map` and `/map` at runtime, inflates obstacles, applies line-of-sight shortcutting, and feeds waypoints to GoToGoal. |
| **bug_IBA** | `bug_IBA` | BUG2 reactive safety layer between GoToGoal and the controller: `PASS_THROUGH` → `PREDICTIVE_BRAKE` → `REFLEX_TURN` (arc escape) → `REFLEX_STOP`. Intercepts `/cmd_raw` and publishes to `/cmd_vel`. |
| **goal_bridge** | `rviz_goal_bridge` | Converts RViz `/clicked_point` (PointStamped) into `/astar/goal` (Pose2D), enabling click-to-navigate from RViz. |
| **mapping_node** | `mapping_node` | Map merger node that accumulates SLAM snapshots using a max-occupancy strategy. Cells marked occupied are never freed. Expands automatically as the map grows. |
| **mcl_node** | `mcl` | SIR Particle Filter localization. Vectorised likelihood-field sensor model with precomputed EDT. Fuses EKF odometry (motion model) with LiDAR scans. Supports `/initialpose` from RViz. |
| **mission_planner** | `mission_planner` | Reads waypoints from `exploration_waypoints.yaml` and publishes them sequentially to `/astar/goal`. Supports looping and configurable timeout per goal. |
| **slam_node** | `slam` | Online SLAM building an occupancy grid with log-odds updates and Bresenham ray-casting. ICP scan matching corrects drift between keyframes. Publishes `/icp/pose` for EKF fusion and `/slam_map` for A\*. |

### `puzzlebot` package

| Node | Executable | Description |
|---|---|---|
| **aruco_detector** | `aruco_detector` | Dual-dictionary ArUco detector (`4X4_50` IDs 0–10 for external/internal WPs, `6X6_50` IDs 0–6 for named waypoints). Estimates 6-DOF pose via `solvePnP`. Publishes ID, distance, angle, waypoint pose, and annotated image. |
| **yolo_vision** | `yolo_vision` | YOLOv8 inference node using custom weights (`best.pt`). Publishes annotated image and JSON detections with class, confidence, and bounding box on `/yolo/detecciones`. |

---

## 📡 Key Topics

| Topic | Type | Description |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command (after BUG2 safety layer) |
| `/cmd_raw` | `geometry_msgs/Twist` | Velocity command (before safety layer, from GoToGoal) |
| `/odom` | `nav_msgs/Odometry` | EKF-fused odometry + covariance |
| `/slam_map` | `nav_msgs/OccupancyGrid` | Live SLAM map (latched) |
| `/map` | `nav_msgs/OccupancyGrid` | Static map served by nav2_map_server |
| `/mcl/pose` | `PoseWithCovarianceStamped` | MCL corrected pose |
| `/icp/pose` | `PoseWithCovarianceStamped` | ICP-corrected pose from SLAM node |
| `/aruco/pose` | `PoseWithCovarianceStamped` | ArUco landmark correction |
| `/ekf/active_source` | `std_msgs/String` | Current EKF fusion source |
| `/astar/path` | `nav_msgs/Path` | Planned path for RViz |
| `/astar/goal` | `geometry_msgs/Pose2D` | Goal input to A\* planner |
| `/goal` | `geometry_msgs/Pose2D` | A\* waypoint output to GoToGoal |
| `/aruco/markers` | `visualization_msgs/MarkerArray` | Landmark spheres for RViz |
| `/aruco/id` | `std_msgs/Int32` | Detected ArUco public ID |
| `/aruco/distance` | `std_msgs/Float32` | Distance to marker [m] |
| `/yolo/detecciones` | `std_msgs/String` | YOLO detections (JSON) |

---

## 🚀 Launch Files

| Launch file | What it starts |
|---|---|
| `manchester.launch.py` | **Full autonomous stack**: Map server + ArUco detector + Odometry (EKF) + ArUco localizer + MCL + SLAM + ArUco map publisher + A\* planner + GoToGoal + Bug IBA + Controller + RViz goal bridge |
| `astar_slam_bug.launch.py` | A\* + SLAM + BUG2 (no ArUco or MCL): static TF + Odometry + Controller + SLAM (2 s delayed) + A\* planner + RViz goal bridge + GoToGoal + Bug IBA |
| `odometry.launch.py` | Legacy: Odometry + Controller only (uses old `odometria`/`controlador` executables) |

> **Primary entry point for competition use:** `manchester.launch.py`

---

## 📋 Requirements

- **ROS 2 Humble** on Ubuntu 22.04
- **Python 3.8+** with `numpy`, `scipy`, `pyyaml`
- **OpenCV** with ArUco module (`opencv-contrib-python`)
- **Ultralytics YOLOv8** (`pip install ultralytics`)
- **nav2_map_server** + **nav2_lifecycle_manager** (only needed for `manchester.launch.py`)
- **Physical Puzzlebot** hardware with RPLiDAR and camera
- Custom **YOLOv8 weights** (`best.pt`) — included in `src/puzzlebot/weights/`

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
pip install ultralytics opencv-contrib-python --break-system-packages
```

### 3. Build

```bash
colcon build --packages-select iolair puzzlebot
source install/setup.bash
```

---

## 🎮 Usage

### Full autonomous navigation (`manchester.launch.py`)

This is the primary competition launch file. It starts the complete stack: ArUco-anchored EKF, MCL, SLAM, A\*, BUG2, and the controller.

```bash
# Provide a map (SLAM_map.yaml) in src/iolair/maps/ before launching
ros2 launch iolair manchester.launch.py

# Optional: override the map file
ros2 launch iolair manchester.launch.py map_yaml:=/path/to/your_map.yaml
```

To navigate, either:
- **RViz click**: activate the *Publish Point* tool and click on the map — `rviz_goal_bridge` forwards it to the A\* planner.
- **CLI**: publish a goal manually:

```bash
ros2 topic pub --once /astar/goal geometry_msgs/Pose2D "{x: 1.5, y: 0.0, theta: 0.0}"
```

### Automated waypoint mission

```bash
# Edit src/iolair/configs/exploration_waypoints.yaml with your waypoints, then:
ros2 run iolair mission_planner --ros-args -p waypoints_file:=<path>/exploration_waypoints.yaml
```

### SLAM mapping

```bash
ros2 launch iolair astar_slam_bug.launch.py
# Drive manually in a second terminal:
ros2 run iolair teleop
# Map is saved automatically to src/iolair/maps/SLAM_map when the node shuts down
```

### Keyboard teleoperation

```bash
ros2 run iolair teleop
# W/S: forward/backward  |  A/D: turn  |  Space: emergency stop  |  Q: quit
```

### Camera calibration

```bash
# Collect calibration images (checkerboard), then:
python3 calibrar.py
# Outputs camera_params.json and camera_params.npz to the working directory
```

### Individual nodes

```bash
ros2 run iolair controller
ros2 run iolair odometry
ros2 run iolair go_to_goal
ros2 run iolair slam
ros2 run iolair mcl
ros2 run iolair aruco_localizer
ros2 run iolair aruco_map_publisher
ros2 run iolair bug_IBA
ros2 run iolair astar_planner
ros2 run iolair rviz_goal_bridge
ros2 run iolair mission_planner
ros2 run iolair mapping_node
ros2 run puzzlebot aruco_detector
ros2 run puzzlebot yolo_vision
```

---

## 📄 License

Developed as part of the TE3003B course curriculum at Tecnológico de Monterrey. See individual files for licensing details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,2,3&height=100&section=footer" width="100%"/>

*TE3003B Final Project — Robotics and Intelligent Systems Implementation*

</div>
