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
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
</p>

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-nodes">Nodes</a> •
  <a href="#-launch-files">Launch Files</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 🏭 Overview

**TE3003B Final Project** is a complete ROS2-based autonomy stack for a **Puzzlebot** (Manchester Robotics) differential-drive robot. The system goes far beyond basic teleoperation: it implements a full perception-localization-planning-control pipeline capable of operating in unknown environments and correcting odometry drift using multiple sensor fusion sources.

> **Objective:** Deliver a production-quality autonomous navigation stack on real Puzzlebot hardware, integrating EKF odometry, SLAM, MCL, ArUco landmark anchoring, YOLO object detection, A\* path planning, and a reactive safety layer — all running on ROS 2 Humble without Nav2 at runtime.

---

## 🧠 Architecture

The system is organized around a layered architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PERCEPTION                                  │
│  aruco_detector ──► /aruco/*   │  yolo_vision ──► /yolo/*          │
│  (dual-dict 4X4+6X6 + QR)      │  (YOLOv8, custom weights)         │
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
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                          CONTROL                                    │
│  puzzlebotGoToGoal ──► /cmd_raw  (dual PID v+ω, align-and-drive)   │
│  bug_IBA ──► /cmd_vel            (safety reflex: brake → turn → stop)│
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
src/
├── iolair/                        # Main ROS 2 package
│   ├── iolair/
│   │   ├── puzzlebotController.py   # PID wheel velocity controller
│   │   ├── puzzlebotGoToGoal.py     # Dual-PID goal navigation
│   │   ├── puzzlebotOdometry.py     # EKF odometry (MCL/ICP/ArUco fusion)
│   │   ├── puzzlebotTeleop.py       # Keyboard teleoperation
│   │   ├── aruco_localizer.py       # ArUco landmark anchoring
│   │   ├── astar_planner.py         # A* path planner
│   │   ├── bug_IBA.py               # Reactive safety layer
│   │   ├── mcl_node.py              # Monte Carlo Localization
│   │   └── slam_node.py             # Online SLAM (ICP + occupancy grid)
│   ├── launch/
│   │   ├── teleop.launch.py         # Teleoperation
│   │   ├── odometry.launch.py       # Odometry + GoToGoal
│   │   ├── slam.launch.py           # Full SLAM stack
│   │   ├── mcl.launch.py            # MCL localization on saved map
│   │   └── aruco_localizer.launch.py # ArUco-based localization
│   ├── maps/
│   │   ├── slam_map.pgm             # Saved occupancy map
│   │   └── slam_map.yaml            # Map metadata
│   └── setup.py
│
└── puzzlebot/                     # Perception package
    └── puzzlebot/
        ├── aruco_detector.py        # Dual-dict ArUco + QR detector
        ├── yolo_vision.py           # YOLOv8 object detector
        └── camera_params.*          # Camera calibration files
```

---

## 🔧 Nodes

### `iolair` package

| Node | Executable | Description |
|---|---|---|
| **puzzlebotController** | `controller` | Closed-loop PID wheel velocity controller. Runs at 50 Hz. Implements inverse differential kinematics, per-wheel PID with anti-windup, and integral reset on direction change. Tunable via `--ros-args -p Kp:=X`. |
| **puzzlebotGoToGoal** | `go_to_goal` | Dual-PID navigation to a `Pose2D` goal. Separates linear (aggressive) and angular (smooth) PID loops. Uses align-and-drive logic: rotates to face goal before advancing. |
| **puzzlebotOdometry** | `odometry` | Full EKF odometry fusing encoder dead-reckoning with up to three external pose sources (MCL, ICP, ArUco). Implements Joseph-form covariance updates, innovation gating, and automatic source-switching with configurable timeouts. |
| **puzzlebotTeleop** | `teleop` | Keyboard teleoperation with velocity ramping at 50 Hz. W/S/A/D keys, space bar for emergency stop. Thread-safe design with guaranteed terminal restoration. |
| **aruco_localizer** | `aruco_localizer` | Drift correction by landmark anchoring. First observation anchors a marker's global position; subsequent observations compute a correction vector and publish it to the EKF. Uses adaptive noise covariance (distance-proportional). |
| **astar_planner** | `astar` | A\* path planner on occupancy grids. Dynamically selects between `/slam_map` and `/map` at runtime, inflates obstacles, applies line-of-sight shortcutting, and feeds waypoints to GoToGoal. |
| **bug_IBA** | `bug_reflex` | Three-zone reactive safety layer between GoToGoal and the controller: `PASS_THROUGH` → `PREDICTIVE_BRAKE` → `REFLEX_TURN` (arc escape) → `REFLEX_STOP`. Hysteresis prevents oscillation. |
| **mcl_node** | `mcl` | SIR Particle Filter localization. Fuses EKF odometry (motion model) with LiDAR scans (beam-range-finder sensor model + ray-casting). Supports RViz pose initialization via `/initialpose`. |
| **slam_node** | `slam` | Online SLAM building an occupancy grid with log-odds updates and Bresenham ray-casting. ICP scan matching corrects drift between keyframes. Publishes `/icp/pose` for EKF fusion. Saves maps via `/slam/save_map` service. |

### `puzzlebot` package

| Node | Executable | Description |
|---|---|---|
| **aruco_detector** | — | Dual-dictionary ArUco detector (4×4\_50 + 6×6\_50) plus QR code detection. Estimates 6-DOF pose via `solvePnP`. Publishes ID, label, distance, angle, waypoint pose, and annotated image. Camera-calibration auto-detected from `.npz` or `.json`. |
| **yolo_vision** | — | YOLOv8 inference node using custom weights (`best.pt`). Publishes annotated image and JSON detections with class, confidence, and bounding box. |

---

## 📡 Key Topics

| Topic | Type | Description |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command (after safety layer) |
| `/cmd_raw` | `geometry_msgs/Twist` | Velocity command (before safety layer) |
| `/odom` | `nav_msgs/Odometry` | EKF-fused odometry + covariance |
| `/slam_map` | `nav_msgs/OccupancyGrid` | Live SLAM map (latched) |
| `/mcl/pose` | `PoseWithCovarianceStamped` | MCL corrected pose |
| `/icp/pose` | `PoseWithCovarianceStamped` | ICP-corrected pose |
| `/aruco/pose` | `PoseWithCovarianceStamped` | ArUco landmark correction |
| `/ekf/active_source` | `std_msgs/String` | Current EKF fusion source |
| `/astar/path` | `nav_msgs/Path` | Planned path for RViz |
| `/reflex_status` | `std_msgs/String` | Safety layer mode |
| `/aruco/id` | `std_msgs/Int32` | Detected ArUco pub ID |
| `/aruco/distance` | `std_msgs/Float32` | Distance to marker [m] |
| `/yolo/detecciones` | `std_msgs/String` | YOLO detections (JSON) |

---

## 🚀 Launch Files

| Launch file | What it starts |
|---|---|
| `teleop.launch.py` | Odometry + Controller (teleop must run in a separate terminal) |
| `odometry.launch.py` | Odometry + Controller + GoToGoal |
| `slam.launch.py` | Static TF + Odometry + Controller + SLAM node (2s delayed) |
| `mcl.launch.py` | Map server + Lifecycle manager + Odometry + MCL + Controller |
| `aruco_localizer.launch.py` | Odometry + ArUco Localizer + Map server + Controller |

---

## 📋 Requirements

- **ROS 2 Humble** on Ubuntu 22.04
- **Python 3.8+** with `numpy`, `scipy`
- **OpenCV** with ArUco module (`opencv-contrib-python`)
- **Ultralytics YOLOv8** (`pip install ultralytics`)
- **Physical Puzzlebot** hardware with RPLiDAR and camera
- Custom **YOLOv8 weights** (`best.pt`) — available via the [Releases](https://github.com/Cook131/mcrxe80-final-challenge/releases) tab

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

### 4. Place YOLOv8 weights

Download `best.pt` from the [Releases](https://github.com/Cook131/mcrxe80-final-challenge/releases/tag/new) tab and update the path in `yolo_vision.py`:

```python
WEIGHTS = "/path/to/best.pt"
```

---

## 🎮 Usage

### Teleoperation (manual driving)

```bash
# Terminal 1 — core nodes
ros2 launch iolair teleop.launch.py

# Terminal 2 — keyboard control
ros2 run iolair teleop
# W/S: forward/backward  |  A/D: turn  |  Space: emergency stop  |  Q: quit
```

### SLAM (mapping an unknown environment)

```bash
ros2 launch iolair slam.launch.py
# Drive with teleop in a second terminal
# When done, save the map:
ros2 service call /slam/save_map std_srvs/srv/Trigger
```

### MCL (localization on a saved map)

```bash
# Place slam_map.pgm and slam_map.yaml in src/iolair/maps/
ros2 launch iolair mcl.launch.py
# Send a 2D Pose Estimate from RViz to initialize the particle cloud
```

### Autonomous navigation (A\* + GoToGoal)

```bash
ros2 launch iolair odometry.launch.py
# Publish a goal:
ros2 topic pub /astar/goal geometry_msgs/Pose2D "{x: 1.5, y: 0.0, theta: 0.0}"
```

### Individual nodes

```bash
ros2 run iolair controller
ros2 run iolair odometry
ros2 run iolair go_to_goal
ros2 run iolair slam
ros2 run iolair mcl
ros2 run iolair aruco_localizer
ros2 run iolair bug_reflex
ros2 run iolair astar
```

---

## 🤝 Contributing

This is the **final project repository** for TE3003B (Robotics and Intelligent Systems). For academic purposes:

- Code reviews and improvements are welcome
- Report issues via [GitHub Issues](https://github.com/Cook131/mcrxe80-final-challenge/issues)
- CI runs automatically on every push via GitHub Actions (ROS 2 Humble build + rosdep check)

## 📄 License

Developed as part of the TE3003B course curriculum at Tecnológico de Monterrey. See individual files for licensing details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,2,3&height=100&section=footer" width="100%"/>

*TE3003B Final Project — Robotics and Intelligent Systems Implementation*

</div>