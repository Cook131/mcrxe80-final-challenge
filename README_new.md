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
  <a href="#-project-structure">Structure</a> •
  <a href="#-features">Features</a> •
  <a href="#-nodes">Nodes</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 🏭 Overview

**TE3003B Final Project** is a comprehensive ROS2-based system for controlling a modified **Puzzlebot robot** in a real industrial warehouse environment. The implementation simulates the functionality of an **E80 LGV (Laser Guided Vehicle) forklift robot**, providing autonomous navigation, mapping, and task execution capabilities.

The system integrates multiple ROS2 nodes for perception, planning, and control, designed for deployment on physical hardware without simulation dependencies.

> **Objective:** Develop an intelligent robotic system capable of operating in warehouse settings, handling navigation challenges, and executing tasks autonomously using real-world sensors and actuators.

---

## 📁 Project Structure

```
src/iolair/
│
├── iolair/                 # Core ROS2 package
│   ├── __init__.py
│   ├── map_publisher_node.py    # Map publishing
│   ├── mcl_node.py              # Monte Carlo Localization
│   ├── odommetry_node.py        # Odometry processing
│   ├── puzzlebot_teleop.py      # Teleoperation control
│   ├── robot_controller.py      # Main robot controller
│   └── slam_node.py             # SLAM implementation
│
├── launch/                 # Launch configurations
│   ├── sim.launch.py       # Real-world deployment launch
│   └── slamSim.launch.py   # SLAM launch
│
├── maps/                   # Warehouse maps
│   └── puzzlebot_map.pgm
│
├── rviz/                   # Visualization configs
│   └── puzzlebot.rviz
│
├── resource/               # Additional resources
├── test/                   # Code quality tests
└── package.xml             # ROS2 package manifest
```

| Directory | Purpose |
|---|---|
| `iolair/` | Python nodes for robot functionality |
| `launch/` | ROS2 launch files for system startup |
| `maps/` | Pre-built warehouse environment maps |
| `rviz/` | RViz configuration for visualization |
| `test/` | Automated testing and code quality checks |

---

## ✨ Features

- 🤖 **Autonomous Navigation** using SLAM and path planning
- 📍 **Real-time Localization** with Monte Carlo methods
- 🎮 **Manual Teleoperation** for direct control
- 🗺️ **Map Generation** and publishing
- 🔊 **Voice Command Recognition** for human-robot interaction
- 🖥️ **Human-Machine Interface** for odometry visualization
- 🔍 **Marker Detection** (ArUco and QR codes) for reference points
- 📊 **Sensor Integration** for real-world operation

---

## 🔧 Nodes

The system comprises several ROS2 nodes, each handling specific aspects of robot operation:

### Core Nodes

| Node | File | Description |
|---|---|---|
| **Map Publisher** | `map_publisher_node.py` | Publishes static and dynamic maps of the environment |
| **MCL Node** | `mcl_node.py` | Implements Monte Carlo Localization for pose estimation |
| **Odometry Node** | `odommetry_node.py` | Processes wheel encoder data for odometry |
| **Teleop Node** | `puzzlebot_teleop.py` | Handles manual control inputs |
| **Controller** | `robot_controller.py` | Main control logic and task execution |
| **SLAM Node** | `slam_node.py` | Simultaneous Localization and Mapping |

### Node Interfaces

- **Topics:** Standard ROS2 topics for sensor data, commands, and state
- **Services:** Configuration and status services
- **Actions:** Long-running tasks like navigation goals

---

## 📋 Requirements

- **ROS2 Humble** or compatible distribution
- **Python 3.8+** with ROS2 dependencies
- **Physical Puzzlebot Hardware** with sensors and actuators
- **Ubuntu 22.04** or compatible Linux distribution

---

## 🛠️ Installation

### Prerequisites

1. Install ROS2 Humble:
   ```bash
   # Follow official ROS2 installation guide
   # https://docs.ros.org/en/humble/Installation.html
   ```

2. Set up ROS2 workspace:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

### Package Installation

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd mcrxe80-final-challenge
   ```

2. **Build package:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select iolair
   source install/setup.bash
   ```

3. **Install dependencies:**
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

---

## 🚀 Usage

### System Startup

1. **Source environment:**
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

2. **Launch core system:**
   ```bash
   ros2 launch iolair sim.launch.py
   ```

3. **Launch with SLAM:**
   ```bash
   ros2 launch iolair slamSim.launch.py
   ```

### Individual Nodes

Run specific nodes for testing:

```bash
# Teleoperation
ros2 run iolair puzzlebot_teleop

# Map publishing
ros2 run iolair map_publisher_node

# Localization
ros2 run iolair mcl_node
```

### Visualization

Launch RViz for monitoring:
```bash
rviz2 -d src/iolair/rviz/puzzlebot.rviz
```

---

## 🔄 System Architecture

```
Physical Puzzlebot
        │
        ▼
┌─────────────────────────────────────┐
│         Sensor Inputs                │
│  • LIDAR                            │
│  • Wheel Encoders                   │
│  • Camera (ArUco/QR)                │
│  • Microphone (Voice)               │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         ROS2 Nodes                  │
│  ┌─────────────────────────────────┐ │
│  │ Odometry Node ← Encoders        │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │ SLAM Node ← LIDAR + Odometry    │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │ MCL Node ← Map + LIDAR          │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │ Controller ← Localization       │ │
│  └─────────────────────────────────┘ │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Actuator Outputs            │
│  • Motor Commands                   │
│  • Status Feedback                  │
└─────────────────────────────────────┘
```

---

## 🤝 Contributing

This is the **final project repository** for TE3003B. For academic purposes:

- Code reviews and improvements are welcome
- Report issues via GitHub issues
- Contact project maintainers for collaboration

## 📄 License

This project is developed as part of the TE3003B course curriculum. See individual files for licensing information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,2,3&height=100&section=footer" width="100%"/>

*TE3003B Final Project — Robotics and Intelligent Systems Implementation*

</div>