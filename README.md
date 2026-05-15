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

**TE3003B Final Project** is a ROS2-based system for controlling a modified **Puzzlebot robot** with core functionalities for navigation and control. The implementation provides essential scripts for robot operation, including controller logic, goal navigation, odometry processing, and teleoperation capabilities.

The system is designed for deployment on physical Puzzlebot hardware, focusing on fundamental robotic control and navigation tasks.

> **Objective:** Implement basic robotic control systems for autonomous and manual operation of a mobile robot platform.

---

## 📁 Project Structure

```
src/iolair/
│
├── iolair/                 # Core ROS2 package
│   ├── puzzlebotController.py    # Main robot controller
│   ├── puzzlebotGoToGoal.py      # Goal navigation
│   ├── puzzlebotOdometry.py      # Odometry processing
│   └── puzzlebotTeleop.py        # Teleoperation control
│
├── launch/                 # Launch configurations
│   └── teleop.launch.py    # Teleoperation launch
│
└── setup.py                # Package setup script
```

| Directory | Purpose |
|---|---|
| `iolair/` | Python scripts for robot control and navigation |
| `launch/` | ROS2 launch file for teleoperation |
| `setup.py` | Python package configuration |

---

## ✨ Features

- 🤖 **Robot Control** with main controller logic
- 🎯 **Goal Navigation** for autonomous movement to targets
- 📍 **Odometry Processing** for position tracking
- 🎮 **Teleoperation** for manual control
- 🚀 **Launch Configuration** for easy system startup

---

## 🔧 Scripts

The package includes several Python scripts for robot operation:

| Script | Description |
|---|---|
| **puzzlebotController.py** | Main control logic for robot operation |
| **puzzlebotGoToGoal.py** | Implements navigation to specified goals |
| **puzzlebotOdometry.py** | Handles odometry calculations and publishing |
| **puzzlebotTeleop.py** | Provides teleoperation capabilities |

### Launch Files

- **teleop.launch.py**: Launches the teleoperation system

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

2. **Launch teleoperation:**
   ```bash
   ros2 launch iolair teleop.launch.py
   ```

### Individual Scripts

Run specific scripts for testing:

```bash
# Robot controller
ros2 run iolair puzzlebotController

# Goal navigation
ros2 run iolair puzzlebotGoToGoal

# Odometry
ros2 run iolair puzzlebotOdometry

# Teleoperation
ros2 run iolair puzzlebotTeleop
```

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