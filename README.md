# 🚀 Final Project for TE3003B: Robotics and Intelligent Systems Implementation

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)

This is the **official repository** for the Final Project of the course **TE3003B: Robotics and Intelligent Systems Implementation**.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

The project consists of a **ROS2 package** designed to control a modified **Puzzlebot robot** operating in a real industrial warehouse arena. The system implements the functionality of an **E80 LGV (Laser Guided Vehicle) forklift robot**, enabling autonomous navigation, mapping, and task execution with a physical robot.

## 📁 Repository Structure

```
src/iolair/
├── iolair/                 # Python modules for robot control
│   ├── map_publisher_node.py
│   ├── mcl_node.py
│   ├── odommetry_node.py
│   ├── puzzlebot_teleop.py
│   ├── robot_controller.py
│   └── slam_node.py
├── launch/                 # Launch files for real-world deployment
├── maps/                   # Pre-built maps for the warehouse arena
├── rviz/                   # RViz configuration files
├── resource/               # Additional resources
└── test/                   # Test files for code quality
```

**Note:** Simulation-related files (Gazebo worlds, plugins, and models) are not included in the final main branch, as the implementation focuses on **real robot deployment**.

## ✨ Features

- 🤖 **Autonomous navigation** using SLAM (Simultaneous Localization and Mapping)
- 📍 **Odometry and localization** nodes
- 🎮 **Teleoperation** capabilities
- 🗺️ **Map publishing** and visualization with RViz
- 🎤 **Voice Commands Recognition**
- 🖥️ **HMI** to visualize the robot odometry
- 🔍 **Aruco and QR reference detections**

## 📋 Requirements

- 🐧 **ROS2** (Humble or compatible version)
- 🐍 **Python dependencies** as listed in `setup.py`
- 🤖 **Physical Puzzlebot robot hardware**

## 🛠️ Installation

1. **Clone** this repository:
   ```bash
   git clone <repository-url>
   cd mcrxe80-final-challenge
   ```

2. **Build** the ROS2 package:
   ```bash
   colcon build
   source install/setup.bash
   ```

## 🚀 Usage

Ensure the physical **Puzzlebot robot** is connected and powered on.

- **Launch SLAM** for mapping:
  ```bash
  ros2 launch iolair slamSim.launch.py  # Configured for real robot SLAM
  ```

- **Run individual nodes** as needed for specific functionalities (e.g., teleoperation, odometry).

## 🤝 Contributing

This is the **final project repository**. For any issues or contributions, please contact the project maintainers.

## 📄 License

[Specify license if applicable]