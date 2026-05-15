# Final Project for TE3003B: Robotics and Intelligent Systems Implementation

This is the official repository for the Final Project of the course TE3003B: Robotics and Intelligent Systems Implementation.

## Project Overview

The project consists of a ROS2 package designed to control a modified Puzzlebot robot operating in a real industrial warehouse arena. The system implements the functionality of an E80 LGV (Laser Guided Vehicle) forklift robot, enabling autonomous navigation, mapping, and task execution with a physical robot.

## Repository Structure

- `src/iolair/`: Main ROS2 package containing:
  - `iolair/`: Python modules for robot control, including nodes for mapping, localization, odometry, teleoperation, and SLAM.
  - `launch/`: Launch files for real-world deployment.
  - `maps/`: Pre-built maps for the warehouse arena.
  - `rviz/`: RViz configuration files for visualization.
  - `resource/`: Additional resources.
  - `test/`: Test files for code quality.

Note: Simulation-related files (Gazebo worlds, plugins, and models) are not included in the final main branch, as the implementation focuses on real robot deployment.

## Features

- Autonomous navigation using SLAM (Simultaneous Localization and Mapping)
- Odometry and localization nodes
- Teleoperation capabilities
- Map publishing and visualization with RViz
- Voice Commands Recognition
- HMI in order to visulize the robot oddometry
- Aruco and QR reference detections

## Requirements

- ROS2 (Humble or compatible version)
- Python dependencies as listed in `setup.py`
- Physical Puzzlebot robot hardware

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd mcrxe80-final-challenge
   ```

2. Build the ROS2 package:
   ```
   colcon build
   source install/setup.bash
   ```

## Usage

Ensure the physical Puzzlebot robot is connected and powered on.

- Launch SLAM for mapping:
  ```
  ros2 launch iolair slamSim.launch.py  # Note: This launch file is configured for real robot SLAM
  ```

- Run individual nodes as needed for specific functionalities (e.g., teleoperation, odometry).

## Contributing

This is the final project repository. For any issues or contributions, please contact the project maintainers.

## License

[Specify license if applicable]