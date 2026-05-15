"""Launch file for teleoperation setup."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for teleop setup."""
    return LaunchDescription([
        Node(
            package='iolair',
            executable='odometria',
            name='puzzlebot_odom'
        ),
        Node(
            package='iolair',
            executable='controlador',
            name='puzzlebot_controller'
        ),
        # Teleop removed — must run in its own terminal
    ])
