from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
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