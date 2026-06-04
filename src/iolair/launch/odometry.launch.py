"""Launch file for odometry node."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for odometry setup."""
    return LaunchDescription([
        # Nodo de Odometría
        Node(
            package='iolair',
            executable='odometria',
            name='puzzlebot_odom'
        ),
        # Nodo de Control Cinemático
        Node(
            package='iolair',
            executable='controlador',
            name='puzzlebot_controller'
        ),
    ])
