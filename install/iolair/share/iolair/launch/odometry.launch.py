from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
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
        # Nodo de Navegación Go-to-Goal
        Node(
            package='iolair',
            executable='go_to_goal',
            name='puzzlebot_navigation'
        )
    ])