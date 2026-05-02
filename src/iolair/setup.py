import os
import glob
from setuptools import find_packages, setup

package_name = 'iolair'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.py')),

        # World files
        (os.path.join('share', package_name, 'worlds'),
            glob.glob('worlds/*')),

        # Map files
        (os.path.join('share', package_name, 'maps'),
            glob.glob('maps/*')),

        # RViz config
        (os.path.join('share', package_name, 'rviz'),
            glob.glob('rviz/*.rviz')),

        # Gazebo model
        (os.path.join('share', package_name, 'gazebo', 'puzzlebot'),
            glob.glob('gazebo/puzzlebot/model.*')),

        (os.path.join('share', package_name, 'gazebo', 'puzzlebot', 'meshes'),
            glob.glob('gazebo/puzzlebot/meshes/*')),

        # Gazebo plugin
        (os.path.join('share', package_name, 'gazebo', 'plugins'),
            glob.glob('gazebo/plugins/*.so')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='serch',
    maintainer_email='sergio.muhi@hotmail.com',
    description='Puzzlebot Gazebo simulation with MCL localization',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'odometry      = iolair.odommetry_node:main',
            'controller    = iolair.robot_controller:main',
            'mcl           = iolair.mcl_node:main',
            'map_publisher = iolair.map_publisher_node:main',
            'slam          = iolair.slam_node:main',
            'teleop        = iolair.puzzlebot_teleop:main',
        ],
    },
)