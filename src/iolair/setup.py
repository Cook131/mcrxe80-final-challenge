import os
import glob
from setuptools import find_packages, setup

package_name = 'iolair'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required by ament so ROS 2 can find the package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Launch files — every .py inside launch/ gets installed
        (os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.py')),
        # Maps — every .yaml inside maps/ gets installed
        (os.path.join('share', package_name, 'maps'),
            glob.glob('maps/*')),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='serch',
    maintainer_email='sergio.muhi@hotmail.com',
    description='Puzzlebot real-robot SLAM (no nav2)',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # format:  'executable_name = package.module:main_function'
            'odometry   = iolair.puzzlebotOdometry:main',
            'controller = iolair.puzzlebotController:main',
            'teleop     = iolair.puzzlebotTeleop:main',
            'slam       = iolair.slam_node:main',
            'go_to_goal  = iolair.puzzlebotGoToGoal:main',
            'mcl        = iolair.mcl_node:main',
            'aruco_localizer       = iolair.aruco_localizer:main',
            'bug_IBA       = iolair.bug_IBA:main',
            'astar_planner   = iolair.astar_planner:main',
        ],
    },
)