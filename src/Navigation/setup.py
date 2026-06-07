import os
import glob
from setuptools import find_packages, setup

package_name = 'Navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required by ament so ROS 2 can find the package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Maps — every .yaml inside maps/ gets installed
        (os.path.join('share', package_name, 'maps'),
            glob.glob('maps/*')),
        (os.path.join('share', package_name, 'configs'),
            glob.glob('configs/*')),  
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

            'astar_planner    = Navigation.astar_planner:main',
            'rviz_goal_bridge = Navigation.goal_bridge:main',
            'mission_planner  = Navigation.mission_planner:main',
            'mission_manager  = Navigation.mission_manager:main',
            'vfh_plus         = Navigation.vfh_plus:main',
            'qr_aligner       = Navigation.qr_aligner_node:main',
        ],
    },
)