from setuptools import find_packages, setup

package_name = 'puzzlebot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/python3.10/site-packages/puzzlebot', [   # ← aquí el cambio
            'puzzlebot/camera_params.npz',
            'puzzlebot/camera_params.json',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='serch',
    maintainer_email='serch@todo.todo',
    description='Camera and Aruco tracking nodes for Puzzlebot on Jetson Nano',
    license='Apache License 2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # The format is: 'executable_name = package_name.file_name:main'
            'aruco_detector = puzzlebot.aruco_detector:main',
            'yolo_vision = puzzlebot.yolo_vision:main',
        ],
    },
)
