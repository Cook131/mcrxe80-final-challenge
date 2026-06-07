from setuptools import find_packages, setup

package_name = 'Vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/python3.10/site-packages/Vision', [
            'Vision/fisheye_params.npz',
            'Vision/fisheye_params.json',
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
            'aruco_detector = Vision.aruco_detector:main',
            'yolo_vision = Vision.yolo_vision:main',
            'qr_zone_checker = Vision.qr_zone_checker:main',
            'qr_detector = Vision.qr_detector:main',
            'yolo_world_pos = Vision.truck_pos:main',
        ],
    },
)