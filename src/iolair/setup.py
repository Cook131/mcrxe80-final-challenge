from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'iolair'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ESTA LÍNEA ES LA NUEVA:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='serch',
    maintainer_email='sergio.muhi@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controlador = iolair.PuzzlebotController:main',
            'odometria = iolair.puzzlebotOdometry:main',
            'go_to_goal = iolair.puzzlebotGoToGoal:main',
        ],
    },
)