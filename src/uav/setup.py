from setuptools import find_packages, setup
import os

package_name = 'uav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), ['models/model.pt']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mustafa',
    maintainer_email='msalihberk@proton.me',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'img = uav.img_process:main',
            'move = uav.uav_movement:main',
            'cal = uav.uav_calculation:main',
            'main = uav.main:main',
            'idle = uav.idle:main',
            'gimbal = uav.cam_controller:main'
        ],
    },
)
