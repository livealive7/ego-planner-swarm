import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'isaac_odom_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ma',
    maintainer_email='ma@todo.todo',
    description='Synthesizes nav_msgs/Odometry for ego-planner from Isaac Sim\'s separate pose/twist topics',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_bridge_node = isaac_odom_bridge.odom_bridge_node:main',
            'cmd_vel_mavlink_bridge = isaac_odom_bridge.cmd_vel_mavlink_bridge:main',
            'cmd_vel_px4_bridge = isaac_odom_bridge.cmd_vel_px4_bridge:main',
            'depth_restamp_node = isaac_odom_bridge.depth_restamp_node:main',
            'poscmd_to_cmdvel_node = isaac_odom_bridge.poscmd_to_cmdvel_node:main',
        ],
    },
)
