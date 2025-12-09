from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'prob_rob_labs'

launch_files = glob('launch/*_launch.py') + \
    glob('launch/*_launch.xml') + \
    glob('launch/*.yaml') + \
    glob('launch/*.yml')

data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    (os.path.join('share', package_name, 'launch'), launch_files),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src', exclude=['test']),
    package_dir={'': 'src'},
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ilija Hadzic',
    maintainer_email='ih2435@columbia.edu',
    description='Prob Rob Labs',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pf_slam = pf_slam.pf_slam:main',
            'lab6_assign5 = lab6_assign5.lab6_assign5:main',
            'localization_error = localization_error.localization_error:main',
            'ekf_localization = ekf_localization.ekf_localization:main',
            'lab5_assign3 = lab5_assign3.lab5_assign3:main',
            'lab5_assign2 = lab5_assign2.lab5_assign2:main',
            'lab5_assign1 = lab5_assign1.lab5_assign1:main',
            'EKF_error_analysis = EKF_error_analysis.EKF_error_analysis:main',
            'EKFNode = EKFNode.EKFNode:main',
            'cmd_vel_stamped = cmd_vel_stamped.cmd_vel_stamped:main',
            'lab4_assign1 = lab4_assign1.lab4_assign1:main',
            'lab3_assign8 = lab3_assign8.lab3_assign8:main',
            'lab3_assign7 = lab3_assign7.lab3_assign7:main',
            'firetruck_signal = firetruck_signal.firetruck_signal:main',
            'open_move_close_stop = open_move_close_stop.open_move_close_stop:main',
            'image_mean_feature_x = image_mean_feature_x.image_mean_feature_x:main',
            'flaky_door_opener = flaky_door_opener.flaky_door_opener:main',
        ],
    }
)
