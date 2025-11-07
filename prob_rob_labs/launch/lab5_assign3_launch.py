import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),
        DeclareLaunchArgument('landmark_target', default_value='cyan',
                              description='color for landmark to be detected'),
        Node(
            package='prob_rob_labs',
            executable='lab5_assign3',
            name='lab5_assign3',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}, {'landmark_target': LaunchConfiguration('landmark_target')}]
        )
    ])
