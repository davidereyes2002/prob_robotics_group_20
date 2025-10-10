import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),

        DeclareLaunchArgument('open_torque', default_value='5.0',
                              description='Torque to open door'),

        DeclareLaunchArgument('close_torque', default_value='-5.0',
                              description='Torque to close door'),

        Node(
            package='prob_rob_labs',
            executable='firetruck_signal',
            name='firetruck_signal',
            parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'), 
                    'open_torque': LaunchConfiguration('open_torque'),
                    'close_torque': LaunchConfiguration('close_torque')
                    }]
        )
    ])