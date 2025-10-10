import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),

        DeclareLaunchArgument('close_torque', default_value='-5.0',
                              description='Torque to apply when closing the door'),
        Node(
            package='prob_rob_labs',
            executable='lab3_assign8',
            name='lab3_assign8',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )
    ])
