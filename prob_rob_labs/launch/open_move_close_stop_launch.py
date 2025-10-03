import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),

        DeclareLaunchArgument('forward_speed', default_value='0.5',
                              description='Forward speed of the robot'),
        
        DeclareLaunchArgument('open_torque', default_value='1.5',
                              description='Torque to apply when opening the door'),

        DeclareLaunchArgument('close_torque', default_value='-1.5',
                              description='Torque to apply when closing the door'),

        DeclareLaunchArgument('move_time', default_value='8.0',
                              description='Time to move forward'),

        DeclareLaunchArgument('stop_time', default_value='2.0',
                              description='Time to keep robot stopped'),

        DeclareLaunchArgument('close_time', default_value='10.0',
                              description='Time to keep applying close torque'),

        Node(
            package='prob_rob_labs',
            executable='open_move_close_stop',
            name='open_move_close_stop',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'forward_speed': LaunchConfiguration('forward_speed'),
                'open_torque': LaunchConfiguration('open_torque'),
                'close_torque': LaunchConfiguration('close_torque'),
                'move_time': LaunchConfiguration('move_time'),
                'stop_time': LaunchConfiguration('stop_time'),
                'close_time': LaunchConfiguration('close_time')
            }]
        )
    ])
