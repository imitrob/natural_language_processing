#!/usr/bin/env python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    user_arg = DeclareLaunchArgument('user_name', default_value='casper', description='User name')

    homing_node = Node(
        package='natural_language_processing',
        executable='nl_node',
        name='nl_node',
        output='screen',
        parameters=[{
            'user_name': LaunchConfiguration('user_name'),
        }]
    )

    return LaunchDescription([
        user_arg,
        homing_node,
    ])
