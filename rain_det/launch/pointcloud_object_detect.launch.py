from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    
    pointpillars_params = os.path.join(
        get_package_share_directory('rain_det'),
        'param',
        'rain_det_param.yaml'
    )

    pointpillars_node = Node(
        package='rain_det',
        executable='pointcloud_object_detector',
        name='pointcloud_object_detector',
        output='screen',
        parameters=[pointpillars_params]
    )

    bbox_cluster_node = Node(
        package='rain_det',
        executable='bbox_cluster',
        name='bbox_cluster',
        output='screen',
        parameters=[pointpillars_params]
    )

    return LaunchDescription([
        pointpillars_node,
        bbox_cluster_node
    ])