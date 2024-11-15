import os

import launch
import launch_ros.actions
from launch.actions import SetEnvironmentVariable

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    main_param_dir = launch.substitutions.LaunchConfiguration(
        'main_param_dir',
        default=os.path.join(
            get_package_share_directory('lidarslam'),
            'param',
            'lidarslam.yaml'))
    
    rviz_param_dir = launch.substitutions.LaunchConfiguration(
        'rviz_param_dir',
        default=os.path.join(
            get_package_share_directory('lidarslam'),
            'rviz',
            'mapping.rviz'))

    # Set default logging level to ERROR
    os.environ['RCUTILS_LOGGING_LEVEL'] = 'ERROR'

    # Environment variable to control output
    env_var = SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}]: {message}')

    mapping = launch_ros.actions.Node(
        package='scanmatcher',
        executable='scanmatcher_node',
        parameters=[main_param_dir],
        remappings=[('/input_cloud','/livox/lidar')], #/livox/lidar
        output='log',
        arguments=['--ros-args', '--log-level', 'ERROR']
    )

    tf = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0','0','0','0','0','0','1','ndt_map','livox_frame'],
        output='log'
    )

    graphbasedslam = launch_ros.actions.Node(
        package='graph_based_slam',
        executable='graph_based_slam_node',
        parameters=[main_param_dir],
        output='log',
        arguments=['--ros-args', '--log-level', 'ERROR']
    )
    
    rviz = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_param_dir],
        output='log'
    )

    return launch.LaunchDescription([
        env_var,
        launch.actions.DeclareLaunchArgument(
            'main_param_dir',
            default_value=main_param_dir,
            description='Full path to main parameter file to load'),
        mapping,
        tf,
        graphbasedslam,
        rviz,
    ])