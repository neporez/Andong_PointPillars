# Autonomous Ship Package Guide

This guide provides step-by-step instructions for setting up a Jetson device with necessary packages for autonomous ship package, including system updates, Jetson configurations, ROS2 Humble installation, PyTorch with TensorRT, and specific package installations.

---
## Table of Contents
### Installation Guide
- [1. Jetson Orin Nano Setup](#1-jetson-orin-nano-setup)
- [2. ROS2 Humble Installation](#2-ros2-humble-installation)
- [3. PyTorch and TensorRT Installation](#3-pytorch-and-tensorrt-installation)
- [4. Installation Livox LiDAR Package](#4-installation-livox-lidar-package)
- [5. Autonomous Ship Package Installation](#5-autonomous-ship-package-installation)
---
### How to Use
- [1. Execute the Launch File](#1-execute-the-launch-file)
- [2. Turn On the LiDAR Sensor](#2-turn-on-the-lidar-sensor)
- [3. Parameters](#3-parameters)
---

## Installation Guide
## 1. Jetson Orin Nano Setup
```bash
sudo apt update
sudo apt upgrade
sudo apt-get install python3-pip

sudo nvpmodel -m0

sudo -H pip install -U jetson-stats

sudo reboot

jtop
```

## 2. ROS2 Humble Installation

```bash
locale  

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale 

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt upgrade

sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## 3. PyTorch and TensorRT Installation

Refer to NVIDIA forum [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

```bash
wget https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
wget https://nvidia.box.com/shared/static/9agsjfee0my4sxckdpuk9x9gt8agvjje.whl
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl

mv mp164asf3sceb570wvjsrezk1p4ftj8t.whl torch-2.3.0-cp310-cp310-linux_aarch64.whl
mv 9agsjfee0my4sxckdpuk9x9gt8agvjje.whl torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
mv xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl

sudo apt install cuda-toolkit-12-2
sudo apt install python3-libnvinfer-dev
sudo apt install tensorrt
```
## 4. Installation Livox LiDAR Package

Refer to Livox SDK repository [here](https://github.com/Livox-SDK/livox_ros_driver2)

```bash
cd ~
sudo apt install cmake
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd ./Livox-SDK2/
mkdir build
cd build
cmake .. && make -j1
sudo make install

cd ~
git clone https://github.com/Livox-SDK/livox_ros_driver2.git ws_livox/src/livox_ros_driver2
cd ~/ws_livox/src/livox_ros_driver2
./build.sh humble

echo "source ~/ws_livox/install/setup.bash" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib" >> ~/.bashrc
```

Setup Config File (~/ws_livox/src/livox_ros_driver2/config/MID360_config.json)
```bash
{
  "lidar_summary_info" : {
    "lidar_type": 8
  },
  "MID360": {
    "lidar_net_info" : {
      "cmd_data_port": 56100,
      "push_msg_port": 56200,
      "point_data_port": 56300,
      "imu_data_port": 56400,
      "log_data_port": 56500
    },
    "host_net_info" : {
      "cmd_data_ip" : "192.168.10.50", #<-----Your Ethernet Static IP
      "cmd_data_port": 56101,
      "push_msg_ip": "192.168.10.50", #<-----Your Ethernet Static IP
      "push_msg_port": 56201,
      "point_data_ip": "192.168.10.50", #<-----Your Ethernet Static IP
      "point_data_port": 56301,
      "imu_data_ip" : "192.168.10.50", #<-----Your Ethernet Static IP
      "imu_data_port": 56401,
      "log_data_ip" : "",
      "log_data_port": 56501
    }
  },
  "lidar_configs" : [
    {
      "ip" : "192.168.10.171", #<-----Your LiDAR IP
      "pcl_data_type" : 1,
      "pattern_mode" : 0,
      "extrinsic_parameter" : {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "x": 0,
        "y": 0,
        "z": 0
      }
    }
  ]
}
```

Switching Message Type (~/ws_livox/src/livox_ros_driver2/launch/msg_MID360_launch.py)
```bash
xfer_format   = 0    # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
```

build Livox SDK

```bash
cd ~/ws_livox/src/livox_ros_driver2
./build.sh humble
```



## 5. Autonomous Ship Package Installation

```bash
cd ~
mkdir -p ndt_ws/src
mkdir -p autonomous_ship_ws/src

echo "source ~/ndt_ws/install/setup.bash" >> ~/.bashrc
echo "source ~/autonomous_ship_ws/install/setup.bash" >> ~/.bashrc

cd ~/autonomous_ship_ws/src

wget https://github.com/neporez/rain_autonomous_ship/archive/refs/heads/jetson_orin_nano.zip
unzip jetson_orin_nano.zip && rm -rf jetson_orin_nano.zip
mv rain_autonomous_ship-jetson_orin_nano.zip/* . && rm -rf rain_autonomous_ship-jetson_orin_nano
mv PointPillars ~ && mv lidarslam_ros2 ~/ndt_ws/src

cd ~/autonomous_ship_ws
colcon build --symlink-install

cd ~/ndt_ws
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release

cd ~/PointPillars/ops && python setup.py develop --user
cd ~/PointPillars
pip install -r requirements.txt

cd ~/PointPillars/deployment && python pytorch2onnx.py --ckpt ../pretrained/best_points16.pth --saved_onnx_path ../pretrained/best_points16_model.onnx

/usr/src/tensorrt/bin/trtexec --onnx=../pretrained/best_points16_model.onnx --saveEngine=../pretrained/best_points16_model.trt \
  --minShapes=input_pillars:50x16x4,input_coors_batch:50x4,input_npoints_per_pillar:50 \
  --maxShapes=input_pillars:10000x16x4,input_coors_batch:10000x4,input_npoints_per_pillar:10000 \
  --optShapes=input_pillars:2000x16x4,input_coors_batch:2000x4,input_npoints_per_pillar:2000


```
## How to Use

## 1. Execute the Launch File
```bash
ros2 launch rain_autonomous_ship rain_autonomous_ship.launch.py
```

## 2. Turn On the LiDAR Sensor
```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```
or
```bash
ros2 bag play mid360_bagfile
```

## 3. Parameters

 ~/rain_autonomous_ship/src/rain_det/param/rain_det_param.yaml
```bash
/pointcloud_object_detector:
  ros__parameters:
    class_num: 2
    max_num_points: 16 # PointPillars에서 한 Pillars에 들어가는 Point의 개수 
    max_num_pillars: 10000 # Pillar의 최대 개수 -> onnx -> TensorRT로 바꾸는 과정에서 선택한 maxShape의 수를 따라가는 것을 권장
    pcd_limit_range: # 입력될 pointcloud의 x1,y1,z1,x2,y2,z2 범위
    - -69.12
    - -69.12
    - -3.0
    - 69.12
    - 69.12
    - 5.0
    voxel_size: # Pillar의 크기
    - 0.32
    - 0.32
    - 8.0
    pointcloud_topic: /rain/autonomous_ship/ndt_filtered_pointcloud # PointPillars Model에 들어가는 Pointcloud2 topic name
    trt_engine: /PointPillars/pretrained/best_points16_model.trt # TensorRT engine path
    torch_ckpt: /PointPillars/pretrained/best_points16.pth
    inference_time_check: True # 추론 시간 확인
    tensorrt_enable: True
    marker_queue_size: 1000 # dbscan에서 사용될 marker의 저장 용량
    dbscan_eps: 1.0 # 클러스터 인정 범위
    dbscan_min_samples: 3 #최소 클러스터 인정 개수
    dbscan_tracking_queue_distance: 2.0 # marker_queue와 tracking_queue 간의 식별 과정을 위한 distance
    dbscan_update_tracking_queue_weight: 0.1 # marker_queue가 tracking_queue를 업데이트 시킬 때 tracking_queue가 기존 정보를 얼마나 남길 것인지
    port_length: 32.0 # 항구의 길이
    port_width: 5.6 # 항구의 너비(왼쪽 선박 중앙에서 오른쪽 선박 중앙까지 길이)
    boat_distance: 4.3 # 보트끼리의 간격
```
~/rain_autonomous_ship/src/rain_autonomous_ship/param/rain_autonomous_ship_param.yaml
```bash
/laserscan_map:
  ros__parameters:
    slam_map_topic_name: /map # local map의 PointCloud2
    pose_topic_name: /current_pose # Local Map 상에서의 Pose
    map_angle_range:
    - -180
    - 180
    map_laser_range:
    - 0.0
    - 200.0
    laserscan_angle_increment: 0.1
    map_vertical_fov_range:
    - -30
    - 30
    visualize_frame: ndt_map
```
~/ndt_ws/src/lidarslam_ros2/lidarslam/param/lidarslam.yaml
```bash
scan_matcher:
  ros__parameters:
    global_frame_id: "ndt_map"
    robot_frame_id: "livox_frame"
    registration_method: "NDT"
    ndt_resolution: 2.0
    ndt_num_threads: 2
    gicp_corr_dist_threshold: 5.0
    trans_for_mapupdate: 1.5
    vg_size_for_input: 0.5
    vg_size_for_map: 0.1
    use_min_max_filter: true
    scan_min_range: 2.0
    scan_max_range: 200.0
    scan_period: 0.2
    map_publish_period: 5.0
    num_targeted_cloud: 20
    set_initial_pose: true
    initial_pose_x: 0.0
    initial_pose_y: 0.0
    initial_pose_z: 0.0
    initial_pose_qx: 0.0
    initial_pose_qy: 0.0
    initial_pose_qz: 0.0
    initial_pose_qw: 1.0
    use_imu: false
    use_odom: false
    debug_flag: false
    ndt_accumulate_stack: 1 # 포인트 클라우드 중첩 개수
    ndt_accumulate_cloud_topic_name: "/rain/autonomous_ship/ndt_filtered_pointcloud" # 모델의 입력으로 들어갈 포인트 클라우드 토픽

graph_based_slam:
    ros__parameters:
      registration_method: "NDT"
      ndt_resolution: 1.0
      ndt_num_threads: 2
      voxel_leaf_size: 0.1
      loop_detection_period: 3000
      threshold_loop_closure_score: 0.7
      distance_loop_closure: 100.0
      range_of_searching_loop_closure: 20.0
      search_submap_num: 2
      num_adjacent_pose_cnstraints: 5
      use_save_map_in_loop: true
      debug_flag: true
```


---
