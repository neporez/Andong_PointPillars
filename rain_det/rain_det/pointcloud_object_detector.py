import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from det_msgs.msg import DetectedObjects

import pycuda.driver as cuda
import time, os

from rain_det.util.pcd_process import point_range_filter, pointcloud2_to_array
from rain_det.model.model_trt import PointPillarsTensorRT
from rain_det.model.model_torch import PointPillarsTorch


cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

class PointCloudObjectDetector(Node):
    def __init__(self):
        super().__init__('pointcloud_object_detector')
        self.home_path = os.path.expanduser('~')
        ##################################Parameter####################################################################
        self.declare_parameters(
            namespace='',
            parameters=[
                ('trt_engine', '/PointPillars/pretrained/best_points16_model.trt'),
                ('torch_ckpt', '/PointPillars/pretrained/best_points16.pth'),
                ('pointcloud_topic', '/rain/autonomous_ship/ndt_filtered_pointcloud'),
                ('class_num', 2),
                ('pcd_limit_range', [-69.12, -69.12, -3.0, 69.12, 69.12, 5.0]),
                ('voxel_size' , [0.32,0.32, 8.0]),
                ('max_num_points', 16),
                ('max_num_pillars', 10000),
                ('inference_time_check', True),
                ('tensorrt_enable', False)
            ]
        )
        self.engine_path = self.home_path + self.get_parameter('trt_engine').get_parameter_value().string_value
        self.torch_ckpt_path = self.home_path + self.get_parameter('torch_ckpt').get_parameter_value().string_value
        self.pointcloud_topic_name = self.get_parameter('pointcloud_topic').get_parameter_value().string_value
        self.class_num = self.get_parameter('class_num').get_parameter_value().integer_value
        self.pcd_limit_range_value = list(self.get_parameter('pcd_limit_range').get_parameter_value().double_array_value)
        self.voxel_size_value = list(self.get_parameter('voxel_size').get_parameter_value().double_array_value)
        self.max_num_points = self.get_parameter('max_num_points').get_parameter_value().integer_value
        self.max_num_pillars = self.get_parameter('max_num_pillars').get_parameter_value().integer_value
        self.inference_time_check = self.get_parameter('inference_time_check').get_parameter_value().bool_value
        self.tensorrt_enable = self.get_parameter('tensorrt_enable').get_parameter_value().bool_value
    
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic_name,
            self.pointcloud_callback,
            10)
        
        self.detected_objects_publisher = self.create_publisher(DetectedObjects, '/rain/det/detected_objects', 10)

        self.last_points = None

        #############################################Model Initialize###############################################

        self.model = None
        if self.tensorrt_enable :        
            self.model = PointPillarsTensorRT(nclasses=self.class_num,
                                voxel_size=self.voxel_size_value,
                                point_cloud_range=self.pcd_limit_range_value, 
                                max_num_points=self.max_num_points, 
                                max_num_pillars=self.max_num_pillars,
                                engine_path=self.engine_path)
        else :
            self.model = PointPillarsTorch(nclasses=self.class_num,
                                voxel_size=self.voxel_size_value,
                                point_cloud_range=self.pcd_limit_range_value, 
                                max_num_points=self.max_num_points, 
                                max_num_pillars=self.max_num_pillars,
                                torch_path=self.torch_ckpt_path)
        
        ############################################################################################################

    def pointcloud_callback(self, msg):
        self.last_points = point_range_filter(pointcloud2_to_array(msg), point_range=self.pcd_limit_range_value)
        self.process_pointcloud(self.last_points)
        
    def process_pointcloud(self, points):
        if self.last_points is None:
            return
        
        current_time = time.time()
        
        
        cuda_driver_context.push()

        lidar_bboxes, labels, scores = self.model.inference(points)
    
        cuda_driver_context.pop()
        

        try :
            detected_obj = DetectedObjects()
            detected_obj.bboxes = lidar_bboxes.reshape(-1).tolist()
            detected_obj.bboxes_num = lidar_bboxes.shape[0]
            detected_obj.labels = labels.tolist()
            detected_obj.scores = scores.tolist()
    
            self.detected_objects_publisher.publish(detected_obj)
        except :
            pass
        
        if self.inference_time_check:
            self.get_logger().info(f'Model inference time: {time.time() - current_time}')

def main(args=None):
    rclpy.init(args=args)
    pointcloud_processor = PointCloudObjectDetector()
    
    try:
        rclpy.spin(pointcloud_processor)
    except KeyboardInterrupt:
        pass
    finally:
        pointcloud_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()