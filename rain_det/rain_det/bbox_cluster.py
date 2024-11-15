import rclpy
import numpy as np
from rclpy.node import Node
from rain_det.util.dbscan import BoundingBoxDBSCAN
from det_msgs.msg import DetectedObjects,ClusteredDetectedObjects
from geometry_msgs.msg import PoseStamped


class BBoxCluster(Node):
    def __init__(self):
        super().__init__('bbox_cluster')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('marker_queue_size', 100),
                ('dbscan_eps', 1.0),
                ('dbscan_min_samples', 5),
                ('dbscan_tracking_queue_distance', 2.0),
                ('dbscan_update_tracking_queue_weight', 0.1),
                ('pose_topic_name', '/current_pose'),
                ]
        )

        self.marker_queue_size = self.get_parameter('marker_queue_size').get_parameter_value().integer_value
        self.dbscan_eps = self.get_parameter('dbscan_eps').get_parameter_value().double_value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').get_parameter_value().integer_value
        self.dbscan_tracking_queue_distance = self.get_parameter('dbscan_tracking_queue_distance').get_parameter_value().double_value
        self.dbscan_update_tracking_queue_weight = self.get_parameter('dbscan_update_tracking_queue_weight').get_parameter_value().double_value
        self.pose_topic_name = self.get_parameter('pose_topic_name').get_parameter_value().string_value


        self.detected_objects_subscription = self.create_subscription(DetectedObjects, '/rain/det/detected_objects', self.detected_objects_callback,10)
        self.local_map_pose_subscription = self.create_subscription(PoseStamped, self.pose_topic_name , self.local_map_pose_callback,10)

        self.clustered_detected_objects_publisher = self.create_publisher(ClusteredDetectedObjects, '/rain/det/clustered_detected_objects',10)

        self.current_pose = None
        self.dbscan = BoundingBoxDBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, queue_size=self.marker_queue_size, tracking_queue_size=self.marker_queue_size, distance_threshold = self.dbscan_tracking_queue_distance, update_weight=self.dbscan_update_tracking_queue_weight)

    def local_map_pose_callback(self,msg):
        self.current_pose = msg.pose
    
    def detected_objects_callback(self, msg: DetectedObjects):
        
        if self.current_pose is None :
            return

        lidar_bboxes = np.array(msg.bboxes).reshape(msg.bboxes_num,-1)
        labels = np.array(msg.labels)
        scores = np.array(msg.scores)

        clustered_bboxes = self.dbscan.clustering((lidar_bboxes, labels, scores), current_pose=self.current_pose)
        if clustered_bboxes:
            
            clustered_detected_objects = ClusteredDetectedObjects()
            bbox_list = []
            label_list = []
            score_list = []
            new_bbox_check_list = []
            id_list = []

            for bbox, label, score, new_bbox_check, id in clustered_bboxes:
                bbox_list.extend(bbox.tolist())
                label_list.append(int(label))
                score_list.append(float(score))
                new_bbox_check_list.append(int(new_bbox_check))
                id_list.append(int(id))

            clustered_detected_objects.bboxes = bbox_list
            clustered_detected_objects.bboxes_num = len(clustered_bboxes)
            clustered_detected_objects.labels = label_list
            clustered_detected_objects.scores = score_list
            clustered_detected_objects.new_bbox_check = new_bbox_check_list
            clustered_detected_objects.id = id_list
            clustered_detected_objects.pose = self.current_pose

            self.clustered_detected_objects_publisher.publish(clustered_detected_objects)
    
def main(args=None):
    rclpy.init(args=args)
    bbox_cluster = BBoxCluster()
    
    try:
        rclpy.spin(bbox_cluster)
    except KeyboardInterrupt:
        pass
    finally:
        bbox_cluster.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()