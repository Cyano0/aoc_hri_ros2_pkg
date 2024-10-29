#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from mmdet3d.apis import LidarDet3DInferencer
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import open3d as o3d

class MMDet3DNode(Node):
    def __init__(self):
        super().__init__('mmdet3d_node')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('weights_path', '')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('point_cloud_topic', '/back_lidar/points')
        self.declare_parameter('bounding_box_topic', '/detected_bounding_boxes')

        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value
        bounding_box_topic = self.get_parameter('bounding_box_topic').get_parameter_value().string_value

        # Define QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Or RELIABLE
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscription with the QoS profile
        self.subscription = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self.listener_callback,
            qos_profile
        )

        # Create publisher for bounding boxes
        self.bbox_publisher = self.create_publisher(MarkerArray, bounding_box_topic, qos_profile)

        self.inferencer = LidarDet3DInferencer(
            model=model_path,
            weights=weights_path,
            device=device
        )

        # Create a visualizer window
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.bounding_boxes = []

    def listener_callback(self, msg):
        points = self.convert_pc2_to_np(msg)
        if points is not None:
            self.get_logger().info(f"Points shape: {points.shape}, dtype: {points.dtype}")
            inputs = dict(points=points)
            for key, value in inputs.items():
                self.get_logger().info(f"Input {key} shape: {value.shape}, dtype: {value.dtype}")
            results = self.inferencer(inputs)
            self.get_logger().info(f"Points: {points[115:120]}")  # Print some points for debugging
            self.get_logger().info(f"Results: {results}")
            self.visualize_results(points, results)
            self.publish_bounding_boxes(results)

    def convert_pc2_to_np(self, msg):
        cloud_arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
        if 'intensity' not in cloud_arr.dtype.names:
            intensity = np.zeros(cloud_arr.shape[0], dtype=np.float32)
        else:
            intensity = cloud_arr['intensity'].astype(np.float32)
        points = np.zeros((cloud_arr.shape[0], 4), dtype=np.float32)
        points[:, 0] = cloud_arr['x'].astype(np.float32)
        points[:, 1] = cloud_arr['y'].astype(np.float32)
        points[:, 2] = cloud_arr['z'].astype(np.float32)
        points[:, 3] = intensity
        if points.size == 0:
            return None
        return points

    def visualize_results(self, points, results):
        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        labels = results['predictions'][0]['labels_3d']
        colors = np.zeros((points.shape[0], 3))
        self.color_map = {
            0: [1, 0, 0],
            1: [0, 1, 0],
            2: [0, 0, 1]
        }
        for i, label in enumerate(labels):
            colors[i] = self.color_map.get(label, [1, 1, 1])
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        for bbox in self.bounding_boxes:
            self.vis.remove_geometry(bbox)
        self.bounding_boxes.clear()
        bbox_data = results['predictions'][0]['bboxes_3d']
        bbox_labels = results['predictions'][0]['labels_3d']
        bbox_scores = results['predictions'][0]['scores_3d']
        self.get_logger().info(f"Bounding box data: {bbox_data}")
        for bbox, label, score in zip(bbox_data, bbox_labels, bbox_scores):
            if score >= 0.3:
                box = self.create_bounding_box(bbox)
                box.color = self.color_map.get(label, [1, 1, 1])
                self.vis.add_geometry(box)
                self.bounding_boxes.append(box)
        self.vis.update_geometry(self.pcd)
        for bbox in self.bounding_boxes:
            self.vis.update_geometry(bbox)
        self.vis.poll_events()
        self.vis.update_renderer()

    def create_bounding_box(self, bbox):
        center = bbox[:3]
        size = bbox[3:6]
        rotation = bbox[6]
        center[2] += size[2] / 2
        self.get_logger().info(f"Center: {center}, Size: {size}, Rotation: {rotation}")
        bbox_3d = o3d.geometry.OrientedBoundingBox()
        bbox_3d.center = center
        bbox_3d.extent = size
        bbox_3d.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, rotation])
        return bbox_3d

    def publish_bounding_boxes(self, results):
        bbox_data = results['predictions'][0]['bboxes_3d']
        bbox_labels = results['predictions'][0]['labels_3d']
        bbox_scores = results['predictions'][0]['scores_3d']
        marker_array = MarkerArray()

        for i, (bbox, label, score) in enumerate(zip(bbox_data, bbox_labels, bbox_scores)):
            if score >= 0.3:
                marker = Marker()
                marker.header = Header()
                marker.header.frame_id = "lidar_frame"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "bounding_boxes"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = bbox[0]
                marker.pose.position.y = bbox[1]
                marker.pose.position.z = bbox[2]
                marker.scale.x = bbox[3]
                marker.scale.y = bbox[4]
                marker.scale.z = bbox[5]
                marker.color.a = 0.5
                marker.color.r = 1.0 if label == 0 else 0.0
                marker.color.g = 1.0 if label == 1 else 0.0
                marker.color.b = 1.0 if label == 2 else 0.0
                marker_array.markers.append(marker)

        self.bbox_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    mmdet3d_node = MMDet3DNode()
    rclpy.spin(mmdet3d_node)
    mmdet3d_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()