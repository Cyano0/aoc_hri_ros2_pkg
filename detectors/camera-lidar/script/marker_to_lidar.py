#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseArray
from human_rgbd_det_uol.msg import HumanInfoArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import math
from builtin_interfaces.msg import Duration
import numpy as np
import open3d as o3d
from collections import deque
import transforms3d.quaternions
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from vision_msgs.msg import Detection3DArray
from human_tracker_uol.trackers.tracker_KMFilter import KalmanFilterTracker

def sanitize_value(value):
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)

class BoundingBoxPublisher(Node):
    def __init__(self):
        super().__init__('bounding_box_publisher')

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('bounding_box_color', [1.0, 0.0, 0.0, 1.0]),
                ('marker_scale', 0.1),
                ('subscription_topic', '/yolo_detection'),
                ('lidar_topic', 'points_raw'),
                ('yolo_3d_result_topic', 'yolo_3d_result'),
                ('use_tracker', False),
                ('tracker_dt', 0.1),
                ('tracker_process_noise', 1e-5),
                ('tracker_measurement_noise', 1e-1),
                ('publish_topic', 'human_bounding_box'),
                ('voxel_leaf_size', 0.5),
                ('cluster_tolerance', 0.5),
                ('min_cluster_size', 50),
                ('max_cluster_size', 25000),
                ('bbox_expansion_ratio', 1.5)  # Adjust the ratio to expand the bounding box
            ]
        )

        # Get parameters from the config file
        color_params = self.get_parameter('bounding_box_color').get_parameter_value().double_array_value
        self.marker_color = {
            'r': color_params[0],
            'g': color_params[1],
            'b': color_params[2],
            'a': color_params[3]
        }
        self.marker_scale = self.get_parameter('marker_scale').get_parameter_value().double_value
        self.subscription_topic = self.get_parameter('subscription_topic').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.yolo_3d_result_topic = self.get_parameter('yolo_3d_result_topic').get_parameter_value().string_value
        self.use_tracker = self.get_parameter('use_tracker').get_parameter_value().bool_value
        self.tracker_dt = self.get_parameter('tracker_dt').get_parameter_value().double_value
        self.tracker_process_noise = self.get_parameter('tracker_process_noise').get_parameter_value().double_value
        self.tracker_measurement_noise = self.get_parameter('tracker_measurement_noise').get_parameter_value().double_value

        # Define QoS policy
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            HumanInfoArray,
            self.subscription_topic,
            self.human_info_callback,
            qos_profile=qos_profile
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            qos_profile=qos_profile
        )

        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.publisher = self.create_publisher(MarkerArray, self.publish_topic, qos_profile=qos_profile)

        self.detection3d_pub = self.create_publisher(Detection3DArray, self.yolo_3d_result_topic, qos_profile)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, "detection_cloud", qos_profile)
        # self.marker_pub = self.create_publisher(MarkerArray, self.publish_topic, qos_profile)
        self.test_bbox_pub = self.create_publisher(MarkerArray, 'test_bbox', 10)
        self.checked_points_pub = self.create_publisher(PointCloud2, 'checked_points', qos_profile)
        self.pose_array_pub = self.create_publisher(PoseArray, 'bbox_poses', qos_profile)

        self.existing_ids = set()
        self.trackers = {}
        self.cloud_buffer = deque(maxlen=10)
        self.human_info_buffer = deque(maxlen=10)

        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.check_synchronization)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.last_call_time = self.get_clock().now()

        # Store transformed bounding boxes
        self.transformed_bboxes = []
        # Initialize the dictionary to store previous positions of bounding boxes
        self.previous_positions = {} 

    def lidar_callback(self, msg):
        self.get_logger().info(f"Received point cloud message.")
        self.cloud_buffer.append(msg)

    def human_info_callback(self, msg):
        self.get_logger().info(f"Received human info message.")
        self.human_info_buffer.append(msg)

    def check_synchronization(self):
        if not self.cloud_buffer or not self.human_info_buffer:
            return

        cloud_msg = self.cloud_buffer[-1]
        human_info_msg = self.human_info_buffer[-1]

        time_diff = abs((cloud_msg.header.stamp.sec + cloud_msg.header.stamp.nanosec * 1e-9) -
                        (human_info_msg.header.stamp.sec + human_info_msg.header.stamp.nanosec * 1e-9))

        if time_diff < self.timer_period:
            self.sync_callback(cloud_msg, human_info_msg)

    def sync_callback(self, cloud_msg, human_info_msg):
        self.get_logger().info("sync_callback called")
        try:
            self.get_logger().info(f"HumanInfoArray Message received")
            current_call_time = self.get_clock().now()
            callback_interval = current_call_time - self.last_call_time
            self.last_call_time = current_call_time

            self.get_logger().info("Creating markers")
            self.transformed_bboxes = self.create_and_publish_test_bbox(human_info_msg, cloud_msg.header)

            self.get_logger().info("Filtering cloud with test_bbox")
            all_filtered_clouds = []
            for bbox in self.transformed_bboxes:
                filtered_points = self.filter_cloud(cloud_msg, bbox, cloud_msg.header)
                self.get_logger().info(f"Filtered detection cloud size: {len(filtered_points.points)}")

                self.get_logger().info("Clustering filtered cloud")
                clusters, closest_cluster = self.cluster_filtered_cloud(filtered_points)

                if closest_cluster:
                    self.get_logger().info(f"Publishing closest cluster with {len(closest_cluster.points)} points")
                    self.publish_filtered_cloud(cloud_msg.header, np.asarray(closest_cluster.points), bbox['id'])
                
                all_filtered_clouds.append((clusters, bbox['id']))

            self.publish_clustered_bboxes(all_filtered_clouds, cloud_msg.header.frame_id)

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")

    def filter_cloud(self, cloud_msg, bbox, header):
        points_list = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array([(p[0], p[1], p[2]) for p in points_list], dtype=np.float32)

        self.get_logger().info(f"Filtering with bbox: {bbox}")

        min_bound = bbox['min']
        max_bound = bbox['max']

        mask = np.all(np.logical_and(points >= min_bound, points <= max_bound), axis=1)
        filtered_points = points[mask]

        checked_cloud = o3d.geometry.PointCloud()
        checked_cloud.points = o3d.utility.Vector3dVector(filtered_points)
        self.publish_checked_cloud(header, checked_cloud, bbox['id'])

        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        return filtered_cloud

    def cluster_filtered_cloud(self, filtered_cloud):
        try:
            self.get_logger().info("Performing Euclidean cluster extraction")
            cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
            min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
            max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value

            self.get_logger().info(f"Cluster tolerance: {cluster_tolerance}")
            self.get_logger().info(f"Min cluster size: {min_cluster_size}")
            self.get_logger().info(f"Max cluster size: {max_cluster_size}")

            points = np.asarray(filtered_cloud.points)
            if points.shape[0] == 0:
                return [], o3d.geometry.PointCloud()

            labels = np.array(filtered_cloud.cluster_dbscan(eps=cluster_tolerance, min_points=min_cluster_size, print_progress=False))
            unique_labels = np.unique(labels)
            clusters = []
            closest_cluster = o3d.geometry.PointCloud()
            min_distance = float('inf')

            self.get_logger().info(f"Unique labels: {unique_labels}")

            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points
                cluster_indices = np.where(labels == label)[0]
                cluster_points = points[cluster_indices]
                self.get_logger().info(f"Label {label} has {len(cluster_points)} points")

                if len(cluster_points) < min_cluster_size or len(cluster_points) > max_cluster_size:
                    self.get_logger().info(f"Cluster with label {label} does not meet size constraints")
                    continue

                cluster = filtered_cloud.select_by_index(cluster_indices)
                clusters.append(cluster)
                centroid = np.mean(np.asarray(cluster.points), axis=0)
                distance = np.linalg.norm(centroid)

                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster

            self.get_logger().info(f"Closest cluster has {len(closest_cluster.points)} points")
            return clusters, closest_cluster

        except Exception as e:
            self.get_logger().error(f"Error in cluster_filtered_cloud: {e}")
            return [], o3d.geometry.PointCloud()

    def publish_checked_cloud(self, header, checked_cloud, bbox_id):
        checked_cloud_msg = point_cloud2.create_cloud_xyz32(header, np.asarray(checked_cloud.points))
        self.checked_points_pub.publish(checked_cloud_msg)

    def publish_filtered_cloud(self, header, filtered_points, bbox_id):
        filtered_cloud_msg = point_cloud2.create_cloud_xyz32(header, filtered_points)
        self.detection_cloud_pub.publish(filtered_cloud_msg)

    # def create_and_publish_test_bbox(self, human_info_msg, cloud_frame_header):
    #     try:
    #         marker_array = MarkerArray()
    #         pose_array = PoseArray()
    #         # pose_array.header.frame_id = cloud_frame_header.frame_id
    #         # pose_array.header.stamp = self.get_clock().now().to_msg()
    #         pose_array.header = cloud_frame_header
    #         transformed_bboxes = []

    #         for human_info in human_info_msg.human_infos:
    #             # Determine the marker ID
    #             try:
    #                 marker_id = int(float(human_info.id)) if human_info.id and human_info.id != '' else self.get_unique_id()
    #             except ValueError:
    #                 marker_id = self.get_unique_id()

    #             # Use tracker if enabled
    #             if self.use_tracker:
    #                 if marker_id not in self.trackers:
    #                     self.trackers[marker_id] = KalmanFilterTracker(
    #                         dt=self.tracker_dt,
    #                         process_noise=self.tracker_process_noise,
    #                         measurement_noise=self.tracker_measurement_noise,
    #                         state_dim=3,
    #                         measure_dim=3
    #                     )
    #                     self.trackers[marker_id].set_initial_state(
    #                         [human_info.pose[0], human_info.pose[1], human_info.pose[2]]
    #                     )
    #                 tracker = self.trackers[marker_id]
    #                 tracker.update([human_info.pose[0], human_info.pose[1], human_info.pose[2]])
    #                 predicted_position = tracker.predict()
    #                 human_info.pose[0], human_info.pose[1], human_info.pose[2] = predicted_position

    #             # Create the transform from the bbox frame to the lidar point cloud frame
    #             try:
    #                 tf_stamped = self.tf_buffer.lookup_transform(cloud_frame_header.frame_id, human_info_msg.header.frame_id, human_info_msg.header.stamp, rclpy.duration.Duration(seconds=0.1))
    #                 transform = self.transform_to_matrix(tf_stamped)
    #                 bbox_position = np.array([float(human_info.pose[0]), float(human_info.pose[1]), float(human_info.pose[2]), 1.0])
    #                 transformed_position = np.dot(transform, bbox_position)

    #                 adjusted_position_x = transformed_position[0]
    #                 adjusted_position_y = transformed_position[1]
    #                 adjusted_position_z = transformed_position[2]

    #                 # Transform the bounding box size
    #                 transformed_bbox_size_min = np.dot(transform[:3, :3], np.array([human_info.min[0], human_info.min[1], human_info.min[2]]))
    #                 transformed_bbox_size_max = np.dot(transform[:3, :3], np.array([human_info.max[0], human_info.max[1], human_info.max[2]]))
    #                 bbox_size_x = abs(transformed_bbox_size_max[0] - transformed_bbox_size_min[0])
    #                 bbox_size_y = abs(transformed_bbox_size_max[1] - transformed_bbox_size_min[1])
    #                 bbox_size_z = abs(transformed_bbox_size_max[2] - transformed_bbox_size_min[2])

    #                 self.get_logger().info(f"Transformed min: {transformed_bbox_size_min}, max: {transformed_bbox_size_max}")

    #                 bbox_expansion_ratio = self.get_parameter('bbox_expansion_ratio').get_parameter_value().double_value
    #                 min_bound = np.array([adjusted_position_x - bbox_size_x * bbox_expansion_ratio, adjusted_position_y - bbox_size_y / 2, adjusted_position_z - bbox_size_z * bbox_expansion_ratio/ 2])
    #                 max_bound = np.array([adjusted_position_x + bbox_size_x * bbox_expansion_ratio, adjusted_position_y + bbox_size_y / 2, adjusted_position_z + bbox_size_z * bbox_expansion_ratio/ 2])
    #                 transformed_bboxes.append({'min': min_bound, 'max': max_bound, 'id': marker_id})

    #                 # Create Marker for RViz visualization
    #                 marker = Marker()
    #                 marker.header.frame_id = cloud_frame_header.frame_id
    #                 marker.id = marker_id
    #                 marker.type = Marker.CUBE
    #                 marker.action = Marker.ADD
    #                 marker.pose.position.x = adjusted_position_x
    #                 marker.pose.position.y = adjusted_position_y
    #                 marker.pose.position.z = adjusted_position_z
    #                 marker.pose.orientation.x = 0.0
    #                 marker.pose.orientation.y = 0.0
    #                 marker.pose.orientation.z = 0.0
    #                 marker.pose.orientation.w = 1.0
    #                 marker.scale.x = bbox_size_x
    #                 marker.scale.y = bbox_size_y
    #                 marker.scale.z = bbox_size_z
    #                 marker.color.r = 1.0
    #                 marker.color.g = 0.0
    #                 marker.color.b = 0.0
    #                 marker.color.a = 0.5
    #                 marker.lifetime = Duration(sec=2)

    #                 marker_array.markers.append(marker)

    #                 # Create Pose for PoseArray
    #                 pose = Pose()
    #                 pose.position.x = adjusted_position_x
    #                 pose.position.y = adjusted_position_y
    #                 pose.position.z = adjusted_position_z

    #                 # Calculate orientation based on current and previous positions
    #                 if marker_id in self.previous_positions:
    #                     prev_position = self.previous_positions[marker_id]
    #                     direction = np.array([adjusted_position_x - prev_position[0], 
    #                                           adjusted_position_y - prev_position[1], 
    #                                           adjusted_position_z - prev_position[2]])

    #                     # Calculate quaternion from direction vector
    #                     if np.linalg.norm(direction) > 0:
    #                         direction /= np.linalg.norm(direction)
    #                         yaw = math.atan2(direction[1], direction[0])
    #                         quaternion = transforms3d.quaternions.axangle2quat([0, 0, 1], yaw)
    #                         pose.orientation.x = quaternion[1]
    #                         pose.orientation.y = quaternion[2]
    #                         pose.orientation.z = quaternion[3]
    #                         pose.orientation.w = quaternion[0]
    #                     else:
    #                         # Use a default orientation if no movement
    #                         pose.orientation.x = 0.0
    #                         pose.orientation.y = 0.0
    #                         pose.orientation.z = 0.0
    #                         pose.orientation.w = 1.0
    #                 else:
    #                     # Default orientation for first instance
    #                     pose.orientation.x = 0.0
    #                     pose.orientation.y = 0.0
    #                     pose.orientation.z = 0.0
    #                     pose.orientation.w = 1.0

    #                 # Update the previous position with the current one
    #                 self.previous_positions[marker_id] = (adjusted_position_x, adjusted_position_y, adjusted_position_z)

    #                 pose_array.poses.append(pose)

    #             except Exception as e:
    #                 self.get_logger().error(f"Error transforming bbox: {e}")

    #         self.test_bbox_pub.publish(marker_array)
    #         self.pose_array_pub.publish(pose_array)  # Publish the PoseArray
    #         self.get_logger().info(f"Published test_bbox markers and PoseArray with {len(pose_array.poses)} poses.")

    #         return transformed_bboxes

    #     except Exception as e:
    #         self.get_logger().error(f"Error in create_and_publish_test_bbox: {e}")
    #         return []

    def create_and_publish_test_bbox(self, human_info_msg, cloud_frame_header):
        try:
            marker_array = MarkerArray()
            pose_array = PoseArray()
            pose_array.header = cloud_frame_header
            transformed_bboxes = []

            for human_info in human_info_msg.human_infos:
                # Determine the marker ID
                try:
                    marker_id = int(float(human_info.id)) if human_info.id and human_info.id != '' else self.get_unique_id()
                except ValueError:
                    marker_id = self.get_unique_id()

                # Use tracker if enabled
                if self.use_tracker:
                    if marker_id not in self.trackers:
                        self.trackers[marker_id] = KalmanFilterTracker(
                            dt=self.tracker_dt,
                            process_noise=self.tracker_process_noise,
                            measurement_noise=self.tracker_measurement_noise,
                            state_dim=3,
                            measure_dim=3
                        )
                        self.trackers[marker_id].set_initial_state(
                            [human_info.pose[0], human_info.pose[1], human_info.pose[2]]
                        )
                    tracker = self.trackers[marker_id]
                    tracker.update([human_info.pose[0], human_info.pose[1], human_info.pose[2]])
                    predicted_position = tracker.predict()
                    human_info.pose[0], human_info.pose[1], human_info.pose[2] = predicted_position

                # Create the transform from the bbox frame to the lidar point cloud frame
                try:
                    tf_stamped = self.tf_buffer.lookup_transform(
                        cloud_frame_header.frame_id, 
                        human_info_msg.header.frame_id, 
                        human_info_msg.header.stamp, 
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    transform = self.transform_to_matrix(tf_stamped)
                    bbox_position = np.array([
                        float(human_info.pose[0]), 
                        float(human_info.pose[1]), 
                        float(human_info.pose[2]), 
                        1.0
                    ])
                    transformed_position = np.dot(transform, bbox_position)

                    adjusted_position_x = transformed_position[0]
                    adjusted_position_y = transformed_position[1]
                    adjusted_position_z = transformed_position[2]

                    # Transform the bounding box size
                    transformed_bbox_size_min = np.dot(
                        transform[:3, :3], 
                        np.array([human_info.min[0], human_info.min[1], human_info.min[2]])
                    )
                    transformed_bbox_size_max = np.dot(
                        transform[:3, :3], 
                        np.array([human_info.max[0], human_info.max[1], human_info.max[2]])
                    )
                    bbox_size_x = abs(transformed_bbox_size_max[0] - transformed_bbox_size_min[0])
                    bbox_size_y = abs(transformed_bbox_size_max[1] - transformed_bbox_size_min[1])
                    bbox_size_z = abs(transformed_bbox_size_max[2] - transformed_bbox_size_min[2])

                    self.get_logger().info(
                        f"Transformed min: {transformed_bbox_size_min}, max: {transformed_bbox_size_max}"
                    )

                    bbox_expansion_ratio = self.get_parameter('bbox_expansion_ratio').get_parameter_value().double_value
                    min_bound = np.array([
                        adjusted_position_x - bbox_size_x * bbox_expansion_ratio, 
                        adjusted_position_y - bbox_size_y / 2, 
                        adjusted_position_z - bbox_size_z * bbox_expansion_ratio / 2
                    ])
                    max_bound = np.array([
                        adjusted_position_x + bbox_size_x * bbox_expansion_ratio, 
                        adjusted_position_y + bbox_size_y / 2, 
                        adjusted_position_z + bbox_size_z * bbox_expansion_ratio / 2
                    ])
                    transformed_bboxes.append({'min': min_bound, 'max': max_bound, 'id': marker_id})

                    # Create Marker for RViz visualization
                    marker = Marker()
                    marker.header.frame_id = cloud_frame_header.frame_id
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose.position.x = adjusted_position_x
                    marker.pose.position.y = adjusted_position_y
                    marker.pose.position.z = adjusted_position_z
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = bbox_size_x
                    marker.scale.y = bbox_size_y
                    marker.scale.z = bbox_size_z
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.5
                    marker.lifetime = Duration(sec=2)

                    marker_array.markers.append(marker)

                    # Create Pose for PoseArray
                    pose = Pose()
                    pose.position.x = adjusted_position_x
                    pose.position.y = adjusted_position_y
                    pose.position.z = adjusted_position_z

                    # Calculate orientation based on movement in X-Y plane only
                    if marker_id in self.previous_positions:
                        prev_position = self.previous_positions[marker_id]
                        direction = np.array([
                            adjusted_position_x - prev_position[0], 
                            adjusted_position_y - prev_position[1], 
                            0  # Ignore Z-axis component
                        ])

                        # Calculate quaternion from direction vector in X-Y plane
                        if np.linalg.norm(direction) > 0:
                            direction /= np.linalg.norm(direction)
                            yaw = math.atan2(direction[1], direction[0])  # Calculate yaw angle in X-Y plane
                            quaternion = transforms3d.quaternions.axangle2quat([0, 0, 1], yaw)  # Rotation around Z-axis
                            pose.orientation.x = quaternion[1]
                            pose.orientation.y = quaternion[2]
                            pose.orientation.z = quaternion[3]
                            pose.orientation.w = quaternion[0]
                        else:
                            # Use a default orientation if no movement
                            pose.orientation.x = 0.0
                            pose.orientation.y = 0.0
                            pose.orientation.z = 0.0
                            pose.orientation.w = 1.0
                    else:
                        # Default orientation for first instance
                        pose.orientation.x = 0.0
                        pose.orientation.y = 0.0
                        pose.orientation.z = 0.0
                        pose.orientation.w = 1.0

                    # Update the previous position with the current one
                    self.previous_positions[marker_id] = (adjusted_position_x, adjusted_position_y, adjusted_position_z)

                    pose_array.poses.append(pose)

                except Exception as e:
                    self.get_logger().error(f"Error transforming bbox: {e}")

            self.test_bbox_pub.publish(marker_array)
            self.pose_array_pub.publish(pose_array)  # Publish the PoseArray
            self.get_logger().info(f"Published test_bbox markers and PoseArray with {len(pose_array.poses)} poses.")

            return transformed_bboxes

        except Exception as e:
            self.get_logger().error(f"Error in create_and_publish_test_bbox: {e}")
            return []

    def publish_clustered_bboxes(self, all_filtered_clouds, frame_id):
        marker_array = MarkerArray()

        for clusters, bbox_id in all_filtered_clouds:
            for cluster in clusters:
                min_bound = np.min(np.asarray(cluster.points), axis=0)
                max_bound = np.max(np.asarray(cluster.points), axis=0)
                bbox_size_x = max_bound[0] - min_bound[0]
                bbox_size_y = max_bound[1] - min_bound[1]
                bbox_size_z = max_bound[2] - min_bound[2]
                centroid = np.mean(np.asarray(cluster.points), axis=0)

                marker = Marker()
                marker.header.frame_id = frame_id
                marker.ns = "clustered_bboxes"
                marker.id = bbox_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = centroid[0]
                marker.pose.position.y = centroid[1]
                marker.pose.position.z = centroid[2]
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = bbox_size_x
                marker.scale.y = bbox_size_y
                marker.scale.z = bbox_size_z
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.5
                marker.lifetime = Duration(sec=2)

                marker_array.markers.append(marker)

        self.publisher.publish(marker_array)

    def transform_to_matrix(self, transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        translation_vec = np.array([translation.x, translation.y, translation.z])
        rotation_mat = transforms3d.quaternions.quat2mat([rotation.w, rotation.x, rotation.y, rotation.z])
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_mat
        transform_matrix[:3, 3] = translation_vec
        return transform_matrix

    def get_unique_id(self):
        new_id = 0
        while new_id in self.existing_ids:
            new_id += 1
        self.existing_ids.add(new_id)
        return new_id

def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
