#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, PointCloud2, Image
from vision_msgs.msg import Detection3DArray, Detection3D
from human_rgbd_det_uol.msg import YoloResultDistance
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import tf2_ros
import image_geometry
import transforms3d.quaternions
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Duration

class TrackerWithCloudNode(Node):
    def __init__(self):
        super().__init__('tracker_with_cloud_node')

        # Declare parameters and set default values
        self.declare_parameter("camera_info_topic", "camera_info")
        self.declare_parameter("lidar_topic", "points_raw")
        self.declare_parameter("yolo_result_topic", "yolo_result")
        self.declare_parameter("yolo_3d_result_topic", "yolo_3d_result")
        self.declare_parameter("cluster_tolerance", 0.5)
        self.declare_parameter("voxel_leaf_size", 0.5)
        self.declare_parameter("min_cluster_size", 100)
        self.declare_parameter("max_cluster_size", 25000)

        # Retrieve parameters
        self.camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.lidar_topic = self.get_parameter("lidar_topic").get_parameter_value().string_value
        self.yolo_result_topic = self.get_parameter("yolo_result_topic").get_parameter_value().string_value
        self.yolo_3d_result_topic = self.get_parameter("yolo_3d_result_topic").get_parameter_value().string_value

        # Define QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Set up subscribers with message filters to synchronize messages
        self.camera_info_sub = Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile)
        self.lidar_sub = Subscriber(self, PointCloud2, self.lidar_topic, qos_profile=qos_profile)
        self.yolo_result_sub = Subscriber(self, YoloResultDistance, self.yolo_result_topic, qos_profile=qos_profile)
        self.sync = ApproximateTimeSynchronizer([self.camera_info_sub, self.lidar_sub, self.yolo_result_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.sync_callback)

        # Set up publishers
        self.detection3d_pub = self.create_publisher(Detection3DArray, self.yolo_3d_result_topic, qos_profile)
        self.detection_cloud_pub = self.create_publisher(PointCloud2, "detection_cloud", qos_profile)
        self.marker_pub = self.create_publisher(MarkerArray, "detection_marker", qos_profile)

        # Initialize CV Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()

        # Initialize time tracking for callbacks
        self.last_call_time = self.get_clock().now()

        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize camera model
        self.cam_model = image_geometry.PinholeCameraModel()

    def sync_callback(self, camera_info_msg, cloud_msg, yolo_result_msg):
        self.get_logger().info("sync_callback called")
        try:
            # Print YOLO result message for debugging
            self.get_logger().info(f"YOLO Result Message: {yolo_result_msg.detections}")

            # Get the current time and calculate the interval since the last call
            current_call_time = self.get_clock().now()
            callback_interval = current_call_time - self.last_call_time
            self.last_call_time = current_call_time

            # Downsample the incoming point cloud
            self.get_logger().info("Downsampling point cloud")
            downsampled_cloud = self.downsample_cloud_msg(cloud_msg)

            # Initialize camera model from the CameraInfo message
            self.get_logger().info("Initializing camera model")
            self.from_camera_info(camera_info_msg)

            # Transform the point cloud to the camera frame
            self.get_logger().info("Transforming point cloud")
            transformed_cloud = self.cloud_to_transformed_cloud(downsampled_cloud, cloud_msg.header.frame_id, self.cam_model.tfFrame(), cloud_msg.header.stamp)

            # Prepare messages for detection results and detection cloud
            detection3d_array_msg = Detection3DArray()
            detection_cloud_msg = PointCloud2()

            # Project the transformed cloud and YOLO results into 3D detection results
            self.get_logger().info("Projecting cloud")
            self.project_cloud(transformed_cloud, yolo_result_msg, cloud_msg.header, detection3d_array_msg, detection_cloud_msg)

            # Check if detections are added to detection3d_array_msg
            self.get_logger().info(f"Detections in detection3d_array_msg: {detection3d_array_msg.detections}")

            # Create markers for visualization
            self.get_logger().info("Creating markers")
            marker_array_msg = self.create_marker_array(detection3d_array_msg, callback_interval.nanoseconds)

            # Publish detection results, detection cloud, and markers
            self.get_logger().info("Publishing detection results and markers")
            self.detection3d_pub.publish(detection3d_array_msg)
            self.detection_cloud_pub.publish(detection_cloud_msg)
            self.marker_pub.publish(marker_array_msg)
        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {e}")

    def from_camera_info(self, camera_info_msg):
        try:
            # Initialize the camera model using the CameraInfo message
            self.cam_model.fromCameraInfo(camera_info_msg)
        except Exception as e:
            self.get_logger().error(f"Error in from_camera_info: {e}")


    def cloud_to_transformed_cloud(self, cloud, source_frame, target_frame, stamp):
        try:
            self.get_logger().info(f"Transforming point cloud from {source_frame} to {target_frame}")
            tf_stamped = self.tf_buffer.lookup_transform(target_frame, source_frame, stamp, rclpy.duration.Duration(seconds=0.1))
            transform = self.transform_to_matrix(tf_stamped.transform)
            transformed_cloud = self.transform_point_cloud(cloud, transform)
            self.get_logger().info(f"Transformed point cloud with {len(transformed_cloud.points)} points")
            return transformed_cloud
        except Exception as e:
            self.get_logger().warn(f"Error in cloud_to_transformed_cloud: {e}")
            return cloud

    def transform_to_matrix(self, transform):
        try:
            # Convert a TransformStamped message to a 4x4 transformation matrix
            translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
            rotation = transforms3d.quaternions.quat2mat([
                transform.rotation.w,  # Note the order: w, x, y, z
                transform.rotation.x, 
                transform.rotation.y, 
                transform.rotation.z
            ])
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
            return matrix
        except Exception as e:
            self.get_logger().error(f"Error in transform_to_matrix: {e}")
            return np.eye(4)

    def transform_point_cloud(self, cloud, transform):
        try:
            # Apply the transformation to the point cloud
            points = np.asarray(cloud.points)
            # Homogeneous coordinates for the point cloud
            ones = np.ones((points.shape[0], 1))
            points_hom = np.hstack([points, ones])
            # Apply transformation
            transformed_points_hom = points_hom.dot(transform.T)
            # Extract the transformed points (ignoring the homogeneous coordinate)
            transformed_points = transformed_points_hom[:, :3]
            # Create a new point cloud with transformed points
            transformed_cloud = o3d.geometry.PointCloud()
            transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
            return transformed_cloud
        except Exception as e:
            self.get_logger().error(f"Error in transform_point_cloud: {e}")
            return cloud

    def downsample_cloud_msg(self, cloud_msg):
        try:
            self.get_logger().info("Downsampling point cloud")
            points_list = list(point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
            points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float32)

            self.get_logger().info(f"Original point cloud size: {len(points)}")

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)

            voxel_leaf_size = self.get_parameter("voxel_leaf_size").get_parameter_value().double_value
            self.get_logger().info(f"Voxel leaf size: {voxel_leaf_size}")
            downsampled_cloud = cloud.voxel_down_sample(voxel_size=voxel_leaf_size)

            self.get_logger().info(f"Downsampled point cloud size: {len(downsampled_cloud.points)}")

            return downsampled_cloud
        except Exception as e:
            self.get_logger().error(f"Error in downsample_cloud_msg: {e}")
            return o3d.geometry.PointCloud()

    def project_cloud(self, cloud, yolo_result_msg, header, detection3d_array_msg, combine_detection_cloud_msg):
        try:
            self.get_logger().info("Projecting the cloud and YOLO results into 3D detection results")
            combine_detection_cloud = o3d.geometry.PointCloud()
            detection3d_array_msg.header = header
            detection3d_array_msg.header.stamp = yolo_result_msg.header.stamp

            for i, detection in enumerate(yolo_result_msg.detections.detections):
                self.get_logger().info(f"Processing detection {i}")
                detection_cloud_raw = o3d.geometry.PointCloud()

                if len(yolo_result_msg.masks) == 0:
                    self.process_points_with_bbox(cloud, detection, detection_cloud_raw)
                else:
                    self.process_points_with_mask(cloud, yolo_result_msg.masks[i], detection_cloud_raw)

                self.get_logger().info(f"Points in detection_cloud_raw: {len(detection_cloud_raw.points)}")

                if len(detection_cloud_raw.points) > 0:
                    self.get_logger().info("Transforming detection cloud to camera frame")
                    detection_cloud = self.cloud_to_transformed_cloud(detection_cloud_raw, header.frame_id, self.cam_model.tfFrame(), header.stamp)
                    self.get_logger().info(f"Points in transformed detection cloud: {len(detection_cloud.points)}")

                    self.get_logger().info("Extracting closest cluster from detection cloud")
                    closest_detection_cloud = self.euclidean_cluster_extraction(detection_cloud)
                    self.get_logger().info(f"Points in closest_detection_cloud: {len(closest_detection_cloud.points)}")
                    
                    combine_detection_cloud += closest_detection_cloud

                    self.create_bounding_box(detection3d_array_msg, closest_detection_cloud, detection.results)

            combine_detection_cloud_msg = point_cloud2.create_cloud_xyz32(header, np.asarray(combine_detection_cloud.points))
            self.get_logger().info(f"Detections added: {len(detection3d_array_msg.detections)}")
        except Exception as e:
            self.get_logger().error(f"Error in project_cloud: {e}")

    def process_points_with_bbox(self, cloud, detection2d_msg, detection_cloud_raw):
        try:
            self.get_logger().info(f"Processing bbox: {detection2d_msg.bbox}")
            # Process points within the bounding box of the detection
            for point in np.asarray(cloud.points):
                pt_cv = np.array([point[0], point[1], point[2]])
                uv = self.cam_model.project3dToPixel(pt_cv)

                if (point[2] > 0 and uv[0] > 0 and uv[0] >= detection2d_msg.bbox.center.position.x - detection2d_msg.bbox.size_x / 2 and
                    uv[0] <= detection2d_msg.bbox.center.position.x + detection2d_msg.bbox.size_x / 2 and
                    uv[1] >= detection2d_msg.bbox.center.position.y - detection2d_msg.bbox.size_y / 2 and
                    uv[1] <= detection2d_msg.bbox.center.position.y + detection2d_msg.bbox.size_y / 2):
                    detection_cloud_raw.points.append(point)
        except Exception as e:
            self.get_logger().error(f"Error in process_points_with_bbox: {e}")

    def process_points_with_mask(self, cloud, mask_image_msg, detection_cloud_raw):
        try:
            self.get_logger().info(f"Processing mask image")
            # Process points within the mask of the detection
            cv_ptr = self.bridge.imgmsg_to_cv2(mask_image_msg, desired_encoding='mono8')

            for point in np.asarray(cloud.points):
                pt_cv = np.array([point[0], point[1], point[2]])
                uv = self.cam_model.project3dToPixel(pt_cv)

                if (point[2] > 0 and uv[0] >= 0 and uv[0] < cv_ptr.shape[1] and uv[1] >= 0 and uv[1] < cv_ptr.shape[0]):
                    if cv_ptr[int(uv[1]), int(uv[0])] > 0:
                        detection_cloud_raw.points.append(point)
                        
            # Debug: Check the number of points added
            self.get_logger().info(f"Points added to detection_cloud_raw: {len(detection_cloud_raw.points)}")
        except Exception as e:
            self.get_logger().error(f"Error in process_points_with_mask: {e}")

    def create_bounding_box(self, detection3d_array_msg, cloud, detections_results_msg):
        try:
            # Create a 3D bounding box for the detected object
            points = np.asarray(cloud.points)
            if points.size == 0:
                self.get_logger().info("No points in cloud, skipping bounding box creation")
                return  # If no points, skip this detection

            self.get_logger().info(f"Creating bounding box for cloud with {len(points)} points")

            centroid = np.mean(points, axis=0)
            self.get_logger().info(f"Centroid: {centroid}")

            theta = -np.arctan2(centroid[1], np.sqrt(centroid[0]**2 + centroid[2]**2))
            self.get_logger().info(f"Theta: {theta}")

            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, theta])
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix

            transformed_cloud = cloud.transform(transform)
            self.get_logger().info("Transformed cloud")

            aabb = transformed_cloud.get_axis_aligned_bounding_box()
            self.get_logger().info(f"AABB: {aabb}")

            bbox_center_transformed = aabb.get_center()
            bbox_center = np.dot(np.linalg.inv(transform), np.hstack((bbox_center_transformed, 1)))[:3]
            self.get_logger().info(f"BBox Center: {bbox_center}")

            min_bound = aabb.min_bound
            max_bound = aabb.max_bound
            extent = max_bound - min_bound
            self.get_logger().info(f"Extent: {extent}")

            q = transforms3d.quaternions.mat2quat(transform[:3, :3])
            self.get_logger().info(f"Quaternion: {q}")

            detection3d_msg = Detection3D()
            detection3d_msg.bbox.center.position.x = bbox_center[0]
            detection3d_msg.bbox.center.position.y = bbox_center[1]
            detection3d_msg.bbox.center.position.z = bbox_center[2]
            detection3d_msg.bbox.center.orientation.x = q[1]
            detection3d_msg.bbox.center.orientation.y = q[2]
            detection3d_msg.bbox.center.orientation.z = q[3]
            detection3d_msg.bbox.center.orientation.w = q[0]
            detection3d_msg.bbox.size.x = extent[0]
            detection3d_msg.bbox.size.y = extent[1]
            detection3d_msg.bbox.size.z = extent[2]
            detection3d_msg.results = detections_results_msg
            detection3d_array_msg.detections.append(detection3d_msg)
        except Exception as e:
            self.get_logger().error(f"Error in create_bounding_box: {e}")


    def euclidean_cluster_extraction(self, cloud):
        try:
            self.get_logger().info("Performing Euclidean cluster extraction")
            cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
            min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
            max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value

            self.get_logger().info(f"Cluster tolerance: {cluster_tolerance}")
            self.get_logger().info(f"Min cluster size: {min_cluster_size}")
            self.get_logger().info(f"Max cluster size: {max_cluster_size}")

            # Increase verbosity of DBSCAN
            labels = np.array(cloud.cluster_dbscan(eps=cluster_tolerance, min_points=min_cluster_size, print_progress=True))
            unique_labels = np.unique(labels)
            closest_cluster = o3d.geometry.PointCloud()
            min_distance = float('inf')

            self.get_logger().info(f"Unique labels: {unique_labels}")

            for label in unique_labels:
                if label == -1:
                    continue
                cluster = cloud.select_by_index(np.where(labels == label)[0])
                self.get_logger().info(f"Label {label} has {len(cluster.points)} points")
                
                if len(cluster.points) < min_cluster_size or len(cluster.points) > max_cluster_size:
                    self.get_logger().info(f"Cluster with label {label} does not meet size constraints")
                    continue

                centroid = np.mean(np.asarray(cluster.points), axis=0)
                distance = np.linalg.norm(centroid)

                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster

            self.get_logger().info(f"Closest cluster has {len(closest_cluster.points)} points")
            return closest_cluster
        except Exception as e:
            self.get_logger().error(f"Error in euclidean_cluster_extraction: {e}")
            return cloud

    def create_marker_array(self, detection3d_array_msg, duration_ns):
        try:
            # Create markers for visualization in RViz
            marker_array_msg = MarkerArray()
            duration_msg = Duration(nanosec=duration_ns)

            for i, detection in enumerate(detection3d_array_msg.detections):
                if np.isfinite(detection.bbox.size.x) and np.isfinite(detection.bbox.size.y) and np.isfinite(detection.bbox.size.z):
                    marker_msg = Marker()
                    marker_msg.header = detection3d_array_msg.header
                    marker_msg.ns = "detection"
                    marker_msg.id = i
                    marker_msg.type = Marker.CUBE
                    marker_msg.action = Marker.ADD
                    marker_msg.pose = detection.bbox.center
                    marker_msg.scale.x = detection.bbox.size.x
                    marker_msg.scale.y = detection.bbox.size.y
                    marker_msg.scale.z = detection.bbox.size.z
                    marker_msg.color.r = 0.0
                    marker_msg.color.g = 1.0
                    marker_msg.color.b = 0.0
                    marker_msg.color.a = 0.5
                    marker_msg.lifetime = duration_msg
                    marker_array_msg.markers.append(marker_msg)

            return marker_array_msg
        except Exception as e:
            self.get_logger().error(f"Error in create_marker_array: {e}")
            return MarkerArray()

def main(args=None):
    # Initialize the rclpy library and the TrackerWithCloudNode
    rclpy.init(args=args)
    node = TrackerWithCloudNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
