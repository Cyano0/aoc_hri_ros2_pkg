#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from mmdet3d.apis import LidarDet3DInferencer
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import open3d as o3d

class MMDet3DNode(Node):
    def __init__(self):
        super().__init__('mmdet3d_node')
        # self.subscription = self.create_subscription(
        #     PointCloud2,
        #     '/livox/lidar_192_168_1_130',
        #     self.listener_callback,
        #     10)
        # self.subscription

        # Define QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Or RELIABLE
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Adjust the depth as needed
        )
        
        # Create subscription with the QoS profile
        self.subscription = self.create_subscription(
            PointCloud2,
            '/back_lidar/points',
            self.listener_callback,
            qos_profile)

        self.inferencer = LidarDet3DInferencer(
            model='/home/developer/aoc_strawberry_scenario_ws/src/external_packages/mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
            weights='/home/developer/aoc_strawberry_scenario_ws/src/external_packages/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth',
            device='cuda:0'
        )

    def listener_callback(self, msg):
        points = self.convert_pc2_to_np(msg)
        if points is not None:
            # Print the shape and type of the points array
            print(f"Points shape: {points.shape}, dtype: {points.dtype}")
            inputs = dict(points=points)
            # Print the shape and type of the inputs dictionary values
            for key, value in inputs.items():
                print(f"Input {key} shape: {value.shape}, dtype: {value.dtype}")
            # inputs = dict(points=points)
            results = self.inferencer(inputs)
            print(f"Points: {points[115:120]}")  # Print first 5 points for debugging
            # print(f"Results: {results['predictions'][0]['bboxes_3d'][:5]}")  # Print first 5 bounding boxes for debuggingprint(results) # {'predictions': [{'labels_3d': [], 'scores_3d': [], 'bboxes_3d': [], 'box_type_3d': 'LiDAR'}], 'visualization': []}
            print(f"Results: {results}")
            self.visualize_results(points, results)
            # self.inferencer.visualize(
            #     inputs=[inputs],
            #     preds=results, 
            #     show=True,  # Set to True to display the results
            #     wait_time=1  # Adjust wait time as needed
            # )

    # def convert_pc2_to_np(self, msg):
    #     cloud_arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
    #     if cloud_arr.size == 0:
    #         return None
    #     return cloud_arr
    # def convert_pc2_to_np(self, msg):
    #     # Extract the points from the PointCloud2 message
    #     cloud_arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
    #     # Convert the structured array to a simple array with float32 type
    #     points = np.zeros((cloud_arr.shape[0], 3), dtype=np.float32)
    #     points[:, 0] = cloud_arr['x']
    #     points[:, 1] = cloud_arr['y']
    #     points[:, 2] = cloud_arr['z']
    #     if points.size == 0:
    #         return None
    #     return points
    # def convert_pc2_to_np(self, msg):
    #     cloud_arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))
    #     # Extract the 'x', 'y', 'z' fields and convert to float32
    #     points = np.zeros((cloud_arr.shape[0], 3), dtype=np.float32)
    #     points[:, 0] = cloud_arr['x'].astype(np.float32)
    #     points[:, 1] = cloud_arr['y'].astype(np.float32)
    #     points[:, 2] = cloud_arr['z'].astype(np.float32)
    #     if points.size == 0:
    #         return None
    #     return points
    # def convert_pc2_to_np(self, msg):
    #     cloud_arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))
    #     points = np.zeros((cloud_arr.shape[0], 4), dtype=np.float32)
    #     points[:, 0] = cloud_arr['x'].astype(np.float32)
    #     points[:, 1] = cloud_arr['y'].astype(np.float32)
    #     points[:, 2] = cloud_arr['z'].astype(np.float32)
    #     points[:, 3] = cloud_arr['intensity'].astype(np.float32)
    #     if points.size == 0:
    #         return None
    #     return points
    def convert_pc2_to_np(self, msg):
        # Read points with required fields. If 'intensity' is not available, initialize it with zeros.
        cloud_arr = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))))

        # Check if the intensity field is available
        if 'intensity' not in cloud_arr.dtype.names:
            # Initialize intensity with zeros if not available
            intensity = np.zeros(cloud_arr.shape[0], dtype=np.float32)
        else:
            intensity = cloud_arr['intensity'].astype(np.float32)

        # Create a points array with 4 features: x, y, z, intensity
        points = np.zeros((cloud_arr.shape[0], 4), dtype=np.float32)
        points[:, 0] = cloud_arr['x'].astype(np.float32)
        points[:, 1] = cloud_arr['y'].astype(np.float32)
        points[:, 2] = cloud_arr['z'].astype(np.float32)
        points[:, 3] = intensity

        if points.size == 0:
            return None
        return points

    def visualize_results(self, points, results):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Color the points based on their labels
        labels = results['predictions'][0]['labels_3d']
        colors = np.zeros((points.shape[0], 3))  # Default color is black

        # Assign colors based on labels
        self.color_map = {
            0: [1, 0, 0],  # Red for label 0
            1: [0, 1, 0],  # Green for label 1
            2: [0, 0, 1]   # Blue for label 2
        }

        for i, label in enumerate(labels):
            colors[i] = self.color_map.get(label, [1, 1, 1])  # Default to white if label not found

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize bounding boxes
        bbox_data = results['predictions'][0]['bboxes_3d']
        # for bbox in bbox_data:
        #     center = bbox[:3]
        #     size = bbox[3:6]
        #     rotation = bbox[6]
        #     box = self.create_bounding_box(center, size, rotation)
        #     box.color = (1, 0, 0)  # Red bounding box
        #     o3d.visualization.draw_geometries([pcd, box])
        bbox_labels = results['predictions'][0]['labels_3d']
        geometries = [pcd]
        print("Bounding box data:", bbox_data)

        for bbox, label in zip(bbox_data, bbox_labels):
            box = self.create_bounding_box(bbox)
            box.color = self.color_map.get(label, [1, 1, 1])  # Use the label color
            geometries.append(box)


        # Create a visualizer window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.run()
        vis.destroy_window()

    # def create_bounding_box(self, center, size, rotation):
    #     bbox = o3d.geometry.OrientedBoundingBox(center, o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, rotation)), size)
    #     return bbox
    def create_bounding_box(self, bbox):
        center = bbox[:3]
        size = bbox[3:6]
        rotation = bbox[6]
        # Adjust the z-coordinate to account for the height of the bounding box
        center[2] += size[2] / 2
        print(f"Center: {center}, Size: {size}, Rotation: {rotation}")  # Debugging statement
        # bbox_3d = o3d.geometry.OrientedBoundingBox(center, o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, rotation)), size)
        bbox_3d = o3d.geometry.OrientedBoundingBox()
        bbox_3d.center = center
        bbox_3d.extent = size
        bbox_3d.R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, 0, rotation])
        return bbox_3d

def main(args=None):
    rclpy.init(args=args)
    mmdet3d_node = MMDet3DNode()
    rclpy.spin(mmdet3d_node)
    mmdet3d_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()