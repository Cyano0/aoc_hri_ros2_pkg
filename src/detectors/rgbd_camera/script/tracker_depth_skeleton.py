#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv_bridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import ImageDraw, ImageFont
from PIL import Image as im
from ultralytics import YOLO
# from std_msgs.msg import Float64
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from human_detection.msg import YoloResultDistance, DistanceArray, HumanInfo, HumanInfoArray
from ultralytics_ros.msg import YoloResult
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time
import math

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.declare_parameter("yolo_model", rclpy.Parameter.Type.STRING)
        self.declare_parameter("input_topic",  rclpy.Parameter.Type.STRING)
        self.declare_parameter("input_depth_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("result_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("result_image_topic", rclpy.Parameter.Type.STRING)
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", list(range(80)))
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 1)
        self.declare_parameter("result_font_size", 1)
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)
        self.declare_parameter("camera_parameters", rclpy.Parameter.Type.DOUBLE_ARRAY)

        path = get_package_share_directory("ultralytics_ros")
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()
        self.use_skeleton = yolo_model.endswith("-pose.pt")

        input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        input_depth_topic = (
            self.get_parameter("input_depth_topic").get_parameter_value().string_value
        )
        result_topic = (
            self.get_parameter("result_topic").get_parameter_value().string_value
        )
        result_image_topic = (
            self.get_parameter("result_image_topic").get_parameter_value().string_value
        )
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        self.image_sub = Subscriber(self, Image, input_topic, qos_profile=qos_policy)
        self.depth_sub = Subscriber(self, Image, input_depth_topic, qos_profile=qos_policy)
        
        self.camera_parameters = self.get_parameter('camera_parameters').get_parameter_value().double_array_value
        self.camera_parameters_fx = self.camera_parameters[0]
        self.camera_parameters_fy = self.camera_parameters[1]
        self.camera_parameters_cx = self.camera_parameters[2]
        self.camera_parameters_cy = self.camera_parameters[3]

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.results_pub = self.create_publisher(HumanInfoArray, result_topic, qos_policy)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 1)

        self.previous_positions = {}  # Dictionary to store previous positions and timestamps

    def image_callback(self, msg, msg_depth):
        print("------------------IMAGE RECEIVED----------------")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')
        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        classes = 0 #For person detection only
        tracker = self.get_parameter("tracker").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value or None
        results = self.model.track(
            source=cv_image,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            classes=classes,
            tracker=tracker,
            device=device,
            verbose=False,
            retina_masks=True,
        )

        if results is not None:
            # print(results[0].keypoints)
            yolo_result_msg = YoloResultDistance()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            human_array_msg = HumanInfoArray()
            human_array_msg.header = msg.header

            yolo_result_image_msg, yolo_result_msg.detections, yolo_result_msg.distances, human_infos = self.create_result(results, depth_image)
            human_array_msg.human_infos = human_infos

            self.results_pub.publish(human_array_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    # funtion used to get depth imformation
    def create_distance(self, bbox, depth_image):
        x = round(float(bbox[0]))
        y = round(float(bbox[1]))
        width = round(float(bbox[2]))
        height = round(float(bbox[3]))
        lower_bound_x = max(0, x-round(width/6))
        upper_bound_x = min(depth_image.shape[1]-1, x+round(width/6))
        lower_bound_y = max(0, y-round(height/6))
        upper_bound_y = min(depth_image.shape[0]-1, y+round(height/6))
        counter = 0
        sum_dis = 0
        for a in range(lower_bound_y, upper_bound_y):
            for b in range(lower_bound_x, upper_bound_x):
                if not np.isnan(depth_image[a][b]):
                    counter = counter + 1
                    sum_dis = sum_dis + depth_image[a][b]
        if counter == 0:
            ave_dis = float('nan') 
        else:
            ave_dis = sum_dis/counter
        X, Y, Z, X_tr, Y_tr, Z_tr, X_bl, Y_bl, Z_trb = self.get_bounding_box_coordinates(x,y, width, height, ave_dis, self.camera_parameters_fx, self.camera_parameters_fy, self.camera_parameters_cx, self.camera_parameters_cy)
        return X, Y, Z, X_tr, Y_tr, Z_tr, X_bl, Y_bl, Z_trb

    def create_distance_with_skeleton(self, bbox, depth_image, centre_x, centre_y):
        x = round(float(centre_x))
        y = round(float(centre_y))
        width = round(float(bbox[2]))
        height = round(float(bbox[3]))
        lower_bound_x = max(0, x-round(width/10))
        upper_bound_x = min(depth_image.shape[1]-1, x+round(width/6))
        lower_bound_y = max(0, y-round(height/10))
        upper_bound_y = min(depth_image.shape[0]-1, y+round(height/6))
        counter = 0
        sum_dis = 0
        for a in range(lower_bound_y, upper_bound_y):
            for b in range(lower_bound_x, upper_bound_x):
                if not np.isnan(depth_image[a][b]):
                    counter = counter + 1
                    sum_dis = sum_dis + depth_image[a][b]
        if counter == 0:
            ave_dis = float('nan') 
        else:
            ave_dis = sum_dis/counter
        X, Y, Z, X_tr, Y_tr, Z_tr, X_bl, Y_bl, Z_trb = self.get_bounding_box_coordinates(x,y, width, height, ave_dis, self.camera_parameters_fx, self.camera_parameters_fy, self.camera_parameters_cx, self.camera_parameters_cy)
        return X, Y, Z, X_tr, Y_tr, Z_tr, X_bl, Y_bl, Z_trb

    def get_3d_point(self, u, v, Z, fx=977.1935424804688, fy=977.1935424804688, cx=649.9141845703125, cy=353.80377197265625):
        """
        This function calculates the 3D world point (X, Y, Z) from image coordinates (u, v),
        depth (Z), and camera intrinsic parameters.
        Args:
            u (float): Image x-coordinate.
            v (float): Image y-coordinate.
            Z (float): Depth value of the point.
            fx (float): Focal length in the x-direction.
            fy (float): Focal length in the y-direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
        Returns:
            tuple: A tuple containing the 3D world point coordinates (X, Y, Z).
        """
        X = (( u - cx) * Z) / (fx)
        Y = (( v - cy) * Z) / (fy)

        return X, Y, Z

    def get_bounding_box_coordinates(self, u_center, v_center, width, height, Z, fx=977.1935424804688, fy=977.1935424804688, cx=649.9141845703125, cy=353.80377197265625):
        """
        This function calculates the 3D world coordinates of the bounding box corners
        from the center coordinates, width, height, and depth.
        Args:
            u_center (float): Image x-coordinate of the bounding box center.
            v_center (float): Image y-coordinate of the bounding box center.
            width (float): Width of the bounding box in pixels.
            height (float): Height of the bounding box in pixels.
            Z (float): Depth value of the bounding box.
            fx (float): Focal length in the x-direction.
            fy (float): Focal length in the y-direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
        Returns:
            dict: the 3D world coordinates of the pose (X, Y, Z) and bounding box corners (Bottom-left-front corner & Top-right-back corner).
        """
        half_width = width / 2
        half_height = height / 2

        # Center
        X, Y, Z = self.get_3d_point(u_center, v_center, Z, fx, fy, cx, cy)

        # Top-right corner - front
        u_tr = u_center + half_width
        v_tr = v_center - half_height
        X_tr, Y_tr, Z_tr = self.get_3d_point(u_tr, v_tr, Z, fx, fy, cx, cy)

        # Bottom-left corner -front
        u_bl = u_center - half_width
        v_bl = v_center + half_height
        X_bl, Y_bl, Z_bl = self.get_3d_point(u_bl, v_bl, Z, fx, fy, cx, cy)

        # Top-right corner - back
        # Assuming the bbox in Z direction is half the size of it in X direction
        Z_diff = 0.5*abs(X_bl - X_tr)
        Z_trb = Z_tr + Z_diff

        return X, Y, Z, X_tr, Y_tr, Z_tr, X_bl, Y_bl, Z_trb

    def create_result_image(self, results):
        result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
        result_line_width = (
            self.get_parameter("result_line_width").get_parameter_value().integer_value
        )
        result_font_size = (
            self.get_parameter("result_font_size").get_parameter_value().integer_value
        )
        result_font = (
            self.get_parameter("result_font").get_parameter_value().string_value
        )
        result_labels = (
            self.get_parameter("result_labels").get_parameter_value().bool_value
        )
        result_boxes = (
            self.get_parameter("result_boxes").get_parameter_value().bool_value
        )
        plotted_image = results[0].plot(
            conf=result_conf,
            line_width=result_line_width,
            font_size=result_font_size,
            font=result_font,
            labels=result_labels,
            boxes=result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")

        return result_image_msg

    # A combination of create_result_image() and create_detection array(), using information from both so one function makes it easier.
    def create_result(self, results, depth_image):
        result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
        result_line_width = (
            self.get_parameter("result_line_width").get_parameter_value().integer_value
        )
        result_font_size = (
            self.get_parameter("result_font_size").get_parameter_value().integer_value
        )
        result_font = (
            self.get_parameter("result_font").get_parameter_value().string_value
        )
        result_labels = (
            self.get_parameter("result_labels").get_parameter_value().bool_value
        )
        result_boxes = (
            self.get_parameter("result_boxes").get_parameter_value().bool_value
        )
        plotted_image = results[0].plot(
            conf=result_conf,
            line_width=result_line_width,
            font_size=result_font_size,
            font=result_font,
            labels=result_labels,
            boxes=result_boxes,
        )
        detections_msg = Detection2DArray()
        distances_msg = DistanceArray()
        distance_store = []
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        tracking_id = results[0].boxes.id
        plotted_image_distance = im.fromarray(plotted_image)
        human_infos = []

        current_time = time.time()

        # If Tracking:
        if result_labels == True and tracking_id is not None:
            if hasattr(results[0], "keypoints") and (results[0].keypoints is not None):
                print('-------------------has skeleton------------------')
                keypoints = results[0].keypoints
                for bbox, cls, conf, kpt, tk_id in zip(bounding_box, classes, confidence_score, keypoints, tracking_id):
                    objectname = results[0].names.get(int(cls))
                    if objectname == 'person':
                        detection = Detection2D()
                        detection.bbox.center.position.x = float(bbox[0])
                        detection.bbox.center.position.y = float(bbox[1])
                        detection.bbox.size_x = float(bbox[2])
                        detection.bbox.size_y = float(bbox[3])
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = objectname
                        hypothesis.hypothesis.score = float(conf)
                        keypoint_confidence = kpt.conf.cpu().numpy()
                        keypoint_locs = kpt.xy.cpu().numpy()
                        total_x = 0
                        total_y = 0
                        counter_key = 0
                        for keypoint_index in (5, 6, 11, 12):  # shoulders and hips
                            if keypoint_confidence[0][keypoint_index] > 0.6 and keypoint_locs[0][keypoint_index][0] > 0 and keypoint_locs[0][keypoint_index][1] > 0:
                                total_x += keypoint_locs[0][keypoint_index][0]
                                total_y += keypoint_locs[0][keypoint_index][1]
                                counter_key += 1
                        if counter_key > 0:
                            centre_x = total_x / counter_key
                            centre_y = total_y / counter_key
                            distance_value_x, distance_value_y, distance_value_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = self.create_distance_with_skeleton(bbox, depth_image, centre_x, centre_y)
                        else:
                            distance_value_x, distance_value_y, distance_value_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = self.create_distance(bbox, depth_image)
                        
                        detection.results.append(hypothesis)
                        human = HumanInfo()
                        human.id = str(tk_id.item())
                        human.pose = [np.float32(distance_value_x).item(), np.float32(distance_value_y).item(), np.float32(distance_value_z).item()]
                        human.min = [np.float32(bbox_min_x).item(), np.float32(bbox_min_y).item(), np.float32(bbox_min_z).item()]
                        human.max = [np.float32(bbox_max_x).item(), np.float32(bbox_max_y).item(), np.float32(bbox_max_z).item()]

                        if tk_id.item() in self.previous_positions:
                            prev_x, prev_z, prev_time = self.previous_positions[tk_id.item()]
                            time_diff = current_time - prev_time
                            if time_diff > 0:
                                velocity = math.sqrt((distance_value_x - prev_x) ** 2 + (distance_value_z - prev_z) ** 2) / time_diff
                                orientation = math.degrees(math.atan2(distance_value_x - prev_x, distance_value_z - prev_z)) % 360
                                human.velocity = [velocity]  # Set as list with one float
                                human.orientation = [orientation]  # Set as list with one float
                        else:
                            human.velocity = [np.nan]  # Set as list with one float
                            human.orientation = [np.nan]  # Set as list with one float

                        self.previous_positions[tk_id.item()] = (distance_value_x, distance_value_z, current_time)
                        human_infos.append(human)

                        ImageDraw.Draw(plotted_image_distance).text(
                            (detection.bbox.center.position.x, detection.bbox.center.position.y), 
                            (f"{np.float32(distance_value_x):.2f}, {np.float32(distance_value_y):.2f}, {np.float32(distance_value_z):.2f}"), 
                            font=ImageFont.load_default(), 
                            fill=(255, 0, 0)
                        )
            else:
                for bbox, cls, conf, tk_id in zip(bounding_box, classes, confidence_score, tracking_id):
                    objectname = results[0].names.get(int(cls))
                    if objectname == 'person':
                        detection = Detection2D()
                        detection.bbox.center.position.x = float(bbox[0])
                        detection.bbox.center.position.y = float(bbox[1])
                        detection.bbox.size_x = float(bbox[2])
                        detection.bbox.size_y = float(bbox[3])
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = objectname
                        hypothesis.hypothesis.score = float(conf)
                        distance_value_x, distance_value_y, distance_value_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = self.create_distance(bbox, depth_image)

                        detection.results.append(hypothesis)
                        human = HumanInfo()
                        human.id = str(tk_id.item())
                        human.pose = [np.float32(distance_value_x).item(), np.float32(distance_value_y).item(), np.float32(distance_value_z).item()]
                        human.min = [np.float32(bbox_min_x).item(), np.float32(bbox_min_y).item(), np.float32(bbox_min_z).item()]
                        human.max = [np.float32(bbox_max_x).item(), np.float32(bbox_max_y).item(), np.float32(bbox_max_z).item()]

                        if tk_id.item() in self.previous_positions:
                            prev_x, prev_z, prev_time = self.previous_positions[tk_id.item()]
                            time_diff = current_time - prev_time
                            if time_diff > 0:
                                velocity = math.sqrt((distance_value_x - prev_x) ** 2 + (distance_value_z - prev_z) ** 2) / time_diff
                                orientation = math.degrees(math.atan2(distance_value_x - prev_x, distance_value_z - prev_z)) % 360
                                human.velocity = [velocity]  # Set as list with one float
                                human.orientation = [orientation]  # Set as list with one float
                        else:
                            human.velocity = [np.nan]  # Set as list with one float
                            human.orientation = [np.nan]  # Set as list with one float

                        self.previous_positions[tk_id.item()] = (distance_value_x, distance_value_z, current_time)
                        human_infos.append(human)

                        ImageDraw.Draw(plotted_image_distance).text(
                            (detection.bbox.center.position.x, detection.bbox.center.position.y), 
                            (f"{np.float32(distance_value_x):.2f}, {np.float32(distance_value_y):.2f}, {np.float32(distance_value_z):.2f}"), 
                            font=ImageFont.load_default(), 
                            fill=(255, 0, 0)
                        )
        else:
            if hasattr(results[0], "keypoints") and (results[0].keypoints is not None):
                print('-------------------has skeleton------------------')
                keypoints = results[0].keypoints
                for bbox, cls, conf, kpt in zip(bounding_box, classes, confidence_score, keypoints):
                    objectname = results[0].names.get(int(cls))
                    if objectname == 'person':
                        detection = Detection2D()
                        detection.bbox.center.position.x = float(bbox[0])
                        detection.bbox.center.position.y = float(bbox[1])
                        detection.bbox.size_x = float(bbox[2])
                        detection.bbox.size_y = float(bbox[3])
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = objectname
                        hypothesis.hypothesis.score = float(conf)
                        keypoint_confidence = kpt.conf.cpu().numpy()
                        keypoint_locs = kpt.xy.cpu().numpy()
                        total_x = 0
                        total_y = 0
                        counter_key = 0
                        for keypoint_index in (5, 6, 11, 12):  # shoulders and hips
                            if keypoint_confidence[0][keypoint_index] > 0.6 and keypoint_locs[0][keypoint_index][0] > 0 and keypoint_locs[0][keypoint_index][1] > 0:
                                total_x += keypoint_locs[0][keypoint_index][0]
                                total_y += keypoint_locs[0][keypoint_index][1]
                                counter_key += 1
                        if counter_key > 0:
                            centre_x = total_x / counter_key
                            centre_y = total_y / counter_key
                            distance_value_x, distance_value_y, distance_value_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = self.create_distance_with_skeleton(bbox, depth_image, centre_x, centre_y)
                        else:
                            distance_value_x, distance_value_y, distance_value_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = self.create_distance(bbox, depth_image)

                        detection.results.append(hypothesis)
                        human = HumanInfo()
                        human.pose = [np.float32(distance_value_x).item(), np.float32(distance_value_y).item(), np.float32(distance_value_z).item()]
                        human.min = [np.float32(bbox_min_x).item(), np.float32(bbox_min_y).item(), np.float32(bbox_min_z).item()]
                        human.max = [np.float32(bbox_max_x).item(), np.float32(bbox_max_y).item(), np.float32(bbox_max_z).item()]
                        human.id = ''
                        human.velocity = [np.nan]  # Set as list with one float
                        human.orientation = [np.nan]  # Set as list with one float

                        human_infos.append(human)

                        ImageDraw.Draw(plotted_image_distance).text(
                            (detection.bbox.center.position.x, detection.bbox.center.position.y), 
                            (f"{np.float32(distance_value_x):.2f}, {np.float32(distance_value_y):.2f}, {np.float32(distance_value_z):.2f}"), 
                            font=ImageFont.load_default(), 
                            fill=(255, 0, 0)
                        )
            else:
                for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
                    objectname = results[0].names.get(int(cls))
                    if objectname == 'person':
                        detection = Detection2D()
                        detection.bbox.center.position.x = float(bbox[0])
                        detection.bbox.center.position.y = float(bbox[1])
                        detection.bbox.size_x = float(bbox[2])
                        detection.bbox.size_y = float(bbox[3])
                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.class_id = objectname
                        hypothesis.hypothesis.score = float(conf)
                        distance_value_x, distance_value_y, distance_value_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z = self.create_distance(bbox, depth_image)

                        detection.results.append(hypothesis)
                        human = HumanInfo()
                        human.pose = [np.float32(distance_value_x).item(), np.float32(distance_value_y).item(), np.float32(distance_value_z).item()]
                        human.min = [np.float32(bbox_min_x).item(), np.float32(bbox_min_y).item(), np.float32(bbox_min_z).item()]
                        human.max = [np.float32(bbox_max_x).item(), np.float32(bbox_max_y).item(), np.float32(bbox_max_z).item()]
                        human.id = ''
                        human.velocity = [np.nan]  # Set as list with one float
                        human.orientation = [np.nan]  # Set as list with one float

                        human_infos.append(human)

                        ImageDraw.Draw(plotted_image_distance).text(
                            (detection.bbox.center.position.x, detection.bbox.center.position.y), 
                            (f"{np.float32(distance_value_x):.2f}, {np.float32(distance_value_y):.2f}, {np.float32(distance_value_z):.2f}"), 
                            font=ImageFont.load_default(), 
                            fill=(255, 0, 0)
                        )

        plotted_image_distance = np.array(plotted_image_distance)
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image_distance, encoding="bgr8")

        return result_image_msg, detections_msg, distances_msg, human_infos

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)

if __name__ == "__main__":
    main()