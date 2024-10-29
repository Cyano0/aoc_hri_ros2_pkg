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
from std_msgs.msg import Float64
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from human_detection.msg import YoloResultDistance, DistanceArray
from ultralytics_ros.msg import YoloResult
from message_filters import ApproximateTimeSynchronizer, Subscriber
import torch
import cv2
import copy
from ultralytics.engine.results import Boxes, Masks

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.declare_parameter("yolo_model", rclpy.Parameter.Type.STRING)
        self.declare_parameter("input_topic_1",  rclpy.Parameter.Type.STRING)
        self.declare_parameter("input_depth_topic_1", rclpy.Parameter.Type.STRING)
        self.declare_parameter("result_topic_1", rclpy.Parameter.Type.STRING)
        self.declare_parameter("result_image_topic_1", rclpy.Parameter.Type.STRING)
        self.declare_parameter("input_topic_2",  rclpy.Parameter.Type.STRING)
        self.declare_parameter("input_depth_topic_2", rclpy.Parameter.Type.STRING)
        self.declare_parameter("result_topic_2", rclpy.Parameter.Type.STRING)
        self.declare_parameter("result_image_topic_2", rclpy.Parameter.Type.STRING)
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
        self.declare_parameter("camera_parameters_1",rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("camera_parameters_2",rclpy.Parameter.Type.DOUBLE_ARRAY)

        path = get_package_share_directory("ultralytics_ros")
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

        input_topic_1 = (
            self.get_parameter("input_topic_1").get_parameter_value().string_value
        )
        input_depth_topic_1 = (
            self.get_parameter("input_depth_topic_1").get_parameter_value().string_value
        )
        input_topic_2 = (
            self.get_parameter("input_topic_2").get_parameter_value().string_value
        )
        input_depth_topic_2 = (
            self.get_parameter("input_depth_topic_2").get_parameter_value().string_value
        )
        result_topic_1 = (
            self.get_parameter("result_topic_1").get_parameter_value().string_value
        )
        result_image_topic_1 = (
            self.get_parameter("result_image_topic_1").get_parameter_value().string_value
        )
        result_topic_2 = (
            self.get_parameter("result_topic_1").get_parameter_value().string_value
        )
        result_image_topic_2 = (
            self.get_parameter("result_image_topic_1").get_parameter_value().string_value
        )
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                                          depth=1)
        # self.sub = self.create_subscription(Image, topic,
                                    # self.subscriber_callback, qos_profile=qos_policy)
        self.image_sub_1 = Subscriber(self, Image, input_topic_1, qos_profile=qos_policy)
        self.depth_sub_1 = Subscriber(self, Image, input_depth_topic_1, qos_profile=qos_policy)
        self.image_sub_2 = Subscriber(self, Image, input_topic_2, qos_profile=qos_policy)
        self.depth_sub_2 = Subscriber(self, Image, input_depth_topic_2, qos_profile=qos_policy)
       
        self.camera_parameters_1 = self.get_parameter('camera_parameters_1').get_parameter_value().double_array_value
        self.camera_parameters_fx_1 = self.camera_parameters_1[0]
        self.camera_parameters_fy_1 = self.camera_parameters_1[1]
        self.camera_parameters_cx_1 = self.camera_parameters_1[2]
        self.camera_parameters_cy_1 = self.camera_parameters_1[3]
        self.camera_parameters_2 = self.get_parameter('camera_parameters_2').get_parameter_value().double_array_value
        self.camera_parameters_fx_2 = self.camera_parameters_2[0]
        self.camera_parameters_fy_2 = self.camera_parameters_2[1]
        self.camera_parameters_cx_2 = self.camera_parameters_2[2]
        self.camera_parameters_cy_2 = self.camera_parameters_2[3]

        self.ts = ApproximateTimeSynchronizer([self.image_sub_1, self.depth_sub_1,self.image_sub_2, self.depth_sub_2], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.image_callback)

        self.results_pub = self.create_publisher(YoloResultDistance, result_topic_1, 1)
        self.result_image_pub = self.create_publisher(Image, result_image_topic_1, 1)

    def image_callback(self, msg1, msg_depth1, msg2, msg_depth2):
        print("------------------IMAGE RECEIVED----------------")
        
        cv_image1 = self.bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        depth_image1 = self.bridge.imgmsg_to_cv2(msg_depth1, desired_encoding='passthrough')
        cv_image2 = self.bridge.imgmsg_to_cv2(msg2, desired_encoding="bgr8")
        depth_image2 = self.bridge.imgmsg_to_cv2(msg_depth2, desired_encoding='passthrough')
        
        # Create a black bar
        height1, width1, _ = cv_image1.shape
        height2, width2, _ = cv_image2.shape
        black_bar_height = max(height1, height2)
        black_bar = np.zeros((black_bar_height, 10, 3), dtype=np.uint8)  # 10 pixel wide black bar

        # Combine images with a black bar in between
        combined_image = np.hstack((cv_image1, black_bar, cv_image2))
        
        # Combine depth images with a black bar in between
        # combined_depth_height = max(depth_image1.shape[0], depth_image2.shape[0])
        # black_bar_depth = np.zeros((combined_depth_height, 10), dtype=np.uint16)  # 10 pixel wide black bar for depth
        # combined_depth_image = np.hstack((depth_image1, black_bar_depth, depth_image2))
        
    
        # Get parameters
        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        classes = 0  # For person detection only
        tracker = self.get_parameter("tracker").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value or None
        
        # Perform object detection and tracking using the YOLO model
        results = self.model.track(
            source=combined_image,
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
            yolo_result_msg1 = YoloResultDistance()
            yolo_result_image_msg1 = Image()
            yolo_result_msg2 = YoloResultDistance()
            yolo_result_image_msg2 = Image()
            # print(results)
            # print('----------------------------')
            # print(results[0].boxes)
            left_results = copy.deepcopy(results)
            right_results = copy.deepcopy(results)

            # Filter and adjust bounding boxes for left and right results
            left_data, right_data = [], []
            left_masks, right_masks = [], []

            for i, bbox in enumerate(results[0].boxes.xywh):  # Access the bounding boxes
                x_center, y_center, width, height = bbox
                cls = results[0].boxes.cls[i]
                conf = results[0].boxes.conf[i]
                data = results[0].boxes.data[i].clone()  # Ensure data is tensor-based
                mask = results[0].masks.data[i] if hasattr(results[0], 'masks') and results[0].masks is not None else None

                if x_center - width / 2 < width1:  # Detection in the left image
                    left_data.append(data)
                    if mask is not None:
                        mask_np = mask.cpu().numpy()
                        # Adjust the mask x-coordinates
                        adjusted_mask = np.zeros_like(mask_np)
                        adjusted_mask[:, :width1] = mask_np[:, :width1]
                        mask_resized = cv2.resize(adjusted_mask, (depth_image1.shape[1], depth_image1.shape[0]), interpolation=cv2.INTER_NEAREST)
                        left_masks.append(torch.tensor(mask_resized))
                else:  # Detection in the right image
                    # Adjust x_center for the right image
                    x_center -= (width1 + 10)
                    data[0] -= (width1 + 10)
                    right_data.append(data)
                    if mask is not None:
                        mask_np = mask.cpu().numpy()
                        # Adjust the mask x-coordinates
                        adjusted_mask = np.zeros_like(mask_np)
                        adjusted_mask[:, :mask_np.shape[1] - (width1 + 10)] = mask_np[:, (width1 + 10):]
                        mask_resized = cv2.resize(adjusted_mask, (depth_image2.shape[1], depth_image2.shape[0]), interpolation=cv2.INTER_NEAREST)
                        right_masks.append(torch.tensor(mask_resized))


            # Update left_results
            if left_data:
                left_results[0].boxes.data = torch.stack(left_data)
                if left_masks:
                    left_results[0].masks.data = torch.stack(left_masks)
            else:
                left_results[0].boxes.data = torch.empty((0, results[0].boxes.data.shape[1]))  # Empty tensor with the correct shape

            # Update right_results
            if right_data:
                right_results[0].boxes.data = torch.stack(right_data)
                if right_masks:
                    right_results[0].masks.data = torch.stack(right_masks)
            else:
                right_results[0].boxes.data = torch.empty((0, results[0].boxes.data.shape[1]))  # Empty tensor with the correct shape


            # Update the images in the results
            left_results[0].orig_img = cv_image1
            right_results[0].orig_img = cv_image2

            # Create and populate YoloResultDistance messages for both parts
            yolo_result_image_msg1, yolo_result_msg1.detections, yolo_result_msg1.distances = self.create_result(left_results, depth_image1, image_loc=1)
            yolo_result_image_msg2, yolo_result_msg2.detections, yolo_result_msg2.distances = self.create_result(right_results, depth_image2, image_loc=2)
            
            yolo_result_msg1.header = msg1.header
            yolo_result_msg2.header = msg2.header
            yolo_result_image_msg1.header = msg1.header
            yolo_result_image_msg2.header = msg2.header
            
            if self.use_segmentation:
                yolo_result_msg1.masks = self.create_segmentation_masks(left_results)
                yolo_result_msg2.masks = self.create_segmentation_masks(right_results)
            
            # Publish results
            self.results_pub.publish(yolo_result_msg1)
            self.result_image_pub.publish(yolo_result_image_msg1)
            self.results_pub.publish(yolo_result_msg2)
            self.result_image_pub.publish(yolo_result_image_msg2)
        
        # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # depth_image = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding='passthrough')
        # conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        # iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        # max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        # classes = (
        #     self.get_parameter("classes").get_parameter_value().integer_array_value
        # )
        # print("------------------SIZE----------------",depth_image.shape)
        # classes = 0 #For person detection only
        # tracker = self.get_parameter("tracker").get_parameter_value().string_value
        # device = self.get_parameter("device").get_parameter_value().string_value or None
        # results = self.model.track(
        #     source=cv_image,
        #     conf=conf_thres,
        #     iou=iou_thres,
        #     max_det=max_det,
        #     classes=classes,
        #     tracker=tracker,
        #     device=device,
        #     verbose=False,
        #     retina_masks=True,
        # )

        # if results is not None:
        #     yolo_result_msg = YoloResultDistance()
        #     yolo_result_image_msg = Image()
        #     yolo_result_msg.header = msg.header
        #     yolo_result_image_msg.header = msg.header
        #     yolo_result_image_msg, yolo_result_msg.detections, yolo_result_msg.distances = self.create_result(results,depth_image)
        #     if self.use_segmentation:
        #         yolo_result_msg.masks = self.create_segmentation_masks(results)
        #     self.results_pub.publish(yolo_result_msg)
        #     self.result_image_pub.publish(yolo_result_image_msg)

    # funtion used to get depth imformation
    def create_distance(self, bbox, depth_image, image_loc=1):
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
                if not np.isnan(depth_image[a][b]) and (-10 < depth_image[a][b] <10):
                    counter = counter + 1
                    sum_dis = sum_dis + depth_image[a][b]
        if counter == 0:
            ave_dis = float('nan') 
        else:
            ave_dis = sum_dis/counter
        if image_loc == 1:
            X,Y,Z = self.get_3d_point(x,y, ave_dis, self.camera_parameters_fx_1, self.camera_parameters_fy_1, self.camera_parameters_cx_1, self.camera_parameters_cy_1)
        else:        
            X,Y,Z = self.get_3d_point(x,y, ave_dis, self.camera_parameters_fx_2, self.camera_parameters_fy_2, self.camera_parameters_cx_2, self.camera_parameters_cy_2)
        return X, Y, Z
    
    def create_distance_with_mask(self, bbox, depth_image, mask, image_loc=1):
        masked_depth_array = depth_image * mask
        filter = ~np.isnan(masked_depth_array) & (masked_depth_array != 0) &(masked_depth_array <10)&(masked_depth_array >-10)
        average_dis = np.mean(masked_depth_array[filter])
        print(average_dis)
        if image_loc == 1:        
            X,Y,Z = self.get_3d_point(round(float(bbox[0])),round(float(bbox[1])), average_dis, self.camera_parameters_fx_1, self.camera_parameters_fy_1, self.camera_parameters_cx_1, self.camera_parameters_cy_1)
        else:
            X,Y,Z = self.get_3d_point(round(float(bbox[0])),round(float(bbox[1])), average_dis, self.camera_parameters_fx_2, self.camera_parameters_fy_2, self.camera_parameters_cx_2, self.camera_parameters_cy_2)
    
        return X,Y,Z

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
    def create_result(self, results, depth_image, image_loc=1):
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
        plotted_image_distance = im.fromarray(plotted_image)
        # if uses mask
        if hasattr(results[0], "masks") and (results[0].masks is not None):
            print('-------------------------------------has masks')
            distance_store = []
            mask = results[0].masks.data
            for bbox, cls, conf, msk in zip(bounding_box, classes, confidence_score, mask):
                objectname = results[0].names.get(int(cls))
                if objectname == 'person': # where int(cls) == 0
                    detection = Detection2D()
                    detection.bbox.center.position.x = float(bbox[0])
                    detection.bbox.center.position.y = float(bbox[1])
                    detection.bbox.size_x = float(bbox[2])
                    detection.bbox.size_y = float(bbox[3])
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = objectname
                    hypothesis.hypothesis.score = float(conf)
                    # print('-----------------MSK----------------', msk)
                    msk_np = (
                            np.squeeze(msk.to("cpu").detach().numpy()).astype(
                                np.uint8
                            )
                            # * 255
                        )
                    distance_value_x, distance_value_y, distance_value_z = self.create_distance_with_mask(bbox, depth_image, msk_np, image_loc)
                    print(distance_value_x, distance_value_y, distance_value_z)
                    distance_store.append(np.float32(distance_value_x).item())
                    distance_store.append(np.float32(distance_value_y).item())
                    distance_store.append(np.float32(distance_value_z).item())
                    # print("-------------------Sent distance----------------:",distance_store)
                    np.append(distances_msg.poses, distance_store)
                    detection.results.append(hypothesis)    
                    ImageDraw.Draw(plotted_image_distance).text((detection.bbox.center.position.x, detection.bbox.center.position.y), 
                                                                (f"{np.float32(distance_value_x):.2f}"+ ", " + f"{np.float32(distance_value_y):.2f}"+ ", " + f"{np.float32(distance_value_z):.2f}"), 
                                                                font = ImageFont.load_default(),fill =(255, 0, 0))
        else:
            for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
                objectname = results[0].names.get(int(cls))
                if objectname == 'person': # where int(cls) == 0
                    detection = Detection2D()
                    detection.bbox.center.position.x = float(bbox[0])
                    detection.bbox.center.position.y = float(bbox[1])
                    detection.bbox.size_x = float(bbox[2])
                    detection.bbox.size_y = float(bbox[3])
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = objectname
                    hypothesis.hypothesis.score = float(conf)
                    distance_value_x, distance_value_y, distance_value_z = self.create_distance(bbox, depth_image, image_loc)
                    distance_store.append(np.float32(distance_value_x).item())
                    distance_store.append(np.float32(distance_value_y).item())
                    distance_store.append(np.float32(distance_value_z).item())
                    np.append(distances_msg.poses, distance_store)
                    detection.results.append(hypothesis)    
                    ImageDraw.Draw(plotted_image_distance).text((detection.bbox.center.position.x, detection.bbox.center.position.y), 
                                                                (f"{np.float32(distance_value_x):.2f}"+ ", " + f"{np.float32(distance_value_y):.2f}"+ ", " + f"{np.float32(distance_value_z):.2f}"), 
                                                                font = ImageFont.load_default(),fill =(255, 0, 0))
        plotted_image_distance =np.array(plotted_image_distance)
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image_distance, encoding="bgr8")
        
        return result_image_msg, detections_msg, distances_msg


    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            cls_list = np.array(result.boxes.cls.tolist())
            humanid_ii = np.where(cls_list == 0.0)[0]
            if len(humanid_ii) > 0:
                for i in humanid_ii:
                    if hasattr(result, "masks") and (result.masks[i] is not None):
                        mask_tensor = result.masks[i]
                        # print('-----------------MASK TENSOR----------------', mask_tensor)
                        mask_numpy = (
                            np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                                np.uint8
                            )
                            * 255
                        )
                        mask_image_msg = self.bridge.cv2_to_imgmsg(
                            mask_numpy, encoding="mono8"
                        )
                        masks_msg.append(mask_image_msg)
        return masks_msg


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

