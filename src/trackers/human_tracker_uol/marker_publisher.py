#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from human_rgbd_det_uol.msg import HumanInfoArray  # Adjust the package and message names accordingly
from human_tracker_uol.trackers.tracker_KMFilter import KalmanFilterTracker  # Import the tracker
import math
from builtin_interfaces.msg import Duration  # Import for setting the marker lifetime

def sanitize_value(value):
    if math.isnan(value) or math.isinf(value):
        return 0.0  # Replace with a default value
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
                ('use_tracker', False),
                ('tracker_dt', 0.1),
                ('tracker_process_noise', 1e-5),
                ('tracker_measurement_noise', 1e-1),
                ('publish_topic', 'human_bounding_box')
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
        self.use_tracker = self.get_parameter('use_tracker').get_parameter_value().bool_value
        self.tracker_dt = self.get_parameter('tracker_dt').get_parameter_value().double_value
        self.tracker_process_noise = self.get_parameter('tracker_process_noise').get_parameter_value().double_value
        self.tracker_measurement_noise = self.get_parameter('tracker_measurement_noise').get_parameter_value().double_value

        # Define QoS policy
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            HumanInfoArray,
            self.subscription_topic,
            self.listener_callback,
            qos_profile=qos_policy  # Use the custom QoS policy
        )
        self.publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value
        self.publisher = self.create_publisher(MarkerArray, self.publish_topic, qos_profile=qos_policy)
        self.subscription  # prevent unused variable warning

        self.existing_ids = set()
        self.trackers = {}  # Trackers for each ID

    def listener_callback(self, msg):
        marker_array = MarkerArray()
        self.publisher.publish(marker_array)
        self.existing_ids = set()
        for idx, human_info in enumerate(msg.human_infos):
            try:
                # Attempt to convert the id to a float and then to an integer
                marker_id = int(float(human_info.id)) if human_info.id and human_info.id != '' else self.get_unique_id()
            except ValueError:
                # If conversion fails, assign a unique ID
                marker_id = self.get_unique_id()

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

            marker = self.create_bounding_box_marker(marker_id, human_info, msg.header)
            text_marker = self.create_text_marker(marker_id, human_info, msg.header)
            marker_array.markers.append(marker)
            marker_array.markers.append(text_marker)
        self.publisher.publish(marker_array)
    
    def get_unique_id(self):
        new_id = 0
        while new_id in self.existing_ids:
            new_id += 1
        self.existing_ids.add(new_id)
        return new_id
    
    def create_bounding_box_marker(self, marker_id, human_info, header):
        marker = Marker()
        marker.header = header
        marker.ns = "bounding_box"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Define the bounding box corners using the sanitized min and max coordinates
        p1 = Point(x=float(sanitize_value(human_info.min[0])), y=float(sanitize_value(human_info.min[1])), z=float(sanitize_value(human_info.min[2])))  # Bottom-left front
        p2 = Point(x=float(sanitize_value(human_info.max[0])), y=float(sanitize_value(human_info.min[1])), z=float(sanitize_value(human_info.min[2])))  # Bottom-right front
        p3 = Point(x=float(sanitize_value(human_info.max[0])), y=float(sanitize_value(human_info.max[1])), z=float(sanitize_value(human_info.min[2])))  # Top-right front
        p4 = Point(x=float(sanitize_value(human_info.min[0])), y=float(sanitize_value(human_info.max[1])), z=float(sanitize_value(human_info.min[2])))  # Top-left front
        p5 = Point(x=float(sanitize_value(human_info.min[0])), y=float(sanitize_value(human_info.min[1])), z=float(sanitize_value(human_info.max[2])))  # Bottom-left back
        p6 = Point(x=float(sanitize_value(human_info.max[0])), y=float(sanitize_value(human_info.min[1])), z=float(sanitize_value(human_info.max[2])))  # Bottom-right back
        p7 = Point(x=float(sanitize_value(human_info.max[0])), y=float(sanitize_value(human_info.max[1])), z=float(sanitize_value(human_info.max[2])))  # Top-right back
        p8 = Point(x=float(sanitize_value(human_info.min[0])), y=float(sanitize_value(human_info.max[1])), z=float(sanitize_value(human_info.max[2])))  # Top-left back

        # Create the lines between the corners to form the bounding box
        # Front face
        marker.points.append(p1)
        marker.points.append(p2)
        marker.points.append(p2)
        marker.points.append(p3)
        marker.points.append(p3)
        marker.points.append(p4)
        marker.points.append(p4)
        marker.points.append(p1)

        # Back face
        marker.points.append(p5)
        marker.points.append(p6)
        marker.points.append(p6)
        marker.points.append(p7)
        marker.points.append(p7)
        marker.points.append(p8)
        marker.points.append(p8)
        marker.points.append(p5)

        # Connecting edges
        marker.points.append(p1)
        marker.points.append(p5)
        marker.points.append(p2)
        marker.points.append(p6)
        marker.points.append(p3)
        marker.points.append(p7)
        marker.points.append(p4)
        marker.points.append(p8)

        marker.scale.x = self.marker_scale
        marker.color.a = self.marker_color['a']
        marker.color.r = self.marker_color['r']
        marker.color.g = self.marker_color['g']
        marker.color.b = self.marker_color['b']

        # Set the lifetime of the marker to 2 seconds
        marker.lifetime = Duration(sec=2)

        return marker

    def create_text_marker(self, marker_id, human_info, header):
        marker = Marker()
        marker.header = header
        marker.ns = "bounding_box_text"
        marker.id = marker_id #+ 1000  # Offset ID to avoid conflicts with bounding box markers
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = (sanitize_value(human_info.min[0]) + sanitize_value(human_info.max[0])) / 2.0
        marker.pose.position.y = (sanitize_value(human_info.min[1]) + sanitize_value(human_info.max[1])) / 2.0
        marker.pose.position.z = sanitize_value(human_info.max[2]) + 0.2  # Adjust height to be above the bounding box
        marker.scale.z = 0.5  # Text height
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.text = str('Person') + str(marker_id)

        # Set the lifetime of the text marker to 2 seconds
        marker.lifetime = Duration(sec=2)

        return marker

def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()