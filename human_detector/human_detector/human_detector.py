#!/usr/bin/env python3
import cv2
import math
from cv_bridge.core import CvBridge
from geometry_msgs.msg import TransformStamped
from human_detector.human_detector_parameters import human_detector_parameters
from image_geometry import PinholeCameraModel
import mediapipe as mp
import rclpy
from rclpy.time import Time
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle.node import LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import CameraInfo, Image
from rclpy.qos_overriding_options import QoSOverridingOptions
from rclpy.qos import qos_profile_sensor_data
from tf2_ros.transform_broadcaster import TransformBroadcaster
from message_filters import ApproximateTimeSynchronizer, Subscriber
import traceback
from rclpy.lifecycle import TransitionCallbackReturn


def mm_to_m(mm):
    return mm / 1000


def yaw_to_quat(yaw):
    half_yaw = yaw / 2.0
    return {
        "z": math.sin(half_yaw),
        "w": math.cos(half_yaw),
    }


class HumanDetector(LifecycleNode):
    def __init__(self):
        super().__init__("human_detector")
        self.param_listener = human_detector_parameters.ParamListener(self)
        self.depth_image: Image = None
        self.image = None
        self.detected_landmarks = None
        self.detected_human_position_world = {"x": 0.0, "y": 0.0}
        self.detected_human_yaw = 0.0
        self.cv_bridge = CvBridge()
        self.model = PinholeCameraModel()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.person_pose_estimator = None
        self.camera_info = None
        self.depth_encoding = None

    def on_configure(self, previous_state: LifecycleState):
        try:
            self.parameters = self.param_listener.get_params()
            self.log_parameters()
            self.time_approximation_slope = self.parameters.time_approximation_slope

            self.get_logger().info("Creating MediaPipe Pose() ...")
            self.person_pose_estimator = mp.solutions.pose.Pose(
                min_detection_confidence=self.parameters.min_detection_confidence,
                min_tracking_confidence=self.parameters.min_tracking_confidence,
            )
            self.get_logger().info("MediaPipe Pose() created OK")

            self.initialize_sync_subscribers()
            self.get_logger().info("Subscribers initialized OK")

            if self.parameters.publish_image_with_detected:
                self.image_with_detected_human_pub = self.create_publisher(
                    Image, "image_with_detected_human", 10
                )

            self.timer = self.create_timer(
                1 / self.parameters.detected_human_transform_frequency,
                self.timer_callback
            )
            self.timer.cancel()

            return TransitionCallbackReturn.SUCCESS

        except Exception:
            self.get_logger().error("on_configure failed:\n" + traceback.format_exc())
            return TransitionCallbackReturn.ERROR

    def initialize_sync_subscribers(self):
        sync_topics = [
            Subscriber(
                self,
                Image,
                "sensors/camera_0/color/image",
                qos_profile=qos_profile_sensor_data,
                #qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
            Subscriber(
                self,
                Image,
                "sensors/camera_0/depth/image",
                qos_profile=qos_profile_sensor_data,
                #qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
            Subscriber(
                self,
                CameraInfo,
                "sensors/camera_0/depth/camera_info",
                qos_profile=qos_profile_sensor_data,
                #qos_overriding_options=QoSOverridingOptions.with_default_policies(),
            ),
        ]

        self.image_approx_time_sync = ApproximateTimeSynchronizer(
            sync_topics,
            queue_size=5,
            slop=self.time_approximation_slope,
        )
        self.image_approx_time_sync.registerCallback(self.on_image_data)

    def on_activate(self, previous_state: LifecycleState):
        self.get_logger().info("IN on_activate")
        self.timer.reset()
        return super().on_activate(previous_state)

    def on_deactivate(self, previous_state: LifecycleState):
        self.get_logger().info("IN on_deactivate")
        self.timer.cancel()
        return super().on_deactivate(previous_state)

    def on_cleanup(self, previous_state: LifecycleState):
        self.get_logger().info("IN on_cleanup")
        self.destroy_resources()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, previous_state: LifecycleState):
        self.get_logger().info("IN on_shutdown")
        self.destroy_resources()
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, previous_state: LifecycleState):
        self.get_logger().info("IN on_error")
        self.destroy_resources()
        return TransitionCallbackReturn.SUCCESS

    def destroy_resources(self):
        self.destroy_timer(self.timer)

    def log_parameters(self):
        self.get_logger().info(f"Human detector uses: {self.parameters.camera_frame_id} as camera link.")
        self.get_logger().info(f"Human detector uses: {self.parameters.detected_human_frame_id} as frame with human.")
        self.get_logger().info(
            "Human detector publishes transform to detected human with "
            f"{self.parameters.detected_human_transform_frequency} Hz."
        )
        self.get_logger().info(
            f"Mediapipe will use {self.parameters.min_detection_confidence} " "as min_detection_confidence"
        )

        self.get_logger().info(
            f"Mediapipe will use {self.parameters.min_tracking_confidence} " "as min_tracking_confidence"
        )
        if self.parameters.publish_image_with_detected:
            self.get_logger().info("Human detector will publish image with detected human.")

    def on_image_data(self, image: Image, depth_image: Image, info: CameraInfo):
        self.camera_info = info
        self.model.fromCameraInfo(self.camera_info)
        self.image_time_stamp = Time.from_msg(image.header.stamp)
        self.image = cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(image), cv2.COLOR_BGR2RGB)
        self.depth_encoding = depth_image.encoding
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        self.store_human_pose()

    def are_rgb_image_same_size_as_depth_image(self):
        rgb_image_height, rgb_image_width, _ = self.image.shape
        depth_image_height, depth_image_width = self.depth_image.shape

        return rgb_image_height == depth_image_height and rgb_image_width == depth_image_width

    def should_detect_human(self):
        if self.camera_info is None or self.image is None or self.depth_image is None:
            self.get_logger().error(
                "No camera info or image or depth image are not stored. Human will not be detected."
            )
            return False

        if not self.are_rgb_image_same_size_as_depth_image():
            self.get_logger().error(
                "Dimensions of rgb image and depth image are not equal. Human will not be detected."
            )
            return False

        return True

    def store_human_pose(self):
        if not self.should_detect_human():
            return

        self.detected_landmarks = self.person_pose_estimator.process(self.image).pose_landmarks
        if self.detected_landmarks is None:
            return

        self.get_3d_human_position()
        human_yaw = self.estimate_human_yaw()
        if human_yaw is not None:
            self.detected_human_yaw = human_yaw

    def get_3d_human_position(self):
        x_pos_of_detected_person, y_pos_of_detected_person = self.get_position_of_human_in_the_image(
            self.detected_landmarks
        )
        if x_pos_of_detected_person <= 0 or y_pos_of_detected_person <= 0:
            return

        planar_point = self.project_pixel_to_planar(x_pos_of_detected_person, y_pos_of_detected_person)
        if planar_point is None:
            return

        self.detected_human_position_world = {"x": planar_point[0], "y": planar_point[1]}

    def get_depth_in_meters(self, x_pos_of_detected_person, y_pos_of_detected_person):
        raw_depth = float(self.depth_image[y_pos_of_detected_person, x_pos_of_detected_person])

        if not math.isfinite(raw_depth) or raw_depth <= 0.0:
            return None

        if self.depth_encoding == "16UC1":
            return mm_to_m(raw_depth)

        if self.depth_encoding in ("32FC1", "64FC1"):
            return raw_depth

        self.get_logger().warn(
            f"Unsupported depth encoding '{self.depth_encoding}'. Assuming meters for depth values."
        )
        return raw_depth

    def project_pixel_to_planar(self, x_pos_of_detected_person, y_pos_of_detected_person):
        depth_of_given_pixel = self.get_depth_in_meters(x_pos_of_detected_person, y_pos_of_detected_person)
        if depth_of_given_pixel is None:
            return None

        ray = self.model.projectPixelTo3dRay((x_pos_of_detected_person, y_pos_of_detected_person))
        ray_3d = [ray_element / ray[2] for ray_element in ray]
        point_xyz = [ray_element * depth_of_given_pixel for ray_element in ray_3d]
        return (point_xyz[2], -point_xyz[0])

    def get_position_of_human_in_the_image(self, landmarks):
        x, y = 0, 0
        if self.detected_landmarks:
            landmarks = mp.solutions.pose.PoseLandmark
            left_hip_landmark = self.detected_landmarks.landmark[landmarks.LEFT_HIP]
            right_hip_landmark = self.detected_landmarks.landmark[landmarks.RIGHT_HIP]
            x, y = self.extract_hip_midpoint(left_hip_landmark, right_hip_landmark)

        return x, y

    def extract_hip_midpoint(self, left_hip_landmark, right_hip_landmark):
        height, width, _ = self.image.shape
        x = int(min((left_hip_landmark.x * width + right_hip_landmark.x * width) / 2, width - 1))
        y = int(min((left_hip_landmark.y * height + right_hip_landmark.y * height) / 2, height - 1))
        return x, y

    def landmark_to_pixel(self, landmark):
        height, width, _ = self.image.shape
        x = int(min(max(landmark.x * width, 0), width - 1))
        y = int(min(max(landmark.y * height, 0), height - 1))
        return (x, y)

    def get_landmark_planar_point(self, landmark_name):
        landmark = self.detected_landmarks.landmark[landmark_name]
        pixel = self.landmark_to_pixel(landmark)
        return self.project_pixel_to_planar(pixel[0], pixel[1])

    def average_planar_points(self, planar_points):
        valid_points = [point for point in planar_points if point is not None]
        if not valid_points:
            return None

        avg_x = sum(point[0] for point in valid_points) / len(valid_points)
        avg_y = sum(point[1] for point in valid_points) / len(valid_points)
        return (avg_x, avg_y)

    def estimate_human_yaw(self):
        landmarks = mp.solutions.pose.PoseLandmark
        left_shoulder = self.get_landmark_planar_point(landmarks.LEFT_SHOULDER)
        right_shoulder = self.get_landmark_planar_point(landmarks.RIGHT_SHOULDER)
        left_hip = self.get_landmark_planar_point(landmarks.LEFT_HIP)
        right_hip = self.get_landmark_planar_point(landmarks.RIGHT_HIP)
        nose = self.get_landmark_planar_point(landmarks.NOSE)

        body_right_vectors = []
        if left_shoulder is not None and right_shoulder is not None:
            body_right_vectors.append(
                (right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
            )
        if left_hip is not None and right_hip is not None:
            body_right_vectors.append((right_hip[0] - left_hip[0], right_hip[1] - left_hip[1]))

        if not body_right_vectors:
            return None

        avg_right_x = sum(vector[0] for vector in body_right_vectors) / len(body_right_vectors)
        avg_right_y = sum(vector[1] for vector in body_right_vectors) / len(body_right_vectors)
        right_norm = math.hypot(avg_right_x, avg_right_y)
        if right_norm < 1e-6:
            return None

        body_right_x = avg_right_x / right_norm
        body_right_y = avg_right_y / right_norm

        forward_x = -body_right_y
        forward_y = body_right_x

        upper_body_center = self.average_planar_points([left_shoulder, right_shoulder])
        lower_body_center = self.average_planar_points([left_hip, right_hip])
        torso_center = self.average_planar_points([upper_body_center, lower_body_center])

        if torso_center is not None and nose is not None:
            torso_to_nose_x = nose[0] - torso_center[0]
            torso_to_nose_y = nose[1] - torso_center[1]
            if forward_x * torso_to_nose_x + forward_y * torso_to_nose_y < 0.0:
                forward_x *= -1.0
                forward_y *= -1.0

        return math.atan2(forward_y, forward_x)

    def timer_callback(self):
        if self.detected_human_position_world["x"] > 0.0:
            self.broadcast_timer_callback()
        self.publish_image_with_detected_human()

    def broadcast_timer_callback(self):
        transform = TransformStamped()
        transform.header.stamp = self.image_time_stamp.to_msg()
        transform.header.frame_id = self.parameters.camera_frame_id
        transform.child_frame_id = self.parameters.detected_human_frame_id
        transform.transform.translation.x = self.detected_human_position_world["x"]
        transform.transform.translation.y = self.detected_human_position_world["y"]
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        yaw_quaternion = yaw_to_quat(self.detected_human_yaw)
        transform.transform.rotation.z = yaw_quaternion["z"]
        transform.transform.rotation.w = yaw_quaternion["w"]
        self.tf_broadcaster.sendTransform(transform)

    def draw_person_pose(self, image):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image,
            self.detected_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
        return image

    def publish_image_with_detected_human(self):
        if not self.parameters.publish_image_with_detected:
            return

        if self.detected_landmarks is not None:
            modified_image_msg = self.cv_bridge.cv2_to_imgmsg(self.draw_person_pose(self.image.copy()))
            self.image_with_detected_human_pub.publish(modified_image_msg)
        elif self.image is not None:
            image_msg = self.cv_bridge.cv2_to_imgmsg(self.image)
            self.image_with_detected_human_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    pose_detector = HumanDetector()
    rclpy.spin(pose_detector)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
