import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from custom_interfaces.msg import CamData
from mavros_msgs.srv import CommandInt
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64

from uav.topic_names import TopicNames
from uav.math_funcs import MathFuncs
from uav.kalman_filter import KalmanFilter


class UAVMovement(Node):
    LOSS_TIMEOUT = 5.0
    CAM_STALE_TIMEOUT = 0.5
    VIRTUAL_DISTANCE = 150.0
    MIN_ALT = 20.0
    EARTH_R = 6378137.0        # m (WGS-84)

    YAW_ERR_RIGHT_POSITIVE = True
    PITCH_ERR_DOWN_POSITIVE = True

    def __init__(self):
        super().__init__("UAVMovement")

        self.pose_client = self.create_client(CommandInt, "/mavros/cmd/command_int")
        self.pose_future = None

        self.create_subscription(
            Imu, "/mavros/imu/data", self.save_imu_data,
            qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            NavSatFix, "/mavros/global_position/global", self.save_gps_data,
            qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            Float64, "/mavros/global_position/rel_alt", self.save_rel_alt,
            qos_profile=qos_profile_sensor_data)
        self.create_subscription(
            CamData, TopicNames.CAM_DATA_TOPIC, self.get_coordinates,
            qos_profile=qos_profile_sensor_data)

        self.timer_period = 0.1
        self.move_timer = self.create_timer(self.timer_period, self.move_callback)

        self.imu_data = Imu()
        self.imu_received = False
        self.current_gps = NavSatFix()
        self.gps_received = False
        self.rel_alt = 0.0
        self.rel_alt_received = False

        self.k_filter = KalmanFilter(dim_x=4, dim_z=2)
        self.filter_initialized = False

        self.target_valid = False
        self.target_px = (0.0, 0.0)
        self.img_width = 0
        self.img_height = 0
        self.last_seen_time = 0.0
        self.last_cam_time = 0.0

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def prepare_pose_command(self, x: int, y: int, z: float) -> CommandInt.Request:
        request = CommandInt.Request()
        request.broadcast = False
        request.frame = 3
        request.command = 192
        request.current = 0
        request.autocontinue = 0
        request.param1 = -1.0
        request.param2 = 1.0
        request.param3 = 0.0
        request.param4 = 0.0
        request.x = int(x)
        request.y = int(y)
        request.z = float(z)
        return request

    def save_imu_data(self, msg: Imu):
        self.imu_data = msg
        self.imu_received = True

    def save_gps_data(self, msg: NavSatFix):
        self.current_gps = msg
        self.gps_received = (
            not (math.isnan(msg.latitude) or math.isnan(msg.longitude))
            and not (msg.latitude == 0.0 and msg.longitude == 0.0)
        )

    def save_rel_alt(self, msg: Float64):
        self.rel_alt = msg.data
        self.rel_alt_received = True

    def get_coordinates(self, msg: CamData):
        now = self.now_sec()
        self.last_cam_time = now
        self.img_width = msg.img_width
        self.img_height = msg.img_height

        if msg.found:
            obs = np.array([[float(msg.center_x)], [float(msg.center_y)]])

            if not self.filter_initialized:
                self.k_filter = KalmanFilter(dim_x=4, dim_z=2)
                self.k_filter.x[0, 0] = obs[0, 0]
                self.k_filter.x[1, 0] = obs[1, 0]
                self.filter_initialized = True
            else:
                self.k_filter.predict()

            self.k_filter.update(obs)
            self.last_seen_time = now
            self.get_logger().info(
                f"Target Found: {msg.center_x}, {msg.center_y}",
                throttle_duration_sec=0.5)
        else:
            if not self.filter_initialized:
                return

            if now - self.last_seen_time > self.LOSS_TIMEOUT:
                self.filter_initialized = False
                self.target_valid = False
                self.find_target()
                return

            self.k_filter.predict()

        self.target_px = (float(self.k_filter.x[0, 0]), float(self.k_filter.x[1, 0]))
        self.target_valid = True

    def compute_target(self, px: float, py: float):
        err_yaw = math.radians(MathFuncs.calculate_angle_error(px, self.img_width))
        err_pitch = math.radians(MathFuncs.calculate_angle_error_y(py, self.img_height))

        yaw_right = err_yaw if self.YAW_ERR_RIGHT_POSITIVE else -err_yaw
        pitch_up = -err_pitch if self.PITCH_ERR_DOWN_POSITIVE else err_pitch

        q = self.imu_data.orientation
        yaw_enu = MathFuncs.get_yaw_from_quaternion(q)
        pitch_flu = MathFuncs.quaternion_to_pitch(q)

        heading = (math.pi / 2.0 - yaw_enu) % (2.0 * math.pi)
        nose_up = -pitch_flu

        bearing = (heading + yaw_right) % (2.0 * math.pi)
        elevation = nose_up + pitch_up

        lat1 = math.radians(self.current_gps.latitude)
        lon1 = math.radians(self.current_gps.longitude)

        horizontal_dist = self.VIRTUAL_DISTANCE * math.cos(elevation)
        vertical_dist = self.VIRTUAL_DISTANCE * math.sin(elevation)
        delta = horizontal_dist / self.EARTH_R

        lat2 = math.asin(math.sin(lat1) * math.cos(delta) +
                         math.cos(lat1) * math.sin(delta) * math.cos(bearing))
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(delta) * math.cos(lat1),
                                 math.cos(delta) - math.sin(lat1) * math.sin(lat2))
        lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi  # [-pi, pi]

        z = max(self.rel_alt + vertical_dist, self.MIN_ALT)

        self.get_logger().info(
            f"yaw_enu={math.degrees(yaw_enu):.1f} heading={math.degrees(heading):.1f} "
            f"bearing={math.degrees(bearing):.1f} elev={math.degrees(elevation):.1f} "
            f"z={z:.1f}",
            throttle_duration_sec=0.5)

        self.get_logger().info(f"X: {int(math.degrees(lat2) * 1e7)} Y: {int(math.degrees(lon2) * 1e7)} Z: {z}")

        return int(math.degrees(lat2) * 1e7), int(math.degrees(lon2) * 1e7), z

    def move_callback(self):
        if not (self.imu_received and self.gps_received and self.rel_alt_received):
            return
        if not self.target_valid or self.img_width == 0 or self.img_height == 0:
            return
        if self.now_sec() - self.last_cam_time > self.CAM_STALE_TIMEOUT:
            return

        x, y, z = self.compute_target(*self.target_px)
        self.move_to(x, y, z)

    def move_to(self, x: int, y: int, z: float):
        if not self.pose_client.service_is_ready():
            return
        if self.pose_future is not None and not self.pose_future.done():
            return
        self.pose_future = self.pose_client.call_async(
            self.prepare_pose_command(x, y, z))

    def find_target(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = UAVMovement()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
