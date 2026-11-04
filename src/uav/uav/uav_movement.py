import math
from custom_interfaces.msg import CamData
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import GlobalPositionTarget
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix
from custom_interfaces.srv import Velocity
from uav.pid_controller import PIDController
from uav.topic_names import TopicNames
from uav.math_funcs import MathFuncs
from uav.kalman_filter import KalmanFilter
import numpy as np

class UAVMovement(Node):
    def __init__(self):
        super().__init__("UAVMovement")

        self.pose_publisher = self.create_publisher(
            GlobalPositionTarget, 
            "/mavros/setpoint_raw/global", 
            qos_profile=qos_profile_sensor_data
        )

        self.imu_subscripter = self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self.save_imu_data,
            qos_profile=qos_profile_sensor_data
        )

        self.gps_subscripter = self.create_subscription(
            NavSatFix,
            "/mavros/global_position/global",
            self.save_gps_data,
            qos_profile=qos_profile_sensor_data
        )

        self.img_subscripter = self.create_subscription(
            CamData,
            TopicNames.CAM_DATA_TOPIC,
            self.get_coordinates,
            qos_profile=qos_profile_sensor_data
        )

        self.timer_period = 0.1
        self.move_timer = self.create_timer(self.timer_period, self.move_callback)
        self.imu_data = Imu()
        self.current_gps = NavSatFix()
        self.x_speed = 5.0
        self.y_speed = 5.0
        self.speed_pid = PIDController(kp=0.5, ki=0.1, kd=0.2, max_integral=10.0, max_output=10.0)
        self.pitch_pid = PIDController(kp=0.5, ki=0.1, kd=0.2, max_integral=10.0, max_output=10.0)
        self.yaw_pid = PIDController(kp=0.5, ki=0.1, kd=0.2, max_integral=10.0, max_output=10.0)

        self.k_filter = KalmanFilter(dim_x=4, dim_z=2)

        self.target_pose = GlobalPositionTarget()
        self.target_pose.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        self.target_pose.header.frame_id = "map"
        self.target_pose.type_mask = (
            GlobalPositionTarget.IGNORE_VX |
            GlobalPositionTarget.IGNORE_VY |
            GlobalPositionTarget.IGNORE_VZ |
            GlobalPositionTarget.IGNORE_AFX |
            GlobalPositionTarget.IGNORE_AFY |
            GlobalPositionTarget.IGNORE_AFZ |
            GlobalPositionTarget.IGNORE_YAW |
            GlobalPositionTarget.IGNORE_YAW_RATE
        )

        self.locked = False

        self.circle_speed = 0.5
        self.circle_time = 0.0
        self.loss_time = 0.0
        self.reseted = False

    def save_gps_data(self, msg):
        self.current_gps = msg

    def get_coordinates(self, msg: CamData):
        if not msg.found:
            if not self.locked:
                self.loss_time = self.get_clock().now().nanoseconds / 1e9 - self.loss_time
                self.circle()
            else:
                if self.reseted:
                    return
                if self.get_clock().now().nanoseconds / 1e9 - self.loss_time > 5.0:
                    self.k_filter = KalmanFilter(dim_x=4, dim_z=2)
                    self.reseted = True

        angle_error_yaw = 0.0
        angle_error_pitch = 0.0

        self.k_filter.predict()

        if msg.found:
            self.locked = False
            self.reseted = False
            observation = np.array([[float(msg.center_x)], [float(msg.center_y)]])
            self.k_filter.update(observation)
            self.get_logger().warn(f"Target Found: {msg.center_x}, {msg.center_y}")

        p_x = self.k_filter.x[0, 0]
        p_y = self.k_filter.x[1, 0]

        self.get_logger().info(f"Kalman Filter Prediction: {p_x}, {p_y}")

        pixel_data_yaw = lambda x: MathFuncs.calculate_angle_error(x, msg.img_width)
        angle_error_yaw = pixel_data_yaw(msg.center_x) if msg.found else pixel_data_yaw(p_x)
        angle_error_yaw = math.radians(angle_error_yaw)

        pixel_data_pitch = lambda y: MathFuncs.calculate_angle_error_y(y, msg.img_height)
        angle_error_pitch = pixel_data_pitch(msg.center_y) if msg.found else pixel_data_pitch(p_y)
        angle_error_pitch = math.radians(angle_error_pitch)

        yaw_rad = MathFuncs.get_yaw_from_quaternion(self.imu_data.orientation)
        pitch_rad = MathFuncs.quaternion_to_pitch(self.imu_data.orientation)

        bearing = (yaw_rad + angle_error_yaw) % (2 * math.pi)
        elevation = pitch_rad + angle_error_pitch

        lat1 = math.radians(self.current_gps.latitude)
        lon1 = math.radians(self.current_gps.longitude)
        alt1 = self.current_gps.altitude

        virtual_distance = 100.0
        R = 6378137.0             # WGS-84 Earth radius (meters)

        horizontal_dist = virtual_distance * math.cos(elevation)

        lat2 = math.asin(math.sin(lat1) * math.cos(horizontal_dist / R) +
                         math.cos(lat1) * math.sin(horizontal_dist / R) * math.cos(bearing))
        
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(horizontal_dist / R) * math.cos(lat1),
                                 math.cos(horizontal_dist / R) - math.sin(lat1) * math.sin(lat2))

        vertical_dist = virtual_distance * math.sin(elevation)
        target_altitude = alt1

        target_latitude = math.degrees(lat2)
        target_longitude = math.degrees(lon2)

        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.target_pose.latitude = target_latitude
        self.target_pose.longitude = target_longitude
        self.target_pose.altitude = target_altitude

    def circle(self):
        """ if not self.locked:
            self.circle_time = self.get_clock().now().nanoseconds / 1e9
            self.locked = True

        r = 1.0
        x_center = self.current_gps.latitude - r
        y_center = self.current_gps.longitude

        elapsed_time = self.get_clock().now().nanoseconds / 1e9 - self.circle_time

        x = x_center + r * math.cos(elapsed_time * self.circle_speed)
        y = y_center + r * math.sin(elapsed_time * self.circle_speed)

        self.target_pose.latitude = x
        self.target_pose.longitude = y
        self.target_pose.altitude = self.current_gps.altitude """
        pass

    def move_callback(self):
        if self.current_gps.latitude == 0.0 and self.current_gps.longitude == 0.0:
            return
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"Publishing Target Pose: {self.target_pose.latitude}, {self.target_pose.longitude}, {self.target_pose.altitude}")
        self.pose_publisher.publish(self.target_pose)

    def save_imu_data(self, msg):
        self.imu_data = msg

def main(args=None):
    rclpy.init()
    node = UAVMovement()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
