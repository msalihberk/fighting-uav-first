import math
from custom_interfaces.msg import CamData
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
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
            PoseStamped, 
            "/mavros/setpoint_position/local", 
            10
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

        self.vel_calculation = self.create_client(Velocity, TopicNames.GET_VEL_VALUES)

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

        self.target_pose = PoseStamped()

    def save_gps_data(self, msg):
        self.current_gps = msg

    def get_coordinates(self, msg: CamData):
        angle_error_yaw = 0.0
        angle_error_pitch = 0.0

        self.k_filter.predict()

        if msg.found:
            observation = np.array([[float(msg.center_x)], [float(msg.center_y)]])
            self.k_filter.update(observation)

            request = Velocity.Request()
            request.width = msg.width
            request.height = msg.height
            request.img_width = msg.img_width
            request.img_height = msg.img_height

            future = self.vel_calculation.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                response = future.result()
                self.save_vel_data(response.vel.linear.x, response.vel.linear.y)
            else:
                self.get_logger().error("Failed to call service for velocity calculation.")

        p_x = self.k_filter.x[0, 0]
        p_y = self.k_filter.x[1, 0]

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
        target_altitude = alt1 + vertical_dist

        target_latitude = math.degrees(lat2)
        target_longitude = math.degrees(lon2)

        self.target_pose = PoseStamped()
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.target_pose.pose.position.x = target_latitude
        self.target_pose.pose.position.y = target_longitude
        self.target_pose.pose.position.z = target_altitude

    def save_vel_data(self, x, y):
        self.x_speed = x
        self.y_speed = y

    def move_callback(self):  
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
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
