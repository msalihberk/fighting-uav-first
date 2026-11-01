import math
from custom_interfaces.msg import CamData
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from custom_interfaces.srv import Float64
from uav.topic_names import TopicNames
from uav.math_funcs import MathFuncs
from uav.pid_controller import PIDController
from custom_interfaces.srv import Velocity

class UAVCalculator(Node):
    def __init__(self):
        super().__init__("UAVCalculator")

        self.imu_subscripter = self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self.save_imu_data,
            qos_profile=qos_profile_sensor_data
        )

        self.img_subscripter = self.create_subscription(
            CamData,
            TopicNames.CAM_DATA_TOPIC,
            self.save_img_data,
            qos_profile=qos_profile_sensor_data
        )

        self.yaw_subscripter = self.create_subscription(
            Float64,
            TopicNames.CAM_YAW_PUBLISHER_TOPIC,
            self.save_yaw_data,
            qos_profile=qos_profile_sensor_data
        )

        self.vel_service = self.create_service(Velocity, TopicNames.GET_VEL_VALUES, self.get_velocity)

        self.cam_yaw = 0.0
        self.img_data = CamData()
        self.pid_controller = PIDController(kp=0.4, ki=0.1, kd=0.4, max_integral=10, max_output=30)

        self.max_speed = 30.0

    def save_yaw_data(self, msg):
        self.cam_yaw = msg.data

    def control_target(self, msg: CamData):
        if msg.img_width <= 1.0 and self.is_task_completed:
            self.get_logger().info("Target lost or out of view.")
            self.stop()

    def get_velocity(self, request, response):
        target_area = request.width * request.height
        img_area = request.img_width * request.img_height
        proportion = img_area / target_area
        total_speed = self.pid_controller.calculate(proportion, 1)
        limited_speed = self.limited_speed(total_speed)
        response.vel.linear.x = limited_speed * math.cos(self.cam_yaw)
        response.vel.linear.y = limited_speed * math.sin(self.cam_yaw)
        return response

    def limited_speed(self, speed):
        return max(min(speed, self.max_speed), -self.max_speed)

    def save_img_data(self, msg):
        self.img_data = msg

def main(args=None):
    rclpy.init()
    node = UAVCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
