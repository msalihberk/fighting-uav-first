import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64

from custom_interfaces.msg import CamData
from uav.pid_controller import PIDController
from uav.topic_names import TopicNames


class UAVCamController(Node):
    def __init__(self):
        super().__init__("UAVCamController")

        self.gimbal_publisher = self.create_publisher(
            Float64,
            "/gimbal/cmd_yaw",
            10,
        )

        self.img_subscription = self.create_subscription(
            CamData,
            TopicNames.CAM_DATA_TOPIC,
            self.save_img_data,
            qos_profile=qos_profile_sensor_data,
        )

        self.pid_controller = PIDController(kp=0.5, ki=0.02, kd=0.4, max_integral=10.0, max_output=15.0)
        self.current_yaw = 0.0
        self.max_yaw = 30.0
        self.found = True

    def save_img_data(self, msg: CamData):
        if msg.img_width <= 1.0:
            correction = self.pid_controller.predict()
            self.found = False
            return

        elif not self.found:
            self.pid_controller.reset()
            self.found = True

        angle_error = self.calculate_angle(msg.center_x, msg.img_width)
        target_yaw = self.current_yaw + angle_error
        correction = self.pid_controller.calculate(target_yaw, self.current_yaw)
        self.current_yaw = self.clamp_yaw(self.current_yaw + correction)
        self.set_angle(self.current_yaw)

    def clamp_yaw(self, yaw):
        return max(-self.max_yaw, min(self.max_yaw, yaw))

    def calculate_angle(self, center_x, img_width):
        pixel_error = center_x - (img_width / 2.0)
        angle = pixel_error / img_width * 1.047
        angle *= -1

        return math.degrees(angle)

    def set_angle(self, new_yaw):
        msg = Float64()
        msg.data = math.radians(float(self.clamp_yaw(new_yaw)))
        self.gimbal_publisher.publish(msg)


def main(args=None):
    rclpy.init()
    node = UAVCamController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
