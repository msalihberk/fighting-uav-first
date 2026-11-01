import math
from custom_interfaces.msg import CamData
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from custom_interfaces.srv import Velocity
from uav.topic_names import TopicNames
from uav.math_funcs import MathFuncs

class UAVMovement(Node):
    def __init__(self):
        super().__init__("UAVMovement")

        self.vel_publisher = self.create_publisher(
            TwistStamped, 
            "/mavros/setpoint_velocity/cmd_vel", 
            10
        )

        self.imu_subscripter = self.create_subscription(
            Imu,
            "/mavros/imu/data",
            self.save_imu_data,
            qos_profile=qos_profile_sensor_data
        )

        self.img_subscripter = self.create_subscription(
            CamData,
            TopicNames.CAM_DATA_TOPIC,
            self.get_velocity,
            qos_profile=qos_profile_sensor_data
        )

        self.vel_calculation = self.create_client(Velocity, TopicNames.GET_VEL_VALUES)

        self.timer_period = 0.1
        self.move_timer = self.create_timer(self.timer_period, self.move_callback)

        self.x_speed = 5.0
        self.y_speed = 5.0

    def get_velocity(self, msg: CamData):
        if msg.found:
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


    def save_vel_data(self, x, y):
        self.x_speed = x
        self.y_speed = y

    def move_callback(self):
        if self.x_speed == 0.0 and self.y_speed == 0.0:
            self.get_logger().info("No movement required.")
            return
        
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = self.x_speed
        msg.twist.linear.y = self.y_speed
        self.vel_publisher.publish(msg)

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
