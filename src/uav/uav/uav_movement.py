import math
from custom_interfaces.msg import CamData
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from custom_interfaces.srv import MovePoint, Coordinate, Float64
from uav.topic_names import TopicNames
from uav.math_funcs import MathFuncs

class UAVMovement(Node):
    def __init__(self):
        super().__init__("UAVMovement")

        self.pos_publisher = self.create_publisher(
            PoseStamped, 
            "/mavros/setpoint_position/local", 
            10
        )

        self.pos_subscripter = self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.save_local_pos,
            qos_profile=qos_profile_sensor_data
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
            self.control_target,
            qos_profile=qos_profile_sensor_data
        )

        self.angle_service = self.create_service(Float64, TopicNames.SET_YAW_TOPIC, self.set_angle)
        self.move_server = self.create_service(MovePoint, TopicNames.MOVEMENT_TOPIC, self.move_to_target)
        
        self.timer_period = 0.1
        self.move_timer = self.create_timer(self.timer_period, self.move_callback)
        self.distance = 1.0

        self.end_point = PoseStamped()
        self.local_pos = PoseStamped()
        self.imu_data = Imu()

        self.is_task_completed = True

    def control_target(self, msg: CamData):
        if msg.img_width <= 1.0 and self.is_task_completed:
            self.get_logger().info("Target lost or out of view.")
            self.stop()

    def set_angle(self, request, response):
        orientation_q = MathFuncs.get_yaw_from_quaternion(self.imu_data.orientation)
        target_yaw = request.data + orientation_q
        
        self.end_point.pose.orientation.z = math.sin(target_yaw / 2)
        self.end_point.pose.orientation.w = math.cos(target_yaw / 2)

        response.result = target_yaw
        return response

    def complete_task(self):
        self.is_task_completed = True

    def start_task(self):
        self.get_logger().info("Starting movement task...")
        self.is_task_completed = False

    def is_empty_point(self, point: Point):
        x_val = point.x <= 0.1 and point.x >= -0.1
        y_val = point.y <= 0.1 and point.y >= -0.1
        z_val = point.z <= 0.1 and point.z >= -0.1
        return x_val and y_val and z_val

    def is_local_pos_initialized(self):
        return self.local_pos.header.stamp.sec > 0 or self.local_pos.header.stamp.nanosec > 0

    def is_ready_to_move(self):
        return self.is_local_pos_initialized() and not self.is_empty_point(self.end_point.pose.position) and not self.is_task_completed

    def move_callback(self):
        if not self.is_ready_to_move():
            return
        
        if self.reached_target():
            self.get_logger().info(f"Reached target: {self.end_point.pose.position.x}, {self.end_point.pose.position.y}, {self.end_point.pose.position.z}")
            self.stop()
            return

        self.end_point.header.stamp = self.get_clock().now().to_msg()
        self.pos_publisher.publish(self.end_point)

    def save_local_pos(self, msg):
        self.local_pos = msg

    def save_imu_data(self, msg):
        self.imu_data = msg

    def calculate_difference(self, vector1, vector2):
        x = vector1.x - vector2.x
        y = vector1.y - vector2.y
        z = vector1.z - vector2.z
        return [x, y, z]

    def calculate_distance(self, vector1, vector2):
        values = self.calculate_difference(vector1, vector2)
        xq = values[0] ** 2
        yq = values[1] ** 2
        zq = values[2] ** 2
        
        return (xq + yq + zq) ** 0.5

    def reached_target(self):
        return self.reached_position(self.end_point.pose.position)

    def reached_position(self, position: Point):
        return self.calculate_distance(self.local_pos.pose.position, position) <= self.distance

    def stop(self):
        if self.is_task_completed: 
            return
        
        self.get_logger().info(f"Stopping movement. Current position: {self.local_pos.pose.position.x}, {self.local_pos.pose.position.y}, {self.local_pos.pose.position.z}")
        self.end_point.pose.position.x = self.local_pos.pose.position.x
        self.end_point.pose.position.y = self.local_pos.pose.position.y
        self.end_point.pose.position.z = self.local_pos.pose.position.z

        self.complete_task()
    
    def move_to_target(self, request, response):
        if not self.is_task_completed or self.reached_position(request.point) or self.is_empty_point(request.point):
            response.success = False
            return response
        
        self.move(request.point)
        self.start_task()
        response.success = True
        return response

    def move(self, point: Point):
        self.end_point.pose.position.x = point.x
        self.end_point.pose.position.y = point.y
        self.end_point.pose.position.z = point.z

        self.get_logger().info(f"Moving to target: {point.x}, {point.y}, {point.z}")

def main(args=None):
    rclpy.init()
    node = UAVMovement()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
