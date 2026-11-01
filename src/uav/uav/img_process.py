import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from uav.topic_names import TopicNames
from rclpy.qos import qos_profile_sensor_data
from custom_interfaces.msg import CamData
from ament_index_python.packages import get_package_share_directory
import os
from ultralytics import YOLO
import cv_bridge
import cv2
import numpy as np

class UAVImageProcessor(Node):
    def __init__(self):
        super().__init__('UAVImageProcessor')

        package_share_dir = get_package_share_directory('uav')
        
        model_path = os.path.join(package_share_dir, 'models', 'model.pt')

        self.model = YOLO(model_path)

        self.get_logger().info(f"Listening... {TopicNames.CAM_TOPIC}")
        self.img_subscriber = self.create_subscription(Image, TopicNames.CAM_TOPIC, self.process, qos_profile_sensor_data)
        self.data_publisher = self.create_publisher(CamData, TopicNames.CAM_DATA_TOPIC, qos_profile_sensor_data)

        self.area_treshold = 2.0

    def get_data(self, center_x: float, center_y: float, width: float, 
                 height: float, img_height: float, img_width: float, found: bool):
        msg = CamData()
        msg.center_x = center_x
        msg.center_y = center_y
        msg.width = width
        msg.height = height
        msg.img_width = img_width
        msg.img_height = img_height
        msg.found = found
        msg.header.stamp = self.get_clock().now().to_msg()

        return msg

    def empty_data(self):
        msg = self.get_data(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
        msg.header.stamp = Time().to_msg()

        return msg

    def process(self, msg):
        frame = cv_bridge.CvBridge().imgmsg_to_cv2(msg)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        prediction = self.model.predict(frame)

        data_msg = self.empty_data()

        img_height, img_width, _ = frame.shape

        if len(prediction) >= 0:
            for target in prediction:
                if len(target.boxes) <= 0:
                    continue

                boxes = target.boxes[0]
                
                x1, y1, x2, y2 = boxes.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 10), 2)
                cv2.putText(frame, f"{target.names[int(boxes.cls)]} {float(boxes.conf.data):.2f}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (220, 220, 220), 1)

                x1, y1, x2, y2 = map(float, boxes.xyxy[0])
                

                data_msg = self.get_data(center_x = center_x, center_y = center_y, width = float(x1 + x2), 
                                         height = float(y1 + y2), img_height = float(img_height), img_width = float(img_width), found = True)

        self.data_publisher.publish(data_msg)

        cv2.imshow("Cam", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

def main(args=None):
    rclpy.init(args=args)

    node = UAVImageProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()