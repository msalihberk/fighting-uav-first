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

class UAVImageProcessor(Node):
    def __init__(self):
        super().__init__('UAVImageProcessor')

        package_share_dir = get_package_share_directory('uav')
        
        model_path = os.path.join(package_share_dir, 'models', 'model.pt')
        model_path = "./model.pt"

        self.model = YOLO(model_path)

        self.get_logger().info(f"Listening... {TopicNames.CAM_TOPIC}")
        self.img_subscriber = self.create_subscription(Image, TopicNames.CAM_TOPIC, self.process, qos_profile_sensor_data)
        self.data_publisher = self.create_publisher(CamData, TopicNames.CAM_DATA_TOPIC, qos_profile_sensor_data)

        self.area_treshold = 2.0
        self.target_class = 0
        self.yolo_size = [640, 640]

        self.min_confident = 0.5

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

    def letterbox(self, img, new_shape=[640, 640], color=(114, 114, 114)):
        shape = img.shape[:2]
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        
        dw /= 2  
        dh /= 2

        if shape[::-1] != new_unpad:  
            interpolation = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, new_unpad, interpolation=interpolation)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    def scale_boxes(self, box, original_size, letterbox_size, proportion, fills):
        left_fill, top_fill = fills
        
        x_min = box[0] - left_fill
        y_min = box[1] - top_fill
        x_max = box[2] - left_fill
        y_max = box[3] - top_fill
        
        x_min = int(round(x_min / proportion))
        y_min = int(round(y_min / proportion))
        x_max = int(round(x_max / proportion))
        y_max = int(round(y_max / proportion))
        
        orig_h, orig_w = original_size[:2]
        x_min = max(0, min(x_min, orig_w))
        y_min = max(0, min(y_min, orig_h))
        x_max = max(0, min(x_max, orig_w))
        y_max = max(0, min(y_max, orig_h))
        
        return [x_min, y_min, x_max, y_max]


    def process(self, msg):
        frame = cv_bridge.CvBridge().imgmsg_to_cv2(msg)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        yolo_frame, proportion, fills = self.letterbox(frame, self.yolo_size)
        results = self.model.predict(frame, verbose=False)

        data_msg = self.empty_data()

        img_height, img_width, _ = frame.shape

        data_msg.img_height = float(img_height)
        data_msg.img_width = float(img_width)

        if len(results) > 0:
            result = results[0]
            
            target_boxes = [b for b in result.boxes if int(b.cls) == self.target_class]
            if target_boxes:
                t_box = max(target_boxes, key=lambda b: float(b.xywh[0][2] * b.xywh[0][3]))
                
                x1, y1, x2, y2 = map(int, t_box.xyxy[0])
                box = [x1, y1, x2, y2]
                box = self.scale_boxes(box, frame.shape, self.yolo_size, proportion, fills)
                x1, y1, x2, y2 = box
                
                center_x = ((x1 + x2) / 2)
                center_y = ((y1 + y2) / 2)

                cv2.line(frame, (img_width // 2, img_height // 2), (int(center_x), int(center_y)), (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 255, 10), 2)
                cv2.putText(frame, f"{result.names[int(t_box.cls)]} {float(t_box.conf.data):.2f}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (220, 220, 220), 1)


                x1, y1, x2, y2 = map(float, t_box.xyxy[0])

                data_msg = self.get_data(center_x = center_x, center_y = center_y, width = float(x1 + x2), 
                                        height = float(y1 + y2), img_height = float(img_height), img_width = float(img_width), found = True)

        cv2.rectangle(frame, (img_width // 4, img_height // 10), (img_width * 3 // 4, img_height * 9 // 10), (0, 255, 255), 2)
        cv2.circle(frame, (img_width // 2, img_height // 2), 2, (0, 0, 255), -1)

        self.data_publisher.publish(data_msg)

        cv2.imshow("YOLO Cam", frame)

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
