import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import numpy as np

import cv2
from cv_bridge import CvBridge
import torch

class Subcriber_Image(Node):
    def __init__(self):
        super().__init__('cnn_image_subscriber')
        self.subscription = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.callback, 
            60
        )
        self.subscription
        self.bridge = CvBridge()

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

        cv2.namedWindow("Display")
        cv2.resizeWindow("Display", 640, 480)

    def tearDown(self):
        cv2.destroyAllWindows()

    def callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model(image) # Run inference

        cv2.imshow('Display', np.squeeze(results.render()))
        cv2.waitKey(1)
        

def main(args=None):
    rclpy.init(args=args)

    subscriber_image = Subcriber_Image()

    rclpy.spin(subscriber_image)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    subscriber_image.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
