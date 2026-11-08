from sympy import Quaternion
import numpy as np
import math

class MathFuncs:
    @staticmethod
    def get_yaw_from_quaternion(q: Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
        
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        return yaw

    @staticmethod
    def quaternion_to_pitch(q: Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        
        return pitch

    @staticmethod
    def quaternion_to_rotation_matrix(q: Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]
        ])

        return R

    @staticmethod
    def calculate_angle_error(center_x, img_width):
        pixel_error = center_x - (img_width / 2.0)
        angle = pixel_error / img_width * 1.047

        return math.degrees(angle)

    @staticmethod
    def calculate_angle_error_y(center_y, img_height):
        fov_vertical = 2* np.arctan(np.tan(1.047 / 2) * 9/16)
        pixel_error = center_y - (img_height / 2.0)
        angle = pixel_error / img_height * fov_vertical

        return math.degrees(angle)
    