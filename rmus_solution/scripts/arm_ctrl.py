#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import IntEnum
from math import fabs
from typing import Union
import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose, Twist
import tf2_ros
from geometry_msgs.msg import Pose, TransformStamped, Vector3


class arm_action:
    def __init__(self):
        self.__gripper_pub = rospy.Publisher("arm_gripper", Point, queue_size=10)
        self.__position_pub = rospy.Publisher("arm_position", Pose, queue_size=10)
        self.__cmd_vel_sub = rospy.Subscriber(
            "/cmd_vel", Twist, self.__cmd_vel_callback
        )
        """ state params """
        self.movement_start_time = 0.0
        self.movement_end_time = 0.0
        self.movement_active_time = 0.0
        self.__vel_old = [0.0, 0.0, 0.0]

    def __cmd_vel_callback(self, msg: Twist):
        self.__vel = [msg.linear.x, msg.linear.y, msg.angular.z]
        v0 = np.linalg.norm(self.__vel_old[0:2])
        v1 = np.linalg.norm(self.__vel[0:2])
        # only checks for the movement in the x-y plane
        if v0 < 0.2 and v1 >= 0.2:
            # get the time when the movement is active
            self.movement_active_time = rospy.get_time()
        if v0 < 0.1 and v1 >= 0.1:
            # get the time when the movement ends
            self.movement_end_time = rospy.get_time()
        if v0 >= 0.1 and v1 < 0.1:
            # get the time when the movement starts
            self.movement_start_time = rospy.get_time()
        self.__vel_old = self.__vel

    def can_arm_grasp(self, target_in_arm_base: list):
        if target_in_arm_base is None:
            return False
        elif abs(target_in_arm_base[1]) > 0.025:
            return False
        elif 0.18 <= target_in_arm_base[0] <= 0.22 and target_in_arm_base[2] >= -0.10:
            return True
        elif 0.09 <= target_in_arm_base[0] < 0.18 and target_in_arm_base[2] > 0.08:
            return True
        else:
            return False

    def is_vel_active(self):
        return np.linalg.norm(self.__vel[0:2]) >= 0.2

    def get_last_vel(self):
        return self.__vel

    def open_gripper(self):
        open_gripper_msg = Point()
        open_gripper_msg.x = 0.0
        open_gripper_msg.y = 0.0
        open_gripper_msg.z = 0.0
        rospy.loginfo("open the gripper")
        self.__gripper_pub.publish(open_gripper_msg)

    def close_gripper(self):
        close_gripper_msg = Point()
        close_gripper_msg.x = 1.0
        close_gripper_msg.y = 0.0
        close_gripper_msg.z = 0.0
        rospy.loginfo("close the gripper")
        self.__gripper_pub.publish(close_gripper_msg)

    def reset_pos(self):
        self.set_arm(0.1, 0.12)
        rospy.loginfo("reset the arm")

    def set_arm(self, x: float, y: float):
        """
        a low level api to set arm position to (x, y) directly, unit: meter;
        axes of x, y point forwards and upwards respectively,
            i.e. horizontal extension and vertical height
        e.g. typical value of (extension, height)
            reset:  (0.1, 0.12);
            drop: (0.21, -0.08);     place: (0.21, h_layer)
        """
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        self.__position_pub.publish(pose)

    def place_pos(self, place_layer: int = 1):
        rospy.loginfo("<manipulator>: now prepare to place (first layer)")
        extension = 0.21
        height = -0.045 + 0.055 * (place_layer - 1)
        self.set_arm(extension, height)

    def grasp(self, align_act, target_in_arm_base: list):
        extension = np.clip(target_in_arm_base[0], 0.09, 0.22)
        height = max(target_in_arm_base[2], -0.08)
        self.set_arm(extension, height)
        print(f"Grasp: move to ({extension:.3f}, {height:.3f})m")
        rospy.sleep(1.5)
        self.close_gripper()
        rospy.sleep(1.5)
        self.reset_pos()
        ## move backwards a little bit
        align_act.send_cmd_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.5)
        align_act.send_cmd_vel([0.0, 0.0, 0.0])

    def place(self, align_act, place_layer):
        rospy.loginfo(f"elevate gripper to layer {place_layer}")
        self.place_pos(place_layer)
        rospy.sleep(2)
        # print("Place: reach the goal for placing.")
        self.open_gripper()
        rospy.sleep(0.7)
        ## move backwards a little bit
        align_act.send_cmd_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.7)
        self.reset_pos()
        align_act.send_cmd_vel([0.0, 0.0, 0.0])
        # rospy.sleep(0.5)

    def preparation_for_grasp(self, align_act):
        """takes ~3 seconds to brake and open gripper"""
        rospy.loginfo("First align then grasp")
        rospy.loginfo("align to the right place")
        align_act.send_cmd_vel([0.0, 0.0, 0.0])
        # rospy.sleep(0.5)
        self.reset_pos()
        self.open_gripper()
        rospy.sleep(0.5)

    # TODO: check if aligning with arm reaching out will cause bug
    def preparation_for_place(self, align_act, place_layer: int):
        """takes ~3 seconds to brake and elevate gripper"""


class align_action:
    def __init__(self, arm_act: arm_action):
        """object reference"""
        self.__cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        """ state params """
        self.__is_near_target_state_old = False
        self.x_sp = None  ## state, set point
        self.x_mv = None  ## state, measured variable
        """ config params """
        self.ss_cfg = {"Kp": 4, "Ki": 2, "dist_thresh": 0.1}
        self.max_vel = 0.3
        self.max_angular_vel = 0.3
        self.min_vel = 0.0
        self.min_angular_vel = 0.0
        ## need to be initialized before init_ctrl
        self.dT = None

    def send_cmd_vel(self, vel: list):
        vel[0] = np.clip(vel[0], -self.max_vel, self.max_vel)
        vel[1] = np.clip(vel[1], -self.max_vel, self.max_vel)
        vel[2] = np.clip(vel[2], -self.max_angular_vel, self.max_angular_vel)
        twist = Twist()
        twist.linear.z = 0.0
        twist.linear.x = vel[0]
        twist.linear.y = vel[1]
        twist.angular.z = vel[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        self.__cmd_vel_pub.publish(twist)
        # print(f"send cmd_vel: {vel}")

    def apply_velocity_limit(
        self,
        max_vel: float,
        max_angular_vel: float,
        min_vel=0.0,
        min_angular_vel=0.0,
    ):
        self.max_vel = max_vel
        self.max_angular_vel = max_angular_vel
        self.min_vel = min_vel
        self.min_angular_vel = min_angular_vel

    def init_ctrl(self):
        ## init state space controller (nothing to do here)
        self.error_sum = np.zeros(3)
        return

    def set_state_sp(self, x_sp):
        """set target state (setpoint) of arm_ctrl"""
        self.x_sp = np.array(x_sp)
        return

    def set_state_mv(self, x_mv):
        """set measure variable of arm_ctrl"""
        self.x_mv = np.array(x_mv)

    def set_sample_time(self, dT:float):
        self.dT = dT

    def calc_cmd_vel(self, measured_pos: list):
        x, y = measured_pos[0:2]
        error = np.array(measured_pos) - np.array(self.x_sp)
        Kp = self.ss_cfg["Kp"]
        Ki = self.ss_cfg["Ki"] if np.linalg.norm(error[0:2]) < self.ss_cfg["dist_thresh"] else 0
        self.error_sum += error * self.dT if np.linalg.norm(error[0:2]) < self.ss_cfg["dist_thresh"] else 0
        # feedback linearization
        vel_ang = Kp * error[2] + Ki * self.error_sum[2]
        vel_x = Kp * error[0] + Ki * self.error_sum[0] + y * vel_ang
        vel_y = Kp * error[1] + Ki * self.error_sum[1] - x * vel_ang
        ## apply saturation and deadzone
        vel_x = np.clip(fabs(vel_x), self.min_vel, self.max_vel) * np.sign(vel_x) if fabs(vel_x) > 0.01 else 0
        vel_y = np.clip(fabs(vel_y), self.min_vel, self.max_vel) * np.sign(vel_y) if fabs(vel_y) > 0.01 else 0
        vel_ang = np.clip(fabs(vel_ang), self.min_angular_vel, self.max_angular_vel) * np.sign(vel_ang) if fabs(vel_ang) > 0.002 else 0
        return [vel_x, vel_y, vel_ang]

    def align(self):
        """
        send cmd_vel by comparing x_mv and x_sp to align. i.e. the controller
        should be called at certain rate (e.g. 30Hz)
        """
        err = np.array(self.x_mv) - np.array(self.x_sp)
        vel = self.calc_cmd_vel(self.x_mv)
        self.send_cmd_vel(vel)
        print(f"(ctrl err: {100*err[0]:.2f}cm, {100*err[1]:.2f}cm, {np.rad2deg(err[2]):.1f}degree")
        return

    def stop(self):
        self.send_cmd_vel([0.0, 0.0, 0.0])

    def finished(self, tolerance: list, timeout: float):
        """ judge if within tolerance in the last few seconds """
        satisfy = self.tolerated(tolerance)
        if satisfy and not self.__is_near_target_state_old:
            # check if the robot is near the target state for the first time
            self.__near_target_state_start_time = rospy.get_time()
            ...

        # if satisfy:
        #     print(f"aligned {satisfy} for {rospy.get_time() - self.__near_target_state_start_time} sec")
        self.__is_near_target_state_old = satisfy

        return satisfy and rospy.get_time() - self.__near_target_state_start_time > timeout

    def tolerated(self, tolerance: list):
        """ judge if the current state is within the tolerance """
        x1, y1, ang1 = self.x_mv
        x2, y2, ang2 = self.x_sp
        x_th, y_th, ang_th = tolerance
        satisfy_xy = abs(x1 - x2) < abs(x_th) and abs(y1 - y2) < abs(y_th)
        satisfy_ang = abs(ang1 - ang2) < abs(ang_th)
        return satisfy_xy and satisfy_ang


if __name__ == "__main__":
    rospy.init_node("arm_ctrl", anonymous=True)
    my_arm_act = arm_action()
    my_align_act = align_action(my_arm_act)
    rospy.spin()
