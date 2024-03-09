#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose, Twist
from simple_pid import PID


class arm_action:

    def __init__(self):
        self.__gripper_pub = rospy.Publisher("arm_gripper", Point, queue_size=10)
        self.__position_pub = rospy.Publisher("arm_position", Pose, queue_size=10)
        self.__cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.movement_start_time = 0.0
        self.movement_end_time = 0.0
        self.movement_saturat_time = 0.0

    def send_cmd_vel(self, vel: list):
        self.__vel = vel

        if np.linalg.norm(vel) > 0.5:
            self.movement_saturat_time = rospy.get_time()

        if np.linalg.norm(vel) < 0.1:
            self.movement_end_time = rospy.get_time()
        else:
            self.movement_start_time = rospy.get_time()

        twist = Twist()
        twist.linear.z = 0.0
        twist.linear.x = vel[0]
        twist.linear.y = vel[1]
        twist.angular.z = vel[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        self.__cmd_vel_pub.publish(twist)

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
        reset_arm_msg = Pose()
        reset_arm_msg.position.x = 0.1
        reset_arm_msg.position.y = 0.12
        reset_arm_msg.position.z = 0.0
        reset_arm_msg.orientation.x = 0.0
        reset_arm_msg.orientation.y = 0.0
        reset_arm_msg.orientation.z = 0.0
        reset_arm_msg.orientation.w = 0.0
        rospy.loginfo("reset the arm")
        self.__position_pub.publish(reset_arm_msg)

    def place_pos(self, place_layer: int = 1):
        rospy.loginfo("<manipulater>: now prepare to place (first layer)")
        pose = Pose()
        pose.position.x = 0.21
        pose.position.y = -0.04 + 0.055 * (place_layer - 1)
        self.__position_pub.publish(pose)

    def grasp_pos(self):
        pose = Pose()
        pose.position.x = 0.19
        pose.position.y = -0.08
        self.__position_pub.publish(pose)

    def grasp_cube(self):
        self.close_gripper()
        rospy.sleep(1)
        self.reset_pos()
        rospy.sleep(1)


class align_action:
    def __init__(self, arm_act: arm_action):
        self.__pid_x = PID()
        self.__pid_y = PID()
        self.__arm_action = arm_act

    def set_pid_param(self, Kp: float, Ki: float, Kd: float, sep_Ki_thres: float):
        self.__Ki = Ki

        self.__pid_x.tunings = (Kp, Ki, Kd)
        self.__pid_y.tunings = (Kp, Ki, Kd)
        self.__pid_x.reset()
        self.__pid_y.reset()
        self.__sep_Ki_thres = sep_Ki_thres

    def set_setpoint(self, setpoint: list):
        self.__setpoint = setpoint
        self.__pid_x.setpoint = setpoint[0]
        self.__pid_y.setpoint = setpoint[1]
        self.__pid_x.reset()
        self.__pid_y.reset()

    def set_sample_time(self, sample_time: float):
        self.__pid_x.sample_time = sample_time
        self.__pid_y.sample_time = sample_time

    def __cal_pid_vel(self, current_pos: list):
        if (
            np.linalg.norm(np.array(current_pos) - np.array(self.__setpoint))
            < self.__sep_Ki_thres
        ):
            self.__pid_x.Ki = 0
            self.__pid_y.Ki = 0
        else:
            self.__pid_x.Ki = self.__Ki
            self.__pid_y.Ki = self.__Ki

        vel_x = self.__pid_x(current_pos[0])
        vel_y = self.__pid_y(current_pos[1])

        return [vel_x, vel_y, 0.0]

    def align(self, current_pos: list):
        vel = self.__cal_pid_vel(current_pos)
        self.__arm_action.send_cmd_vel(vel)


if __name__ == "__main__":
    rospy.init_node("arm_ctrl", anonymous=True)
    my_arm_act = arm_action()
    my_align_act = align_action(1 / 30, my_arm_act)
    rospy.spin()
