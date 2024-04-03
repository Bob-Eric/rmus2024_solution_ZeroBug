#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import IntEnum
from math import fabs
from typing import Union
import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose, Twist
from simple_pid import PID
import tf2_ros
from geometry_msgs.msg import Pose, TransformStamped, Vector3


class AlignMode(IntEnum):
    OpenLoop = 0
    StateSpace = 1
    PID = 2


class arm_action:

    def __init__(self):
        self.__gripper_pub = rospy.Publisher("arm_gripper", Point, queue_size=10)
        self.__position_pub = rospy.Publisher("arm_position", Pose, queue_size=10)
        self.__cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.__cmd_vel_sub = rospy.Subscriber(
            "/cmd_vel", Twist, self.__cmd_vel_callback
        )
        self.__gripper_state_sub = rospy.Subscriber(
            "/gripper_state", Point, self.__gripper_state_callback
        )
        self.movement_start_time = 0.0
        self.movement_end_time = 0.0
        self.movement_active_time = 0.0
        self.__vel_old = [0.0, 0.0, 0.0]
        self.__can_arm_grasp_old = False
        self.max_vel = 0.3
        self.max_angular_vel = 0.3
        self.min_vel = 0.0
        self.min_angular_vel = 0.0
        self.align_mode = AlignMode.OpenLoop

    def __cmd_vel_callback(self, msg: Twist):
        vel = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.__vel = vel

        # only checks for the movement in the x-y plane
        if (
            np.linalg.norm(vel[0:2]) >= 0.2
            and np.linalg.norm(self.__vel_old[0:2]) < 0.2
        ):
            # get the time when the movement is active
            self.movement_active_time = rospy.get_time()
        if (
            np.linalg.norm(vel[0:2]) < 0.1
            and np.linalg.norm(self.__vel_old[0:2]) >= 0.1
        ):
            # get the time when the movement ends
            self.movement_end_time = rospy.get_time()
        if (
            np.linalg.norm(vel[0:2]) >= 0.1
            and np.linalg.norm(self.__vel_old[0:2]) < 0.1
        ):
            # get the time when the movement starts
            self.movement_start_time = rospy.get_time()

        self.__vel_old = vel

    def __gripper_state_callback(self, msg: Point):
        # 1 for grasped something, 0 for not
        self.__gripper_state = msg.x

    def can_arm_grasp_sometime(
        self, target_in_arm_base: list, timeout: float, align_mode: AlignMode
    ):

        return True

        if align_mode == AlignMode.OpenLoop:
            return True
        else:
            satisfy = self.can_arm_grasp(target_in_arm_base)
            if satisfy and not self.__can_arm_grasp_old:
                # check if the robot can grasp for the first time
                self.__can_arm_grasp_start_time = rospy.get_time()
                ...
            self.__can_arm_grasp_old = satisfy

            if satisfy and rospy.get_time() - self.__can_arm_grasp_start_time > timeout:
                # check if the robot can grasp for timeout
                return True
            else:
                return False

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

    def has_grasped(self) -> bool:
        return self.__gripper_state == 1

    def is_vel_active(self):
        return np.linalg.norm(self.__vel[0:2]) >= 0.2

    def send_cmd_vel(self, vel: list):
        vel[0] = max(self.min_vel, min(self.max_vel, fabs(vel[0]))) * np.sign(vel[0])
        vel[1] = max(self.min_vel, min(self.max_vel, fabs(vel[1]))) * np.sign(vel[1])
        vel[2] = max(
            self.min_angular_vel, min(self.max_angular_vel, fabs(vel[2]))
        ) * np.sign(vel[2])

        twist = Twist()
        twist.linear.z = 0.0
        twist.linear.x = vel[0]
        twist.linear.y = vel[1]
        twist.angular.z = vel[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        self.__cmd_vel_pub.publish(twist)
        # rospy.loginfo(f"send cmd_vel: {vel}")

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

    def place_fake(self):
        rospy.loginfo("<manipulator>: place fake cube")
        pose = Pose()
        pose.position.x = 0.21
        pose.position.y = -0.08
        self.__position_pub.publish(pose)
        rospy.sleep(2.0)
        self.open_gripper()
        rospy.sleep(2.0)
        self.reset_pos()

    def place_pos(self, place_layer: int = 1):
        rospy.loginfo("<manipulator>: now prepare to place (first layer)")
        pose = Pose()
        pose.position.x = 0.21
        pose.position.y = -0.04 + 0.055 * (place_layer - 1)
        self.__position_pub.publish(pose)

    def grasp_pos(self, target_in_arm_base: list):
        pose = Pose()

        # limit the grasp position 0.09 <= x <= 0.22, -0.08 <= y
        pose.position.x = max(min(target_in_arm_base[0], 0.22), 0.09)
        pose.position.y = max(target_in_arm_base[2], -0.08)
        rospy.loginfo(f"moving to grasp position: {pose.position.x}, {pose.position.y}")
        self.__position_pub.publish(pose)

    def grasp_cube(self):
        self.close_gripper()
        rospy.sleep(1)
        self.reset_pos()
        rospy.sleep(1)

    def go_and_grasp(self, target_in_arm_base: list, align_mode: AlignMode):
        print(f"-------------------- {target_in_arm_base} --------------------")
        if align_mode == AlignMode.OpenLoop:
            self.send_cmd_vel([0.25, 0.0, 0.0])
            rospy.sleep(0.3)
            self.send_cmd_vel([0.0, 0.0, 0.0])
            self.grasp_pos([0.19, 0.0, -0.08])
        else:
            self.send_cmd_vel([0.0, 0.0, 0.0])
            self.grasp_pos(target_in_arm_base)

        rospy.sleep(1.5)
        rospy.loginfo("Place: reach the goal for placing.")

        ## TODO: find out why example project do close_gripper() twice
        for _ in range(2):
            self.close_gripper()
            rospy.sleep(0.25)
        self.reset_pos()

        self.send_cmd_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.5)
        self.send_cmd_vel([0.0, 0.0, 0.0])

    def go_and_place(self):
        if self.align_mode == AlignMode.OpenLoop:
            self.send_cmd_vel([0.25, 0.0, 0.0])
            rospy.sleep(0.3)

        self.send_cmd_vel([0.0, 0.0, 0.0])
        ## stay still for 1 sec to ensure accuracy, 0.5sec proved to be too short
        rospy.sleep(1.0)
        rospy.loginfo("Place: reach the goal for placing.")

        self.open_gripper()
        rospy.sleep(2.0)

        self.send_cmd_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.5)
        self.reset_pos()
        self.send_cmd_vel([0.0, 0.0, 0.0])

    def preparation_for_grasp(self):
        rospy.loginfo("First align then grasp")
        rospy.loginfo("align to the right place")
        self.send_cmd_vel([0.0, 0.0, 0.0])
        rospy.sleep(0.5)
        self.open_gripper()
        rospy.sleep(2.0)

    def preparation_for_place(self, place_layer: int):
        self.send_cmd_vel([0.0, 0.0, 0.0])
        rospy.sleep(0.5)
        rospy.loginfo("First align then place")
        self.place_pos(place_layer)
        rospy.sleep(2.0)
        rospy.loginfo("adjusting arm pose for place.")


class align_action:
    def __init__(self, arm_act: arm_action):
        self.__arm_action = arm_act
        self.__base_link_pos_old = [0.0, 0.0]
        self.__delta_pos = 0.0
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        self.__base_link_pos_timer = rospy.Timer(
            rospy.Duration(2.5), self.__base_link_pos_callback
        )
        self.pid_cfg: dict[str, Union[float, PID]] = {
            "Ki": 0.0,
            "sep_dist": 0.0,
            "xctl": PID(),
            "yctl": PID(),
        }  ## Ki is for sep_dist
        self.ss_cfg = {"decay": 3, "decay_near": 9, "dist_thresh": 0.1}
        self.__decay = 3
        self.__decay_near = 9
        self.__decay_seperate_dist = 0.1
        self.__is_near_target_state_old = False
        self.align_angle = False
        self.align_mode = AlignMode.OpenLoop

    def __base_link_pos_callback(self, timer_event):
        base_link_tf_stamped: TransformStamped = self.tfBuffer.lookup_transform(
            "base_link", "map", rospy.Time(0), rospy.Duration(1.0)
        )
        base_link_pos_vec: Vector3 = base_link_tf_stamped.transform.translation
        base_link_pos = [base_link_pos_vec.x, base_link_pos_vec.y]
        self.__delta_pos = np.linalg.norm(
            np.array(base_link_pos) - np.array(self.__base_link_pos_old)
        )
        self.__base_link_pos_old = base_link_pos

    def set_pid_param(self, Kp: float, Ki: float, Kd: float, sep_Ki_thres: float):
        ## alias
        xctl = self.pid_cfg["xctl"]
        yctl = self.pid_cfg["yctl"]
        ## set params
        xctl.tunings = (Kp, Ki, Kd)
        yctl.tunings = (Kp, Ki, Kd)
        xctl.reset()
        yctl.reset()
        self.pid_cfg["sep_dist"] = sep_Ki_thres
        self.pid_cfg["Ki"] = sep_Ki_thres

    def set_sample_time(self, sample_time: float):
        self.pid_cfg["xctl"].sample_time = sample_time
        self.pid_cfg["yctl"].sample_time = sample_time

    def set_decay(self, decay: float):
        self.__decay = decay

    def __cal_pid_vel(self, measured_pos: list):
        if (
            np.linalg.norm(
                np.array(measured_pos[0:2]) - np.array(self.__target_state[0:2])
            ) > self.pid_cfg["sep_dist"]
        ):
            self.pid_cfg["xctl"].Ki = 0
            self.pid_cfg["yctl"].Ki = 0
        else:
            self.pid_cfg["xctl"].Ki = self.pid_cfg["Ki"]
            self.pid_cfg["yctl"].Ki = self.pid_cfg["Ki"]

        vel_x = -self.pid_cfg["xctl"](measured_pos[0])
        vel_y = -self.pid_cfg["yctl"](measured_pos[1])

        return [vel_x, vel_y, 0.0]

    def set_align_config(self, align_angle: bool, align_mode: AlignMode):
        self.align_angle = align_angle
        self.align_mode = align_mode
        self.__arm_action.align_mode = align_mode

    # x y theta
    def set_target_state(self, target_state: list):
        self.__target_state = target_state

        if self.align_mode == AlignMode.PID:
            self.pid_cfg["xctl"].setpoint = self.__target_state[0]
            self.pid_cfg["yctl"].setpoint = self.__target_state[1]
            self.pid_cfg["xctl"].reset()
            self.pid_cfg["yctl"].reset()
        elif self.align_mode == AlignMode.OpenLoop:
            self.__target_state[0] = 0.385
            self.__target_state[1] = 0.0
            self.pid_cfg["xctl"].setpoint = self.__target_state[0]
            self.pid_cfg["yctl"].setpoint = self.__target_state[1]
            self.pid_cfg["xctl"].reset()
            self.pid_cfg["yctl"].reset()

    def set_measured_state(self, measured_state: list):
        self.__measured_state = measured_state

    def __cal_custom_vel(self, measured_pos: list):
        x = measured_pos[0]
        y = measured_pos[1]
        error = np.array(measured_pos) - np.array(self.__target_state)

        decay = 0
        if np.linalg.norm(error[0:2]) > self.__decay_seperate_dist:
            decay = self.__decay
        else:
            decay = self.__decay_near

        vel_ang = decay * error[2] if self.align_angle else 0.0
        vel_x = decay * error[0] + y * vel_ang
        vel_y = decay * error[1] - x * vel_ang
        return [vel_x, vel_y, vel_ang]

    def align(self):
        vel = [0.0, 0.0, 0.0]
        if self.align_mode == AlignMode.PID or self.align_mode == AlignMode.OpenLoop:
            vel = self.__cal_pid_vel(self.__measured_state)
        elif self.align_mode == AlignMode.StateSpace:
            vel = self.__cal_custom_vel(self.__measured_state)
        self.__arm_action.send_cmd_vel(vel)

        # only check if robot is saturating for 2.5s
        if (
            self.__arm_action.is_vel_active()
            and rospy.get_time() - self.__arm_action.movement_active_time > 2.5
        ):
            if self.__delta_pos < 0.05:
                # robot is stuck
                rospy.logwarn("robot is stuck")
                reverse_vel_x = -np.sign(vel[0]) * 0.3
                reverse_vel_y = -np.sign(vel[1]) * 0.3
                self.__arm_action.send_cmd_vel([reverse_vel_x, reverse_vel_y, 0.0])
                rospy.sleep(0.5)
                self.__arm_action.send_cmd_vel([0.0, 0.0, 0.0])
                rospy.sleep(0.5)
                return False

        return True

    def is_near_target_state_sometime(self, tolerance: list, timeout: float):
        satisfy = self.is_near_target_state(tolerance)
        if satisfy and not self.__is_near_target_state_old:
            # check if the robot is near the target state for the first time
            self.__near_target_state_start_time = rospy.get_time()
            ...

        self.__is_near_target_state_old = satisfy

        if satisfy and rospy.get_time() - self.__near_target_state_start_time > timeout:
            # check if the robot is near the target state for timeout
            return True
        else:
            return False

    def is_near_target_state(self, tolerance: list):
        x1, y1, ang1 = self.__measured_state
        x2, y2, ang2 = self.__target_state
        x_th, y_th, ang_th = tolerance
        satisfy_xy = abs(x1 - x2) < abs(x_th) and abs(y1 - y2) < abs(y_th)
        satisfy_ang = abs(ang1 - ang2) < abs(ang_th) if self.align_angle else True
        return satisfy_xy and satisfy_ang

    def is_target_state_too_faraway(self):
        return self.__measured_state[0] <= -0.5 or abs(self.__measured_state[1]) >= 2.0


if __name__ == "__main__":
    rospy.init_node("arm_ctrl", anonymous=True)
    my_arm_act = arm_action()
    my_align_act = align_action(my_arm_act)
    rospy.spin()
