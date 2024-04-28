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
        height = 0.005 + 0.055 * (place_layer - 1)
        self.set_arm(extension, height)

    def grasp(self, align_act, target_in_arm_base: list):
        # print(f"-------------------- {target_in_arm_base} --------------------")
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
        # rospy.sleep(0.5)

    def place(self, align_act):
        align_act.send_cmd_vel([0.0, 0.0, 0.0])
        ## stay still for 1 sec to ensure accuracy, 0.5sec proved to be too short
        rospy.sleep(0.3)
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
        rospy.loginfo("First align then place")
        align_act.send_cmd_vel([0.0, 0.0, 0.0])
        # rospy.sleep(0.5)
        self.place_pos(place_layer)
        rospy.sleep(0.5)
        rospy.loginfo(f"elevate gripper to layer {place_layer}")


class align_action:
    def __init__(self, arm_act: arm_action):
        """object reference"""
        self.__cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        """ state params """
        self.align_mode = AlignMode.OpenLoop
        self.__is_near_target_state_old = False
        self.x_sp = None  ## state, set point
        self.x_mv = None  ## state, measured variable
        """ config params """
        self.pid_cfg: dict[str, Union[float, PID]] = {
            "Ki": 0.0,
            "Isep": 0.0,
            "xctl": PID(),
            "yctl": PID(),
        }  ## Isep is integral separation
        self.open_cfg = {
            "v1": 0.3,
            "v2": 0.2,
            "offset_x": 0.1,
            "t_swtch": 1,
        }  ## Make sure that v1 and v2 are within range of max/min_vel of arm_action
        self.ss_cfg = {"decay": 1, "decay_near": 1.5, "dist_thresh": 0.1}
        self.align_angle = False
        self.max_vel = 0.3
        self.max_angular_vel = 0.3
        self.min_vel = 0.0
        self.min_angular_vel = 0.0

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

    def set_pid_param(self, Kp: float, Ki: float, Kd: float, sep_Ki_thres: float):
        ## alias
        xctl = self.pid_cfg["xctl"]
        yctl = self.pid_cfg["yctl"]
        ## set params
        xctl.tunings = (Kp, Ki, Kd)
        yctl.tunings = (Kp, Ki, Kd)
        xctl.reset()
        yctl.reset()
        self.pid_cfg["Isep"] = sep_Ki_thres
        self.pid_cfg["Ki"] = Ki

    def set_sample_time(self, sample_time: float):
        self.pid_cfg["xctl"].sample_time = sample_time
        self.pid_cfg["yctl"].sample_time = sample_time

    def __cal_pid_vel(self, measured_pos: list):
        if (
            np.linalg.norm(np.array(measured_pos[0:2]) - np.array(self.x_sp[0:2]))
            > self.pid_cfg["Isep"]
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

    def __cal_custom_vel(self, measured_pos: list):
        x, y = measured_pos[0:2]
        error = np.array(measured_pos) - np.array(self.x_sp)
        decay = 0
        if np.linalg.norm(error[0:2]) > self.ss_cfg["dist_thresh"]:
            decay = self.ss_cfg["decay"]
        else:
            decay = self.ss_cfg["decay_near"]

        vel_ang = decay * error[2] if self.align_angle else 0.0
        vel_x = decay * error[0] + y * vel_ang
        vel_y = decay * error[1] - x * vel_ang
        ## apply velocity limit
        vel_x = (
            np.clip(fabs(vel_x), self.min_vel, self.max_vel) * np.sign(vel_x)
            if fabs(vel_x) > 0.01
            else 0
        )
        vel_y = (
            np.clip(fabs(vel_y), self.min_vel, self.max_vel) * np.sign(vel_y)
            if fabs(vel_y) > 0.01
            else 0
        )
        vel_ang = (
            np.clip(fabs(vel_ang), self.min_angular_vel, self.max_angular_vel)
            * np.sign(vel_ang)
            if fabs(vel_ang) > 0.002
            else 0
        )
        return [vel_x, vel_y, vel_ang]

    def init_ctrl(self):
        """init all controllers for align_action afterwards"""
        self.__aligned = False  ## indicate if angle is aligned
        ## init pid controller (integral, ...)
        self.pid_cfg["xctl"].reset()
        self.pid_cfg["yctl"].reset()
        ## init state space controller (nothing to do here)
        pass
        ## init open loop controller ()
        self.t1_open = None
        self.t2_open = None
        return

    def set_state_sp(self, x_sp):
        """set target state (setpoint) of arm_ctrl"""
        self.x_sp = np.array(x_sp)
        if self.align_mode == AlignMode.PID:
            self.pid_cfg["xctl"].setpoint = self.x_sp[0]
            self.pid_cfg["yctl"].setpoint = self.x_sp[1]
        return

    def set_state_mv(self, x_mv):
        """set measure variable of arm_ctrl"""
        self.x_mv = np.array(x_mv)

    def align(self):
        """
        send cmd_vel by comparing x_mv and x_sp to align. i.e. the controller
        should be called at certain rate (e.g. 30Hz)
        """
        err = np.array(self.x_mv) - np.array(self.x_sp)
        vel = [0.0, 0.0, 0.0]
        align_mode = self.align_mode
        if self.align_angle and not self.__aligned:
            if abs(err[2]) > 0.1:
                align_mode = AlignMode.StateSpace
            else:
                self.__aligned = True
                self.pid_cfg["xctl"].reset()
                self.pid_cfg["yctl"].reset()
        ########## for debug ##########
        print(
            f"(AlignMode: {align_mode}) ctrl err: {100*err[0]:.2f}cm, {100*err[1]:.2f}cm, {np.rad2deg(err[2]):.1f}degree"
        )
        ###############################
        if align_mode == AlignMode.PID:
            vel = self.__cal_pid_vel(self.x_mv)
        elif align_mode == AlignMode.StateSpace:
            vel = self.__cal_custom_vel(self.x_mv)
        elif align_mode == AlignMode.OpenLoop:
            t = rospy.get_time()
            if self.t1_open == None:
                x, y = self.x_mv[:2] - self.x_sp[:2]
                x -= self.open_cfg["offset_x"]
                T1 = np.linalg.norm([x, y]) / self.open_cfg["v1"]
                self.t1_open = t + T1
                self.v1_open = [x / T1, y / T1, 0.0]
                print(f"open loop: PHASE1; vel: ({x/T1:.2f}, {y/T1:.2f})m/s")
            elif t < self.t1_open:
                vel = self.v1_open.copy()
            elif t < self.t1_open + self.open_cfg["t_swtch"]:
                ## brake to make better observation
                vel = [0.0, 0.0, 0.0]
            elif self.t2_open == None:
                x, y = self.x_mv[:2] - self.x_sp[:2]
                T2 = np.linalg.norm([x, y]) / self.open_cfg["v2"]
                self.t2_open = t + T2
                self.v2_open = [x / T2, y / T2, 0.0]
                print(f"open loop: PHASE2; vel: ({x/T2:.2f}, {y/T2:.2f})m/s")
            elif t < self.t2_open:
                vel = self.v2_open.copy()
        self.send_cmd_vel(vel)

    def finished(self, tolerance: list, timeout: float):
        if self.align_mode == AlignMode.OpenLoop:
            return self.t2_open is not None and rospy.get_time() > self.t2_open
        else:
            return self.is_near_target_state_sometime(tolerance, timeout)

    def stop(self):
        self.send_cmd_vel([0.0, 0.0, 0.0])

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
        x1, y1, ang1 = self.x_mv
        x2, y2, ang2 = self.x_sp
        x_th, y_th, ang_th = tolerance
        satisfy_xy = abs(x1 - x2) < abs(x_th) and abs(y1 - y2) < abs(y_th)
        satisfy_ang = True
        if self.align_mode == AlignMode.StateSpace and self.align_angle:
            satisfy_ang = abs(ang1 - ang2) < abs(ang_th)
        return satisfy_xy and satisfy_ang


if __name__ == "__main__":
    rospy.init_node("arm_ctrl", anonymous=True)
    my_arm_act = arm_action()
    my_align_act = align_action(my_arm_act)
    rospy.spin()
