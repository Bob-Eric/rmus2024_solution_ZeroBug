#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sympy import false
import rospy
import numpy as np
from geometry_msgs.msg import Point, Pose, Twist
from simple_pid import PID
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, TransformStamped, Vector3


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
        self.movement_saturat_time = 0.0
        self.__vel_old = [0.0, 0.0, 0.0]

    def __cmd_vel_callback(self, msg: Twist):
        vel = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.__vel = vel

        # only checks for the movement in the x-y plane
        if (
            np.linalg.norm(vel[0:2]) >= 0.5
            and np.linalg.norm(self.__vel_old[0:2]) < 0.5
        ):
            # get the time when the movement saturates
            self.movement_saturat_time = rospy.get_time()
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

    def has_grasped(self):
        return self.__gripper_state == 1

    def is_vel_saturating(self):
        return np.linalg.norm(self.__vel[0:2]) >= 0.5

    def send_cmd_vel(self, vel: list):
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

    def go_and_grasp(self, target_in_arm_base: list):
        self.send_cmd_vel([0.0, 0.0, 0.0])
        self.grasp_pos(target_in_arm_base)
        rospy.sleep(0.5)
        rospy.loginfo("Place: reach the goal for placing.")

        self.close_gripper()
        rospy.sleep(0.5)

        self.close_gripper()
        rospy.sleep(0.5)
        self.reset_pos()

        self.send_cmd_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.5)
        self.send_cmd_vel([0.0, 0.0, 0.0])

    def go_and_place(self):
        rospy.loginfo("Align well in the all dimention, going open loop")
        rospy.loginfo("Place: reach the goal for placing.")
        rospy.loginfo("Align well in the horizon dimention")

        self.send_cmd_vel([0.0, 0.0, 0.0])
        ## stay still for 1 sec to ensure accuracy, 0.5sec proved to be too short
        rospy.sleep(2)
        self.open_gripper()
        rospy.sleep(0.5)

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
        rospy.sleep(0.1)

    def preparation_for_place(self, place_layer: int):
        self.send_cmd_vel([0.0, 0.0, 0.0])
        rospy.sleep(0.5)
        rospy.loginfo("First align then place")
        self.place_pos(place_layer)
        ...


class align_action:
    def __init__(self, arm_act: arm_action):
        self.__pid_x = PID()
        self.__pid_y = PID()
        self.__arm_action = arm_act
        self.__base_link_pos_old = [0.0, 0.0]
        self.__delta_pos = 0.0
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        self.__base_link_pos_timer = rospy.Timer(
            rospy.Duration(2.5), self.__base_link_pos_callback
        )

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

    def set_measured_point(self, measured_point: list):
        self.__measured_point = measured_point

    def __cal_pid_vel(self, measured_pos: list):
        if (
            np.linalg.norm(np.array(measured_pos[0:2]) - np.array(self.__setpoint))
            > self.__sep_Ki_thres
        ):
            self.__pid_x.Ki = 0
            self.__pid_y.Ki = 0
        else:
            self.__pid_x.Ki = self.__Ki
            self.__pid_y.Ki = self.__Ki

        vel_x = -self.__pid_x(measured_pos[0])
        vel_y = -self.__pid_y(measured_pos[1])

        return [vel_x, vel_y, 0.0]

    def align(self):
        vel = self.__cal_pid_vel(self.__measured_point)
        self.__arm_action.send_cmd_vel(vel)

        # only check if robot is saturating for 2.5s
        if (
            self.__arm_action.is_vel_saturating()
            and rospy.get_time() - self.__arm_action.movement_saturat_time > 2.5
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

    def is_near_setpoint(self, tolerance: float):
        satisfy = (
            np.linalg.norm(
                np.array(self.__measured_point[0:2]) - np.array(self.__setpoint)
            )
            <= tolerance
        )
        return satisfy

    def is_setpoint_too_faraway(self):
        return self.__measured_point[0] <= -0.5 or abs(self.__measured_point[1]) >= 2.0


if __name__ == "__main__":
    rospy.init_node("arm_ctrl", anonymous=True)
    my_arm_act = arm_action()
    my_align_act = align_action(my_arm_act)
    rospy.spin()
