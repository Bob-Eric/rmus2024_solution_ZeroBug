#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import numpy as np
from scipy.spatial.transform import Rotation as sciR
import rospy
from geometry_msgs.msg import Twist, Pose, Point
from rmus_solution.srv import graspsignal, graspsignalRequest, graspsignalResponse
from rmus_solution.msg import MarkerInfo, MarkerInfoList
import tf2_ros
import tf2_geometry_msgs
from simple_pid import PID
from dynamic_reconfigure.server import Server
from rmus_solution.cfg import manipulater_PIDConfig
from enum import IntEnum
from rmus_solution.msg import MarkerInfo


class TrimerworkRequest(IntEnum):
    Reset = 0
    Grasp = 1
    Place = 2
    PlaceHighly = 3


class manipulater:

    def __init__(self) -> None:
        self.arm_gripper_pub = rospy.Publisher("arm_gripper", Point, queue_size=2)
        self.arm_position_pub = rospy.Publisher("arm_position", Pose, queue_size=2)
        self.cmd_vel_puber = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(
            "/get_blockinfo", MarkerInfoList, self.markerPoseCb, queue_size=1
        )

        self.current_marker_poses = Pose()
        self.image_time_now = 0.0

        self.desired_cube_id = 0

        self.desired_cube_pos_in_cam = [0.385, 0.00]
        self.xy_goal_tolerance = [0.02, 0.01]
        pid_cal_time = 1 / 30

        self.pid_P = 4.0
        self.pid_I = 8.0
        self.pid_D = 0.0
        self.xy_seperate_I_threshold = [0.1, 0.1]

        self.pos_x_pid = PID(
            self.pid_P,
            self.pid_I,
            self.pid_D,
            self.desired_cube_pos_in_cam[0],
            pid_cal_time,
            (-0.5, 0.5),
        )
        self.pos_y_pid = PID(
            self.pid_P,
            self.pid_I,
            self.pid_D,
            self.desired_cube_pos_in_cam[1],
            pid_cal_time,
            (-0.5, 0.5),
        )

        self.target_pos_old = np.array([100, 100, 100])

        self.ros_rate = 30
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.service = rospy.Service(
            "/let_manipulater_work", graspsignal, self.trimerworkCallback
        )
        self.server = Server(manipulater_PIDConfig, self.dynamic_reconfigure_callback)

    def dynamic_reconfigure_callback(self, config: dict, level: int):
        self.pid_P = config["Kp"]
        self.pid_I = config["Ki"]
        self.pid_D = config["Kd"]
        self.pos_x_pid.Kp = self.pid_P
        self.pos_x_pid.Ki = self.pid_I
        self.pos_x_pid.Kd = self.pid_D
        self.pos_y_pid.Kp = self.pid_P
        self.pos_y_pid.Ki = self.pid_I
        self.pos_y_pid.Kd = self.pid_D
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.xy_seperate_I_threshold = [
            config["x_seperate_I_threshold"],
            config["y_seperate_I_threshold"],
        ]
        return config

    def markerPoseCb(self, msg: MarkerInfoList):
        for markerInfo in msg.markerInfoList:
            markerInfo: MarkerInfo
            if markerInfo.id == self.desired_cube_id:
                self.current_marker_poses = markerInfo.pose
                self.image_time_now = rospy.get_time()

    def getTargetPosAndAngleInBaseLinkFrame(self, pose_in_cam: Pose):
        if not self.tfBuffer.can_transform(
            "base_link", "camera_aligned_depth_to_color_frame_correct", rospy.Time.now()
        ):
            rospy.logerr(
                "pick_and_place: cannot find transform between base_link and camera_aligned_depth_to_color_frame_correct"
            )
            return None, None
        posestamped_in_cam = tf2_geometry_msgs.PoseStamped()
        posestamped_in_cam.header.stamp = rospy.Time.now()
        posestamped_in_cam.header.frame_id = (
            "camera_aligned_depth_to_color_frame_correct"
        )
        posestamped_in_cam.pose = pose_in_cam
        posestamped_in_base = self.tfBuffer.transform(posestamped_in_cam, "base_link")
        pose_in_base = posestamped_in_base.pose
        pos = np.array(
            [pose_in_base.position.x, pose_in_base.position.y, pose_in_base.position.z]
        )
        quat = np.array(
            [
                pose_in_cam.orientation.x,
                pose_in_cam.orientation.y,
                pose_in_cam.orientation.z,
                pose_in_cam.orientation.w,
            ]
        )
        angle = sciR.from_quat(quat).as_euler("YXZ")[0]

        return pos, angle

    def trimerworkCallback(self, req: graspsignalRequest):
        self.desired_cube_id = req.cube_id
        # Reset the arm
        if req.mode == TrimerworkRequest.Reset:
            self.arm_reset()
            rospy.sleep(0.2)
            self.open_gripper()
            resp = graspsignalResponse()
            resp.res = True
            resp.response = "reset arm position and open gripper"
            return resp

        # If image_time is too old, rejects to grasp or place cube, instead just go back for sometime
        initial_time = rospy.get_time()
        while rospy.get_time() - self.image_time_now > 0.1:
            rospy.loginfo("latency detected!")

            # Go back for 0.5s
            if rospy.get_time() - initial_time > 1.0:
                self.sendBaseVel([-0.2, 0.0, 0.0])
                rospy.sleep(0.1)
                self.sendBaseVel([0.0, 0.0, 0.0])
            if rospy.get_time() - initial_time > 2.0:
                resp = graspsignalResponse()
                resp.res = True
                resp.response = "Successfully Grasp fake"
                return resp
            rospy.sleep(0.1)

        rate = rospy.Rate(self.ros_rate)

        if req.mode == TrimerworkRequest.Grasp:
            resp = self.grasp_cube(rate)
            return resp

        elif req.mode == TrimerworkRequest.Place:
            resp = self.place_cube(rate, False)
            return resp
        elif req.mode == TrimerworkRequest.PlaceHighly:
            resp = self.place_cube(rate, True)
            return resp

    def grasp_cube(self, rate):
        rospy.loginfo("First trim then grasp")
        rospy.loginfo("Trim to the right place")
        self.sendBaseVel([0.0, 0.0, 0.0])
        rospy.sleep(2.0)
        self.open_gripper()
        rospy.sleep(0.1)
        resp = graspsignalResponse()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()

        not_update_cnt = 0

        while not rospy.is_shutdown():
            target_marker_pose = self.current_marker_poses
            if target_marker_pose is None:
                continue

            target_pos, target_angle = self.getTargetPosAndAngleInBaseLinkFrame(
                target_marker_pose
            )

            # if np.linalg.norm(target_pos - self.target_pos_old) < 0.001:
            #     if not_update_cnt > 60:
            #         rospy.loginfo("target_pos is not updated")
            #         resp.res = False
            #         return resp
            #     not_update_cnt += 1
            # else:
            #     not_update_cnt = 0
            self.target_pos_old = target_pos

            if self.is_near_desired_position(target_pos):
                self.sendBaseVel([0.25, 0.0, 0.0])
                rospy.sleep(0.3)
                self.sendBaseVel([0.0, 0.0, 0.0])
                rospy.sleep(1.0)
                self.arm_grasp_pos()
                rospy.sleep(1.0)
                rospy.loginfo("Place: reach the goal for placing.")

                target_marker_pose = self.current_marker_poses
                self.close_gripper()
                rospy.sleep(1.0)

                self.close_gripper()
                rospy.sleep(1.0)
                self.arm_reset()

                self.sendBaseVel([-0.3, 0.0, 0.0])
                rospy.sleep(0.4)
                self.sendBaseVel([0.0, 0.0, 0.0])

                resp.res = True
                resp.response = str(target_angle)
                break
            else:
                self.cal_cmd_vel_pid(target_pos)

            rate.sleep()
        return resp

    def place_cube(self, rate, should_place_highly=False):
        self.sendBaseVel([0.0, 0.0, 0.0])
        rospy.sleep(2.0)
        rospy.loginfo("First trim then place")
        if should_place_highly:
            self.arm_place_highly_pos()
        else:
            self.arm_place_pos()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        while not rospy.is_shutdown():
            target_marker_pose = self.current_marker_poses
            if target_marker_pose is None:
                continue

            target_pos, target_angle = self.getTargetPosAndAngleInBaseLinkFrame(
                target_marker_pose
            )
            if self.is_near_desired_position(target_pos):
                rospy.loginfo("Trim well in the all dimention, going open loop")
                self.sendBaseVel([0.0, 0.0, 0.0])
                rospy.sleep(1.0)
                self.sendBaseVel([0.2, 0.0, 0.0])
                rospy.sleep(0.2)
                self.sendBaseVel([0.2, 0.0, 0.0])
                rospy.sleep(0.2)
                self.sendBaseVel([0.0, 0.0, 0.0])
                rospy.loginfo("Place: reach the goal for placing.")

                rospy.loginfo("Trim well in the horizon dimention")

                target_pos, target_angle = self.getTargetPosAndAngleInBaseLinkFrame(
                    self.current_marker_poses
                )
                self.sendBaseVel([0.0, 0.0, 0.0])
                rospy.sleep(1.0)
                self.open_gripper()
                rospy.sleep(1.0)
                reset_thread = threading.Thread(target=self.arm_reset)
                reset_thread.start()

                self.sendBaseVel([-0.3, 0.0, 0.0])
                rospy.sleep(0.6)
                self.sendBaseVel([0.0, 0.0, 0.0])

                resp = graspsignalResponse()
                resp.res = True
                resp.response = "Successfully Place"
                break

            else:
                self.cal_cmd_vel_pid(target_pos)

            rate.sleep()
        return resp

    def is_near_desired_position(self, target_pos):
        return (
            abs(target_pos[0] - self.desired_cube_pos_in_cam[0])
            <= self.xy_goal_tolerance[0]
            and abs(target_pos[1] - self.desired_cube_pos_in_cam[1])
            <= self.xy_goal_tolerance[1]
        )

    def cal_cmd_vel_pid(self, target_pos):
        cmd_vel = [0.0, 0.0, 0.0]

        if (
            abs(target_pos[0] - self.desired_cube_pos_in_cam[0])
            < self.xy_seperate_I_threshold[0]
        ):
            self.pos_x_pid.Ki = self.pid_I
        else:
            self.pos_x_pid.Ki = 0.0

        if (
            abs(target_pos[1] - self.desired_cube_pos_in_cam[1])
            < self.xy_seperate_I_threshold[1]
        ):
            self.pos_y_pid.Ki = self.pid_I
        else:
            self.pos_y_pid.Ki = 0.0

        output_x = -self.pos_x_pid(target_pos[0])
        output_y = -self.pos_y_pid(target_pos[1])
        # rospy.loginfo(f"output_x: {output_x}, output_y: {output_y}")
        cmd_vel[0] = output_x
        cmd_vel[1] = output_y
        cmd_vel[2] = 0
        self.sendBaseVel(cmd_vel)

    def arm_grasp_pos(self):
        pose = Pose()
        pose.position.x = 0.19
        pose.position.y = -0.08
        self.arm_position_pub.publish(pose)

    def sendBaseVel(self, vel: list):
        twist = Twist()
        twist.linear.z = 0.0
        twist.linear.x = vel[0]
        twist.linear.y = vel[1]
        twist.angular.z = vel[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        self.cmd_vel_puber.publish(twist)

    def open_gripper(self):
        open_gripper_msg = Point()
        open_gripper_msg.x = 0.0
        open_gripper_msg.y = 0.0
        open_gripper_msg.z = 0.0
        rospy.loginfo("open the gripper")
        self.arm_gripper_pub.publish(open_gripper_msg)

    def close_gripper(self):
        close_gripper_msg = Point()
        close_gripper_msg.x = 1.0
        close_gripper_msg.y = 0.0
        close_gripper_msg.z = 0.0
        rospy.loginfo("close the gripper")
        self.arm_gripper_pub.publish(close_gripper_msg)

    def arm_reset(self):
        reset_arm_msg = Pose()
        reset_arm_msg.position.x = 0.1
        reset_arm_msg.position.y = 0.12
        reset_arm_msg.position.z = 0.0
        reset_arm_msg.orientation.x = 0.0
        reset_arm_msg.orientation.y = 0.0
        reset_arm_msg.orientation.z = 0.0
        reset_arm_msg.orientation.w = 0.0
        rospy.loginfo("reset the arm")
        self.arm_position_pub.publish(reset_arm_msg)

    def arm_place_pos(self):
        rospy.loginfo("<manipulater>: now prepare to grip")
        pose = Pose()
        pose.position.x = 0.21
        pose.position.y = -0.02
        self.arm_position_pub.publish(pose)

    def arm_place_highly_pos(self):
        rospy.loginfo("<manipulater>: now prepare to grip")
        pose = Pose()
        pose.position.x = 0.21
        pose.position.y = 0.03
        self.arm_position_pub.publish(pose)


if __name__ == "__main__":
    rospy.init_node("manipulater_node", anonymous=True)
    rter = manipulater()

    rospy.spin()
