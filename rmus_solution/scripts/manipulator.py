#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as sciR
import rospy
from geometry_msgs.msg import Pose
from rmus_solution.srv import graspsignal, graspsignalRequest, graspsignalResponse
from rmus_solution.msg import MarkerInfo, MarkerInfoList
import tf2_ros
import tf2_geometry_msgs
from dynamic_reconfigure.server import Server
from rmus_solution.cfg import manipulater_PIDConfig
from enum import IntEnum
from arm_ctrl import arm_action, align_action


class AlignRequest(IntEnum):
    Reset = 0
    Grasp = 1
    Place = 2


class ErrorCode(IntEnum):
    Success = 0
    # Marker info is too old, latency detected
    Latency = 1
    # Marker ID is invalid
    InvalidMarkerID = 2
    # Layer is invalid
    InvalidLayer = 3
    # Robot is stuck during the movement
    Stuck = 4
    # Target is too far away, cannot grasp/place just by pid control
    TargetTooFaraway = 5
    # Gripper is failed to grasp the cube, maybe the esitmated position is wrong
    Fail = 6


class manipulator:

    def __init__(self) -> None:
        self.arm_act = arm_action()
        self.align_act = align_action(self.arm_act)

        self.current_marker_poses = Pose()
        self.image_time_now = 0
        self.desired_marker_id = 0

        self.cube_goal_tolerance = 0.01
        self.tag_goal_tolerance = 0.01

        ############### Dynamic params ###############
        self.ros_rate = 10
        self.Kp = 0.5
        self.Ki = 0.0
        self.Kd = 0.0
        self.desired_cube_pos_base_link = [0.5, 0.0]
        self.desired_tag_pos_base_link = [0.5, 0.0]
        ############### Dynamic params ###############

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        self.__marker_pose_sub = rospy.Subscriber(
            "/get_blockinfo", MarkerInfoList, self.marker_pose_callback, queue_size=1
        )
        self.__service = rospy.Service(
            "/let_manipulater_work", graspsignal, self.grasp_signal_callback
        )
        self.__dynamic_reconfigure_server = Server(
            manipulater_PIDConfig, self.dynamic_reconfigure_callback
        )

    def marker_pose_callback(self, msg: MarkerInfoList):
        """update self.current_marker_poses of self.desired_cube_id"""
        for markerInfo in msg.markerInfoList:
            markerInfo: MarkerInfo
            if markerInfo.id == self.desired_marker_id:
                self.current_marker_poses = markerInfo.pose
                self.image_time_now = rospy.get_time()

    def grasp_signal_callback(self, req: graspsignalRequest):
        self.desired_marker_id = req.marker_id
        # Reset the arm
        if req.mode == AlignRequest.Reset:
            self.arm_act.reset_pos()
            self.arm_act.open_gripper()
            resp = graspsignalResponse()
            resp.res = True
            resp.response = "reset arm position and open gripper"
            resp.error_code = ErrorCode.Success
            rospy.loginfo(resp.response)
            return resp

        # If image_time is too old, rejects to grasp or place cube, instead just go back for sometime
        initial_time = rospy.get_time()
        while rospy.get_time() - self.image_time_now > 0.1:
            rospy.loginfo("latency detected!")

            # Go back for 0.5s
            if rospy.get_time() - initial_time > 1.0:
                self.arm_act.send_cmd_vel([-0.2, 0.0, 0.0])
                rospy.sleep(0.1)
                self.arm_act.send_cmd_vel([0.0, 0.0, 0.0])
            if rospy.get_time() - initial_time > 2.0:
                resp = graspsignalResponse()
                resp.res = False
                resp.response = "Can't get the marker info in time"
                resp.error_code = ErrorCode.Latency
                rospy.logwarn(resp.response)
                return resp
            rospy.sleep(0.1)

        rate = rospy.Rate(self.ros_rate)

        if req.mode == AlignRequest.Grasp:
            resp = self.grasp_cube_resp(rate)
            return resp

        elif req.mode == AlignRequest.Place:
            resp = self.place_cube_resp(rate, req.layer)
            return resp
        else:
            rospy.logerr("Invalid mode")
            resp = graspsignalResponse()
            resp.res = False
            resp.error_code = ErrorCode.Fail
            resp.response = "Invalid mode"
            rospy.logwarn(resp.response)
            return resp

    def grasp_cube_resp(self, rate):
        resp = graspsignalResponse()

        if not 1 <= self.desired_marker_id <= 6:
            resp.res = False
            resp.response = "Invalid marker id"
            resp.error_code = ErrorCode.InvalidMarkerID
            rospy.logwarn(f"Invalid marker id: {self.desired_marker_id}")
            return resp

        self.arm_act.preparation_for_grasp()

        self.align_act.set_setpoint(self.desired_cube_pos_base_link)

        while not rospy.is_shutdown():
            target_marker_pose = self.current_marker_poses

            measured_pos, _ = self.trans_cam_frame_to_target_frame(target_marker_pose)
            cube_in_arm_base, _ = self.trans_cam_frame_to_target_frame(
                target_marker_pose, "arm_base"
            )
            self.align_act.set_measured_point(measured_pos)

            if self.arm_act.can_arm_grasp(cube_in_arm_base):
                self.arm_act.go_and_grasp(cube_in_arm_base)
                rospy.sleep(1.0)
                if self.arm_act.has_grasped():
                    resp.res = True
                    resp.response = "Successfully Grasp"
                    resp.error_code = ErrorCode.Success
                    rospy.loginfo(resp.response)
                else:
                    resp.res = False
                    resp.response = "Fail to grasp"
                    resp.error_code = ErrorCode.Fail
                    rospy.logwarn(resp.response)
                break
            elif self.align_act.is_setpoint_too_faraway():
                resp.res = False
                resp.response = "Target is too far away"
                resp.error_code = ErrorCode.TargetTooFaraway
                rospy.logwarn(resp.response)
                self.arm_act.send_cmd_vel([0.0, 0.0, 0.0])
                rospy.sleep(0.5)
                break

            elif not self.align_act.align():
                resp.res = False
                resp.error_code = ErrorCode.Stuck
                resp.response = "Robot is stuck, goint backward!"
                rospy.logwarn(resp.response)
                break

            rate.sleep()
        return resp

    def place_cube_resp(self, rate, place_layer: int = 1):
        resp = graspsignalResponse()

        if 1 <= self.desired_marker_id <= 6:
            # align to the cube
            ...
        elif 7 <= self.desired_marker_id <= 9:
            # align to the marker
            ...
        else:
            resp.res = False
            resp.response = "Invalid marker id"
            resp.error_code = ErrorCode.InvalidMarkerID
            rospy.logwarn(f"Invalid marker id: {self.desired_marker_id}")
            return resp

        if 1 <= place_layer <= 3:
            self.arm_act.preparation_for_place(place_layer)
        else:
            rospy.logerr("place_layer should be 1, 2 or 3")
            resp.res = False
            resp.response = "Invalid layer"
            resp.error_code = ErrorCode.InvalidLayer
            rospy.logwarn(resp.response)
            return resp

        self.align_act.set_setpoint(self.desired_tag_pos_base_link)

        while not rospy.is_shutdown():
            target_marker_pose = self.current_marker_poses

            measured_pos, _ = self.trans_cam_frame_to_target_frame(target_marker_pose)
            self.align_act.set_measured_point(measured_pos)
            if self.align_act.is_near_setpoint(self.tag_goal_tolerance):
                self.arm_act.go_and_place()

                resp.res = True
                resp.error_code = ErrorCode.Success
                resp.response = "Successfully Place"
                rospy.loginfo(resp.response)
                break
            elif not self.align_act.align():
                resp.res = False
                resp.error_code = ErrorCode.Stuck
                resp.response = "Robot is stuck, goint backward!"
                rospy.logwarn(resp.response)
                break
            rate.sleep()
        return resp

    def dynamic_reconfigure_callback(self, config: dict, level: int):
        self.ros_rate = config["control_frequency"]
        self.desired_cube_pos_base_link[0] = config["desired_cube_x"]
        self.desired_cube_pos_base_link[1] = config["desired_cube_y"]
        self.Kp = config["Kp"]
        self.Ki = config["Ki"]
        self.Kd = config["Kd"]
        self.desired_tag_pos_base_link[0] = config["desired_tag_x"]
        self.desired_tag_pos_base_link[1] = config["desired_tag_y"]
        self.seperate_I_threshold = config["seperate_I_threshold"]

        self.align_act.set_pid_param(
            self.Kp, self.Ki, self.Kd, self.seperate_I_threshold
        )
        self.align_act.set_sample_time(1.0 / self.ros_rate)

        return config

    def trans_cam_frame_to_target_frame(
        self, pose_in_cam: Pose, target_frame="base_link"
    ):
        if not self.tfBuffer.can_transform(
            target_frame,
            "camera_aligned_depth_to_color_frame_correct",
            rospy.Time.now(),
        ):
            rospy.logerr(
                f"pick_and_place: cannot find transform between {target_frame} and camera_aligned_depth_to_color_frame_correct"
            )
            return None, None
        posestamped_in_cam = tf2_geometry_msgs.PoseStamped()
        posestamped_in_cam.header.stamp = rospy.Time.now()
        posestamped_in_cam.header.frame_id = (
            "camera_aligned_depth_to_color_frame_correct"
        )
        posestamped_in_cam.pose = pose_in_cam
        posestamped_in_base = self.tfBuffer.transform(posestamped_in_cam, target_frame)
        pose_in_base = posestamped_in_base.pose
        pos = [
            pose_in_base.position.x,
            pose_in_base.position.y,
            pose_in_base.position.z,
        ]

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


if __name__ == "__main__":
    rospy.init_node("manipulater_node", anonymous=True)
    rter = manipulator()

    rospy.spin()
