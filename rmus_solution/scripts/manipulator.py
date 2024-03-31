#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as sciR
import rospy
from geometry_msgs.msg import Pose
from rmus_solution.srv import (
    graspsignal,
    graspsignalRequest,
    graspsignalResponse,
    graspconfig,
    graspconfigRequest,
    graspconfigResponse,
)
from rmus_solution.msg import MarkerInfo, MarkerInfoList
import tf2_ros
import tf2_geometry_msgs
from dynamic_reconfigure.server import Server
from rmus_solution.cfg import manipulater_PIDConfig
from enum import IntEnum
from arm_ctrl import arm_action, align_action, AlignMode


class AlignRequest(IntEnum):
    Reset = 0
    Grasp = 1
    Place = 2
    PlaceFake = 3


prefix = "[manipulator]"


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
    # Timeout
    TImeout = 6
    # Gripper is failed to grasp the cube, maybe the esitmated position is wrong
    Fail = 7


arm_base_frame = "arm_base"
camera_frame = "camera_aligned_depth_to_color_frame_correct"
base_link_frame = "base_link"


class manipulator:

    def __init__(self) -> None:
        self.arm_act = arm_action()
        self.align_act = align_action(self.arm_act)

        self.current_marker_poses = Pose()
        self.image_time_now = 0
        self.desired_marker_id = 0

        self.state_tolerance = [0.01, 0.01, 0.1]

        ############### Dynamic params ###############
        self.ros_rate = 10
        self.timeout = 10
        self.Kp = 0.5
        self.Ki = 0.0
        self.Kd = 0.0
        self.desired_cube_pos_arm_base = [0.5, 0.0]
        self.desired_tag_pos_arm_base = [0.5, 0.0]
        self.max_velocity = 0.5
        self.min_velocity = 0.1
        self.max_angular_velocity = 0.5
        self.min_angular_velocity = 0.1
        ############### Dynamic params ###############

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        self.__marker_pose_sub = rospy.Subscriber(
            "/get_blockinfo", MarkerInfoList, self.marker_pose_callback, queue_size=1
        )
        self.__grasp_signal_service = rospy.Service(
            "/let_manipulater_work", graspsignal, self.grasp_signal_callback
        )
        self.__grasp_config_service = rospy.Service(
            "/manipulater_config", graspconfig, self.grasp_config_callback
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
            rospy.loginfo(prefix + "latency detected!")

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
        elif req.mode == AlignRequest.PlaceFake:
            self.arm_act.place_fake()
            resp = graspsignalResponse()
            resp.res = True
            resp.response = "Place fake cube"
            resp.error_code = ErrorCode.Success
            rospy.loginfo(resp.response)
            return resp
        else:
            rospy.logerr(prefix + "Invalid mode")
            resp = graspsignalResponse()
            resp.res = False
            resp.error_code = ErrorCode.Fail
            resp.response = "Invalid mode"
            rospy.logwarn(resp.response)
            return resp

    def grasp_config_callback(self, req: graspconfigRequest):
        resp = graspconfigResponse()
        self.align_act.set_align_config(req.align_angle, AlignMode(req.align_mode))
        resp.res = True
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

        # self.align_act.set_setpoint(self.desired_cube_pos_arm_base)

        desired_pose_arm_base = Pose()
        desired_pose_arm_base.position.x = self.desired_cube_pos_arm_base[0]
        desired_pose_arm_base.position.y = self.desired_cube_pos_arm_base[1]
        desired_pose_arm_base.position.z = 0.0
        desired_pose_arm_base.orientation.x = 0.0
        desired_pose_arm_base.orientation.y = 0.0
        desired_pose_arm_base.orientation.z = 0.0
        desired_pose_arm_base.orientation.w = 1.0

        desidre_cube_pos_base_link, desired_tag_ang_base_link = self.transfer_frame(
            desired_pose_arm_base,
            source_frame=arm_base_frame,
            target_frame=base_link_frame,
        )

        target_state = [
            desidre_cube_pos_base_link[0],
            desidre_cube_pos_base_link[1],
            desired_tag_ang_base_link,
        ]
        self.align_act.set_target_state(target_state)

        max_time = rospy.get_time() + self.timeout

        while not rospy.is_shutdown():
            if rospy.get_time() > max_time:
                resp.res = False
                resp.response = "Timeout"
                resp.error_code = ErrorCode.TImeout
                rospy.logwarn(resp.response)
                break

            marker_pose_in_cam = self.current_marker_poses

            marker_in_base_link, marker_ang_in_base_link = self.transfer_frame(
                marker_pose_in_cam,
                source_frame=camera_frame,
                target_frame=base_link_frame,
            )
            marker_in_arm_base, marker_ang_in_arm_base = self.transfer_frame(
                marker_pose_in_cam,
                source_frame=camera_frame,
                target_frame=arm_base_frame,
            )
            # self.align_act.set_measured_point(marker_in_arm_base)
            measured_state = [
                marker_in_base_link[0],
                marker_in_base_link[1],
                marker_ang_in_base_link,
            ]
            self.align_act.set_measured_state(measured_state)
            rospy.loginfo(f"error: {np.array( measured_state)-np.array(target_state)}")

            if self.arm_act.can_arm_grasp_sometime(marker_in_arm_base, 1.0):
                self.arm_act.go_and_grasp(marker_in_arm_base)
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
            rospy.logwarn(prefix + f"Invalid layer: {place_layer}")
            resp.res = False
            resp.response = "Invalid layer"
            resp.error_code = ErrorCode.InvalidLayer
            rospy.logwarn(resp.response)
            return resp

        desired_pose_arm_base = Pose()
        desired_pose_arm_base.position.x = self.desired_tag_pos_arm_base[0]
        desired_pose_arm_base.position.y = self.desired_tag_pos_arm_base[1]
        desired_pose_arm_base.position.z = 0.0
        desired_pose_arm_base.orientation.x = 0.0
        desired_pose_arm_base.orientation.y = 0.0
        desired_pose_arm_base.orientation.z = 0.0
        desired_pose_arm_base.orientation.w = 1.0

        desidre_tag_pos_base_link, desired_tag_ang_base_link = self.transfer_frame(
            desired_pose_arm_base,
            source_frame=arm_base_frame,
            target_frame=base_link_frame,
        )

        target_state = [
            desidre_tag_pos_base_link[0],
            desidre_tag_pos_base_link[1],
            desired_tag_ang_base_link,
        ]
        self.align_act.set_target_state(target_state)

        max_time = rospy.get_time() + self.timeout

        while not rospy.is_shutdown():
            if rospy.get_time() > max_time:
                resp.res = False
                resp.response = "Timeout"
                resp.error_code = ErrorCode.TImeout
                rospy.logwarn(resp.response)
                break

            marker_pose_in_cam = self.current_marker_poses

            marker_in_base_link, marker_ang_in_base_link = self.transfer_frame(
                marker_pose_in_cam,
                source_frame=camera_frame,
                target_frame=base_link_frame,
            )

            measured_state = [
                marker_in_base_link[0],
                marker_in_base_link[1],
                marker_ang_in_base_link,
            ]

            # rospy.loginfo(f"target_state: {target_state}")
            # rospy.loginfo(f"measured_state: {measured_state}")
            # rospy.loginfo(f"vel: {self.arm_act.get_last_vel()}")

            rospy.loginfo(f"error: {np.array( measured_state)-np.array(target_state)}")
            self.align_act.set_measured_state(measured_state)
            if self.align_act.is_near_target_state_sometime(self.state_tolerance, 1.0):
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
        self.desired_cube_pos_arm_base[0] = config["desired_cube_x"]
        self.desired_cube_pos_arm_base[1] = config["desired_cube_y"]
        self.timeout = config["timeout"]
        self.Kp = config["Kp"]
        self.Ki = config["Ki"]
        self.Kd = config["Kd"]
        self.desired_tag_pos_arm_base[0] = config["desired_tag_x"]
        self.desired_tag_pos_arm_base[1] = config["desired_tag_y"]
        self.max_velocity = config["max_velocity"]
        self.min_velocity = config["min_velocity"]
        self.max_angular_velocity = config["max_angular_velocity"]
        self.min_angular_velocity = config["min_angular_velocity"]
        self.seperate_I_threshold = config["seperate_I_threshold"]

        self.arm_act.apply_velocity_limit(
            self.max_velocity,
            self.max_angular_velocity,
            self.min_velocity,
            self.min_angular_velocity,
        )
        self.align_act.set_pid_param(
            self.Kp, self.Ki, self.Kd, self.seperate_I_threshold
        )
        self.align_act.set_sample_time(1.0 / self.ros_rate)

        return config

    def transfer_frame(
        self,
        pose_source: Pose,
        source_frame,
        target_frame,
    ):
        if not self.tfBuffer.can_transform(
            target_frame,
            source_frame,
            rospy.Time.now(),
        ):
            rospy.logerr(
                prefix
                + f"pick_and_place: cannot find transform between {target_frame} and {source_frame}"
            )
            return None, None
        posestamped_source = tf2_geometry_msgs.PoseStamped()
        posestamped_source.header.stamp = rospy.Time.now()
        posestamped_source.header.frame_id = source_frame
        posestamped_source.pose = pose_source
        posestamped_target = self.tfBuffer.transform(posestamped_source, target_frame)
        pose_target = posestamped_target.pose
        pos = [
            pose_target.position.x,
            pose_target.position.y,
            pose_target.position.z,
        ]

        quat = np.array(
            [
                pose_source.orientation.x,
                pose_source.orientation.y,
                pose_source.orientation.z,
                pose_source.orientation.w,
            ]
        )
        angle = sciR.from_quat(quat).as_euler("YXZ")[0]
        # limit angle from -pi to pi
        angle = np.floor((angle + np.pi) / (2 * np.pi)) * 2 * np.pi - angle
        return pos, angle


if __name__ == "__main__":
    rospy.init_node("manipulater_node", anonymous=True)
    rter = manipulator()

    rospy.spin()
