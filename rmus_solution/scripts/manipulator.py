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
from rmus_solution.cfg import manipulator_PIDConfig
from enum import IntEnum
from arm_ctrl import arm_action, align_action, AlignMode


class AlignRequest(IntEnum):
    Reset = 0
    Grasp = 1
    Place = 2
    PlaceFake = 3


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
        """ actuator """
        self.arm_act = arm_action()
        self.align_act = align_action(self.arm_act)
        """ data """
        self.stamp = 0              ## time stamp when received id and pose of target block
        self.id_targ = 0            ## target block id
        self.pose_targ = Pose()     ## target block pose
        """ hyperparams """
        self.state_tolerance = [0.02, 0.02, 0.1]

        ############### Dynamic params ###############
        self.ros_rate = 10
        self.timeout = 10
        self.Kp = 0.5
        self.Ki = 0.0
        self.Kd = 0.0
        self.SP_pos_grasp = [0.5, 0.0]    ## desired pos of block in arm_base frame when grasping (SetPoint)
        self.SP_pos_place = [0.5, 0.0]    ## desired pos of tag in arm_base frame when aligning to place (SetPoint)
        self.max_velocity = 0.5
        self.min_velocity = 0.1
        self.max_angular_velocity = 0.5
        self.min_angular_velocity = 0.1
        ############### Dynamic params ###############

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        ## update markerinfo of target block
        self.__marker_pose_sub = rospy.Subscriber("/get_blockinfo", MarkerInfoList, self.marker_pose_callback, queue_size=1)
        ## switch `ig_targ` and reset/grasp/place target block
        self.__grasp_signal_service = rospy.Service("/let_manipulator_work", graspsignal, self.grasp_signal_callback)
        self.__grasp_config_service = rospy.Service("/manipulator_config", graspconfig, self.grasp_config_callback)
        self.__dynamic_reconfigure_server = Server(manipulator_PIDConfig, self.dynamic_reconfigure_callback)
        self.align_angle = False
        self.align_mode = AlignMode.OpenLoop
        self.align_act.set_align_config(self.align_angle, self.align_mode)

    def marker_pose_callback(self, msg: MarkerInfoList):
        """update self.current_marker_poses of self.desired_cube_id"""
        for markerInfo in msg.markerInfoList:
            markerInfo: MarkerInfo
            if markerInfo.id == self.id_targ:
                self.pose_targ = markerInfo.pose
                self.stamp = rospy.get_time()

    def grasp_signal_callback(self, req: graspsignalRequest):
        """ call `grasp()`, `place()` or `place_fake()` according to grasp signal """
        self.id_targ = req.marker_id
        # reset the arm
        if req.mode == AlignRequest.Reset:
            self.arm_act.reset_pos()
            self.arm_act.open_gripper()
            resp = graspsignalResponse()
            resp.res = True
            resp.response = "reset arm position and open gripper"
            resp.error_code = ErrorCode.Success
            rospy.loginfo(resp.response)
            return resp
        # if image_time is too old, rejects to grasp or place cube, instead just go back for sometime
        initial_time = rospy.get_time()
        while rospy.get_time() - self.stamp > 0.1:
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
            return self.grasp(rate)
        elif req.mode == AlignRequest.Place:
            return self.place(rate, req.layer)
        elif req.mode == AlignRequest.PlaceFake:
            return self.drop()
        else:
            rospy.logerr("Invalid mode")
            resp = graspsignalResponse()
            resp.res = False
            resp.error_code = ErrorCode.Fail
            resp.response = "Invalid mode"
            rospy.logwarn(resp.response)
            return resp

    def grasp_config_callback(self, req: graspconfigRequest):
        self.align_angle = req.align_angle
        self.align_mode = AlignMode(req.align_mode)
        resp = graspconfigResponse()
        self.align_act.set_align_config(self.align_angle, self.align_mode)
        resp.res = True
        return resp

    def grasp(self, rate):
        """ call arm_ctrl to grasp block, will take a few seconds to return (AKA blocking type) """
        resp = graspsignalResponse()

        if not 1 <= self.id_targ <= 6:
            resp.res = False
            resp.response = "Invalid marker id"
            resp.error_code = ErrorCode.InvalidMarkerID
            rospy.logwarn(f"Invalid marker id: {self.id_targ}")
            return resp

        self.arm_act.preparation_for_grasp()

        ## setpoint of block pose in frame `arm_base`
        tmp = Pose()
        tmp.position.x = self.SP_pos_grasp[0]
        tmp.position.y = self.SP_pos_grasp[1]
        tmp.orientation.w = 1.0
        ## convert from frame `arm_base` to frame `base_link`
        pos_sp, ang_sp = self.transfer_frame(
            tmp,
            source_frame=arm_base_frame,
            target_frame=base_link_frame,
        )
        x_sp = [pos_sp[0], pos_sp[1], ang_sp]
        self.align_act.set_target_state(x_sp)

        max_time = rospy.get_time() + self.timeout
        while not rospy.is_shutdown():
            if rospy.get_time() > max_time:
                resp.res = False
                resp.response = "Timeout"
                resp.error_code = ErrorCode.TImeout
                rospy.logwarn(resp.response)
                break

            marker_pose_in_cam = self.pose_targ

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
            err = np.array(measured_state)-np.array(x_sp)
            rospy.loginfo(f"error: {1000*err[0]:.1f}mm, {1000*err[1]:.1f}mm, {np.rad2deg(err[2]):.1f}degree")

            if self.arm_act.can_arm_grasp_sometime(
                marker_in_arm_base, 1.0, self.align_mode
            ) and self.align_act.is_near_target_state_sometime(
                self.state_tolerance, 1.0
            ):
                self.arm_act.go_and_grasp(marker_in_arm_base, self.align_mode)
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
            elif self.align_act.is_target_state_too_faraway():
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

    def place(self, rate, place_layer: int = 1):
        """ call arm_ctrl to place block, will take a few seconds to return (AKA blocking type) """
        resp = graspsignalResponse()

        if 1 <= self.id_targ <= 6:
            # align to the cube
            ...
        elif 7 <= self.id_targ <= 9:
            # align to the marker
            ...
        else:
            resp.res = False
            resp.response = "Invalid marker id"
            resp.error_code = ErrorCode.InvalidMarkerID
            rospy.logwarn(f"Invalid marker id: {self.id_targ}")
            return resp

        if 1 <= place_layer <= 3:
            self.arm_act.preparation_for_place(place_layer)
        else:
            rospy.logwarn(f"Invalid layer: {place_layer}")
            resp.res = False
            resp.response = "Invalid layer"
            resp.error_code = ErrorCode.InvalidLayer
            rospy.logwarn(resp.response)
            return resp

        tmp = Pose()
        tmp.position.x = self.SP_pos_place[0]
        tmp.position.y = self.SP_pos_place[1]
        tmp.orientation.w = 1.0

        pos_sp, ang_sp = self.transfer_frame(
            tmp,
            source_frame=arm_base_frame,
            target_frame=base_link_frame,
        )

        x_sp = [pos_sp[0], pos_sp[1], ang_sp]
        self.align_act.set_target_state(x_sp)

        max_time = rospy.get_time() + self.timeout

        while not rospy.is_shutdown():
            if rospy.get_time() > max_time:
                resp.res = False
                resp.response = "Timeout"
                resp.error_code = ErrorCode.TImeout
                rospy.logwarn(resp.response)
                break

            marker_pose_in_cam = self.pose_targ

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

            err = np.array(measured_state)-np.array(x_sp)
            rospy.loginfo(f"error: {1000*err[0]:.1f}mm, {1000*err[1]:.1f}mm, {np.rad2deg(err[2]):.1f}degree")
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

    def drop(self):
        """ drop block """
        rospy.loginfo("<manipulator>: place fake cube")
        self.arm_act.set_arm(0.21, -0.08)
        rospy.sleep(2.0)
        self.arm_act.open_gripper()
        rospy.sleep(2.0)
        self.arm_act.reset_pos()
        resp = graspsignalResponse()
        resp.res = True
        resp.response = "Place fake cube"
        resp.error_code = ErrorCode.Success
        rospy.loginfo(resp.response)
        return resp

    def dynamic_reconfigure_callback(self, config: dict, level: int):
        self.ros_rate = config["control_frequency"]
        self.SP_pos_grasp[0] = config["desired_cube_x"]
        self.SP_pos_grasp[1] = config["desired_cube_y"]
        self.timeout = config["timeout"]
        self.Kp = config["Kp"]
        self.Ki = config["Ki"]
        self.Kd = config["Kd"]
        self.SP_pos_place[0] = config["desired_tag_x"]
        self.SP_pos_place[1] = config["desired_tag_y"]
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
                f"pick_and_place: cannot find transform between {target_frame} and {source_frame}"
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
    rospy.init_node("manipulator_node", anonymous=True)
    rter = manipulator()
    rospy.spin()
