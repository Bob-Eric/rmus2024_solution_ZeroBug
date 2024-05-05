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
    Drop = 3


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
    Timeout = 6
    # Gripper is failed to grasp the cube, maybe the esitmated position is wrong
    Fail = 7


frame_glb = "map"
frame_arm = "arm_base"
frame_cam = "camera_aligned_depth_to_color_frame_correct"
frame_chassis = "base_link"


class manipulator:
    def __init__(self) -> None:
        """object reference"""
        self.arm_act: arm_action = arm_action()
        self.align_act: align_action = align_action(self.arm_act)
        """ state params """
        self.stamp = 0  ## time stamp when received id and pose of target block
        self.id_targ = 0  ## target block id
        self.pose_targ = Pose()  ## target block pose
        self.gpose_targ = Pose()  ## target block gpose
        """ config params """
        self.state_tolerance = [0.015, 0.015, 0.1]

        ############### Dynamic params ###############
        self.ros_rate = 30
        self.timeout = 10
        self.Kp = 4
        self.Ki = 2
        self.Kd = 0
        self.pos_sp_grasp = [
            0.2,
            0.0,
        ]  ## desired pos of block in arm_base frame when grasping (SetPoint)
        self.pos_sp_place = [
            0.18,
            0.0,
        ]  ## desired pos of tag in arm_base frame when aligning to place (SetPoint)
        self.max_velocity = 0.3
        self.min_velocity = 0.05
        self.max_angular_velocity = 0.3
        self.min_angular_velocity = 0.01
        ############### Dynamic params ###############

        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)

        ## update markerinfo of target block
        self.__marker_pose_sub = rospy.Subscriber(
            "/get_blockinfo", MarkerInfoList, self.marker_pose_callback, queue_size=1
        )
        ## switch `ig_targ` and reset/grasp/place target block
        self.__grasp_signal_service = rospy.Service(
            "/manipulator/grasp", graspsignal, self.grasp_signal_callback
        )
        self.__grasp_config_service = rospy.Service(
            "/manipulator/grasp_config", graspconfig, self.grasp_config_callback
        )
        self.__dynamic_reconfigure_server = Server(
            manipulator_PIDConfig, self.dynamic_reconfigure_callback
        )
        self.align_angle = True
        self.align_mode = AlignMode.PID
        self.align_act.set_align_config(self.align_angle, self.align_mode)

    def marker_pose_callback(self, msg: MarkerInfoList):
        """update self.current_marker_poses of self.desired_cube_id"""
        for markerInfo in msg.markerInfoList:
            markerInfo: MarkerInfo
            if markerInfo.id == self.id_targ:
                self.pose_targ = markerInfo.pose
                self.gpose_targ = markerInfo.gpose
                self.stamp = rospy.get_time()

    def grasp_signal_callback(self, req: graspsignalRequest):
        """call `grasp()`, `place()` or `drop()` according to grasp signal. Note: arm pos will be RESET when action done"""
        self.id_targ = req.marker_id
        # reset the arm
        if req.mode == AlignRequest.Reset:
            self.arm_act.reset_pos()
            self.arm_act.open_gripper()
            return self.grp_sig_resp(
                True, "reset arm position and open gripper", ErrorCode.Success
            )
        # if image_time is too old, rejects to grasp or place cube, instead just go back for sometime
        initial_time = rospy.get_time()
        while rospy.get_time() - self.stamp > 0.1:
            rospy.loginfo("latency detected!")
            # Go back for 0.5s
            if rospy.get_time() - initial_time > 1.0:
                self.align_act.send_cmd_vel([-0.2, 0.0, 0.0])
                rospy.sleep(0.3)
                self.align_act.send_cmd_vel([0.0, 0.0, 0.0])
            if rospy.get_time() - initial_time > 2.0:
                return self.grp_sig_resp(
                    False, "Can't get the marker info in time", ErrorCode.Latency
                )
            rospy.sleep(0.2)
        rate = rospy.Rate(self.ros_rate)
        if req.mode == AlignRequest.Grasp:
            return self.grasp(rate)
        elif req.mode == AlignRequest.Place:
            return self.place(rate, req.layer)
        elif req.mode == AlignRequest.Drop:
            return self.drop()
        else:
            rospy.logerr("Invalid mode")
            return self.grp_sig_resp(False, "Invalid mode", ErrorCode.Fail)

    def grasp_config_callback(self, req: graspconfigRequest):
        self.align_angle = req.align_angle
        self.align_mode = AlignMode(req.align_mode)
        resp = graspconfigResponse()
        self.align_act.set_align_config(self.align_angle, self.align_mode)
        resp.res = "Set align mode to " + str(self.align_mode)
        return resp

    @property
    def x_sp_grasp(self):
        """get setpoint of grasp (in frame_chassis)"""
        ## pos_sp_grasp is in frame_arm
        tmp = Pose()
        tmp.position.x = self.pos_sp_grasp[0]
        tmp.position.y = self.pos_sp_grasp[1]
        tmp.orientation.w = 1.0
        ## convert from frame_arm to frame_chassis
        pos_sp, ang_sp = self.transfer_frame(
            tmp,
            frame_src=frame_arm,
            frame_dst=frame_chassis,
        )
        return np.array([pos_sp[0], pos_sp[1], ang_sp])

    @property
    def x_sp_place(self):
        """get setpoint of place (in frame_chassis)"""
        ## pos_sp_grasp is in frame_arm
        tmp = Pose()
        tmp.position.x = self.pos_sp_place[0]
        tmp.position.y = self.pos_sp_place[1]
        tmp.orientation.w = 1.0
        ## convert from frame_arm to frame_chassis
        pos_sp, ang_sp = self.transfer_frame(
            tmp,
            frame_src=frame_arm,
            frame_dst=frame_chassis,
        )
        return np.array([pos_sp[0], pos_sp[1], ang_sp])

    def anti_wall(self, wdir, ang_chassis):
        bdir = self.rot_mat(frame_src=frame_cam, frame_dst=frame_glb, pose_src=self.pose_targ) @ np.array([0, 0, 1])
        up = np.array([0, 0, 1])
        ang = self.sgn_angle(bdir, wdir, up)
        ## use 75 deg instead of 90 deg to avoid oscillation around 90 deg
        if abs(ang) > 5/12*np.pi:
            sgn = np.sign(ang)
            print(f"<anti-wall> ang_chassis before: {np.rad2deg(ang_chassis):>6.1f} deg; ", end="")
            ang_chassis += sgn * np.pi/2
            print(f"after: {np.rad2deg(ang_chassis):>6.1f} deg")
        return ang_chassis

    @property
    def x_mv(self):
        """get measured variable of target block (in frame_chassis)"""
        pos_chassis, ang_chassis = self.transfer_frame(
            self.pose_targ, frame_src=frame_cam, frame_dst=frame_chassis
        )
        gp = self.gpose_targ.position
        if gp.x > 2.45 or gp.y > 3.45:
            print(f"({gp.x:.3f}, {gp.y:.3f})")
            wdir = np.array([1, 0, 0]) if gp.x > 2.45 else np.array([0, 1, 0])
            ang_chassis = self.anti_wall(wdir, ang_chassis)
        elif np.abs(ang_chassis) > np.pi / 4:
            print(f"<nearest-dir> ang_chassis before: {np.rad2deg(ang_chassis):>6.1f} deg; ", end="")
            ang_chassis -= np.sign(ang_chassis) * np.pi / 2
            print(f"after: {np.rad2deg(ang_chassis):>6.1f} deg")
        return np.array([pos_chassis[0], pos_chassis[1], ang_chassis])

    def grasp(self, rate):
        """call arm_ctrl to grasp block, will take a few seconds to return (AKA blocking type)"""
        if not 1 <= self.id_targ <= 6:
            print(f"Invalid marker id: {self.id_targ}")
            return self.grp_sig_resp(
                False, "Invalid marker id", ErrorCode.InvalidMarkerID
            )
        ## takes ~3 seconds to brake and open gripper
        self.arm_act.preparation_for_grasp(self.align_act)
        """ align first """
        print("+++ aligning")
        x_sp = self.x_sp_grasp
        self.align_act.init_ctrl()
        self.align_act.set_state_sp(x_sp)
        t_end = rospy.get_time() + self.timeout
        while not rospy.is_shutdown():
            if rospy.get_time() > t_end:
                self.align_act.send_cmd_vel([-0.3, 0.0, 0.0])
                rospy.sleep(0.7)
                self.align_act.stop()
                return self.grp_sig_resp(False, "Timeout", ErrorCode.Timeout)
            x_mv = self.x_mv
            ## calc sp in frame_arm for arm_act
            self.align_act.set_state_mv(x_mv)
            self.align_act.align()
            if self.align_act.finished(self.state_tolerance, 1.0):
                print("==> align finished")
                self.align_act.stop()
                break
            rate.sleep()
        """ grasp when aligned """
        pos_arm, _ = self.transfer_frame(
            self.pose_targ, frame_src=frame_cam, frame_dst=frame_arm
        )
        self.arm_act.grasp(self.align_act, pos_arm)
        """ response with success """
        return self.grp_sig_resp(True, "Grasp block", ErrorCode.Success)

    def place(self, rate, place_layer: int = 1):
        """call arm_ctrl to place block, will take a few seconds to return (AKA blocking type)"""
        if place_layer < 1 or place_layer > 3:
            print(f"Invalid layer: {place_layer}")
            return self.grp_sig_resp(False, "Invalid layer", ErrorCode.InvalidLayer)
        if not 1 <= self.id_targ <= 9:
            print(f"Invalid marker id: {self.id_targ}")
            return self.grp_sig_resp(
                False, "Invalid marker id", ErrorCode.InvalidMarkerID
            )
        ## takes half second to brake
        print("+++ aligning")
        self.align_act.send_cmd_vel([0.0, 0.0, 0.0])
        rospy.sleep(0.5)
        """ align first """
        x_sp = self.x_sp_place
        self.align_act.init_ctrl()
        self.align_act.set_state_sp(x_sp)
        t_end = rospy.get_time() + self.timeout
        while not rospy.is_shutdown():
            if rospy.get_time() > t_end:
                self.align_act.stop()
                return self.grp_sig_resp(False, "Timeout", ErrorCode.Timeout)
            x_mv = self.x_mv
            ## calc sp in frame_arm for arm_act
            self.align_act.set_state_mv(x_mv)
            self.align_act.align()
            if self.align_act.finished(self.state_tolerance, 1.0):
                print("==> align finished")
                self.align_act.stop()
                break
            rate.sleep()
        """ place when aligned """
        self.arm_act.place(self.align_act, place_layer)
        """ response with success """
        return self.grp_sig_resp(True, "Place block", ErrorCode.Success)

    def drop(self):
        """drop block"""
        print("<manipulator>: drop cube")
        self.arm_act.set_arm(0.21, -0.08)
        rospy.sleep(2.0)
        self.arm_act.open_gripper()
        rospy.sleep(2.0)
        self.arm_act.reset_pos()
        return self.grp_sig_resp(True, "Open gripper to drop block", ErrorCode.Success)

    def grp_sig_resp(self, res=False, response="", error_code=ErrorCode.Fail):
        resp = graspsignalResponse()
        resp.res = res
        resp.response = response
        resp.error_code = error_code
        rospy.loginfo(response)
        return resp

    def dynamic_reconfigure_callback(self, config: dict, level: int):
        self.ros_rate = config["control_frequency"]
        self.pos_sp_grasp[0] = config["desired_cube_x"]
        self.pos_sp_grasp[1] = config["desired_cube_y"]
        self.timeout = config["timeout"]
        self.Kp = config["Kp"]
        self.Ki = config["Ki"]
        self.Kd = config["Kd"]
        self.pos_sp_place[0] = config["desired_tag_x"]
        self.pos_sp_place[1] = config["desired_tag_y"]
        self.max_velocity = config["max_velocity"]
        self.min_velocity = config["min_velocity"]
        self.max_angular_velocity = config["max_angular_velocity"]
        self.min_angular_velocity = config["min_angular_velocity"]
        self.seperate_I_threshold = config["seperate_I_threshold"]

        self.align_act.apply_velocity_limit(
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

    def transfer_frame(self, pose_src: Pose, frame_src, frame_dst):
        while not self.tfBuffer.can_transform(frame_dst, frame_src, rospy.Time.now()):
            rospy.logerr(f"pick_and_place: cannot find transform between {frame_dst} and {frame_src}")
            rospy.sleep(0.1)
        posestamped_src = tf2_geometry_msgs.PoseStamped()
        posestamped_src.header.stamp = rospy.Time.now()
        posestamped_src.header.frame_id = frame_src
        posestamped_src.pose = pose_src
        posestamped_dst = self.tfBuffer.transform(posestamped_src, frame_dst)
        pose_dst = posestamped_dst.pose
        pos = [pose_dst.position.x, pose_dst.position.y, pose_dst.position.z]
        quat = np.array(
            [
                pose_src.orientation.x,
                pose_src.orientation.y,
                pose_src.orientation.z,
                pose_src.orientation.w,
            ]
        )
        angle = sciR.from_quat(quat).as_euler("YXZ")[0]
        # limit angle to [-pi, pi]
        angle = np.floor((angle + np.pi) / (2 * np.pi)) * 2 * np.pi - angle
        return pos, angle

    def rot_mat(self, frame_src, frame_dst, pose_src=None):
        while not self.tfBuffer.can_transform(frame_dst, frame_src, rospy.Time()):
            rospy.logerr(f"pick_and_place: cannot find transform between {frame_dst} and {frame_src}")
            rospy.sleep(0.1)
        posestamp_src = tf2_geometry_msgs.PoseStamped()
        posestamp_src.header.stamp = rospy.Time()
        posestamp_src.header.frame_id = frame_src
        if pose_src is None:
            posestamp_src.pose.orientation.w = 1
        else:
            posestamp_src.pose = pose_src
        rot_dst = self.tfBuffer.transform(posestamp_src, frame_dst).pose.orientation
        return sciR.from_quat([rot_dst.x, rot_dst.y, rot_dst.z, rot_dst.w]).as_matrix()

    def sgn_angle(self, vec_from, vec_to, axis):
        """calculate the sign of the angle between two vectors, return: signed angle, [-pi, pi]"""
        ang = np.arccos(np.dot(vec_from, vec_to) / (np.linalg.norm(vec_from) * np.linalg.norm(vec_to)))
        sgn = np.sign(np.cross(vec_from, vec_to).dot(axis))
        return ang * sgn

if __name__ == "__main__":
    rospy.init_node("manipulator_node", anonymous=True)
    rter = manipulator()
    rospy.spin()
