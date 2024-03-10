#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as sciR
import rospy
from geometry_msgs.msg import Pose, TransformStamped, Vector3
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

prefix = "[manipulator]"

class manipulator:

    def __init__(self) -> None:
        self.arm_act = arm_action()
        self.align_act = align_action(self.arm_act)

        self.current_marker_poses = Pose()
        self.image_time_now = 0.0

        self.desired_cube_id = 0

        self.cube_goal_tolerance = 0.01
        self.tag_goal_tolerance = 0.01

        self.check_stuck_time_old = 0
        self.base_link_pos_old = np.array([-1.0, -1.0])

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
            if markerInfo.id == self.desired_cube_id:
                self.current_marker_poses = markerInfo.pose
                self.image_time_now = rospy.get_time()

    def grasp_signal_callback(self, req: graspsignalRequest):
        self.desired_cube_id = req.marker_id
        # Reset the arm
        if req.mode == AlignRequest.Reset:
            self.arm_act.reset_pos()
            self.arm_act.open_gripper()
            resp = graspsignalResponse()
            resp.res = True
            resp.response = "reset arm position and open gripper"
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
                resp.res = True
                resp.response = "Successfully Grasp fake"
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
            rospy.logerr(prefix + "Invalid mode")
            resp = graspsignalResponse()
            resp.res = False
            return resp

    def grasp_cube_resp(self, rate):
        self.arm_act.preparation_for_grasp()
        resp = graspsignalResponse()

        self.align_act.set_setpoint(self.desired_cube_pos_base_link)

        while not rospy.is_shutdown():
            target_marker_pose = self.current_marker_poses

            measured_pos, _ = self.trans_cam_frame_to_target_frame(target_marker_pose)
            self.align_act.set_measured_point(measured_pos)

            if self.align_act.is_near_setpoint(self.cube_goal_tolerance):
                self.arm_act.go_and_grasp()

                resp.res = True
                resp.response = "Successfully Grasp"
                break
            elif not self.movement_process(resp):
                break

            rate.sleep()
        return resp

    def place_cube_resp(self, rate, place_layer: int = 1):
        resp = graspsignalResponse()

        if 1 <= place_layer <= 3:
            self.arm_act.preparation_for_place(place_layer)
        else:
            rospy.logerr(prefix + "place_layer should be 1, 2 or 3")
            resp.res = False
            return resp

        self.align_act.set_setpoint(self.desired_tag_pos_base_link)

        while not rospy.is_shutdown():
            target_marker_pose = self.current_marker_poses

            measured_pos, _ = self.trans_cam_frame_to_target_frame(target_marker_pose)
            self.align_act.set_measured_point(measured_pos)
            if self.align_act.is_near_setpoint(self.tag_goal_tolerance):
                self.arm_act.go_and_place()

                resp.res = True
                resp.response = "Successfully Place"
                break
            elif not self.movement_process(resp):
                break
            rate.sleep()
        return resp

    # return False if stuck
    def movement_process(self, resp):
        self.align_act.align()
        output = self.arm_act.get_last_vel()
        output_x = output[0]
        output_y = output[1]

        if (
            rospy.get_time() - self.arm_act.movement_saturat_time > 2.5
            and self.arm_act.is_vel_saturating()
        ):
            if self.check_stuck():
                resp.res = False
                error_msg = "Robot is stuck, goint backward!"
                resp.response = error_msg
                rospy.logwarn(prefix + error_msg)
                reverse_vel_x = -np.sign(output_x) * 0.3
                reverse_vel_y = -np.sign(output_y) * 0.3
                self.arm_act.send_cmd_vel([reverse_vel_x, reverse_vel_y, 0.0])
                rospy.sleep(0.5)
                self.arm_act.send_cmd_vel([0.0, 0.0, 0.0])
                rospy.sleep(0.5)
                return False

        return True

    def check_stuck(self):
        base_link_tf_stamped: TransformStamped = self.tfBuffer.lookup_transform(
            "base_link", "map", rospy.Time(0), rospy.Duration(1.0)
        )
        base_link_pos_vec: Vector3 = base_link_tf_stamped.transform.translation
        base_link_pos = np.array([base_link_pos_vec.x, base_link_pos_vec.y])

        if np.linalg.norm(base_link_pos - self.base_link_pos_old) < 0.05:
            rospy.logwarn(prefix + "robot is stuck")
            return True

        self.base_link_pos_old = base_link_pos
        return False

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
            rospy.logerr(prefix + 
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
