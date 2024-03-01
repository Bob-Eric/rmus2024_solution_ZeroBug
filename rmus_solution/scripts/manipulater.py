#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.spatial.transform import Rotation as sciR
import rospy
from geometry_msgs.msg import Twist, Pose, Point, TransformStamped, Vector3
from rmus_solution.srv import graspsignal, graspsignalRequest, graspsignalResponse
from rmus_solution.msg import MarkerInfo, MarkerInfoList
import tf2_ros
import tf2_geometry_msgs
from simple_pid import PID
from dynamic_reconfigure.server import Server
from rmus_solution.cfg import manipulater_PIDConfig
from enum import IntEnum
from rmus_solution.msg import MarkerInfo


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


class manipulater:
    def __init__(self) -> None:
        self.arm_gripper_pub = rospy.Publisher("arm_gripper", Point, queue_size=2)
        self.arm_position_pub = rospy.Publisher("arm_position", Pose, queue_size=2)
        self.cmd_vel_puber = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(
            "/get_blockinfo", MarkerInfoList, self.marker_pose_cb, queue_size=1
        )

        rospy.Subscriber("/gripper_state", Point, self.gripper_state_cb, queue_size=1)

        self.current_marker_poses = Pose()
        self.gripper_state = -1
        self.gripper_state_old = -1
        self.marker_time_now = 0.0

        self.align_marker_id = 0

        self.tag_goal_tolerance = [0.005, 0.005]

        # TODO: update the dynamic params config
        ############### Dynamic params ###############
        self.ros_rate = 10
        self.desired_cube_pos_in_cam = [0.5, 0.0]
        self.desired_tag_pos_in_cam = [0.5, 0.0]
        self.pid_P = 0.0
        self.pid_I = 0.0
        self.pid_D = 0.0
        self.xy_seperate_I_threshold = [0.1, 0.1]
        ############### Dynamic params ###############

        pid_cal_time = 1 / self.ros_rate

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

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.base_link_pos_old = np.array([-1.0, -1.0])
        self.check_stuck_time_old = 0.0

        self.service = rospy.Service(
            "/let_manipulater_work", graspsignal, self.AlignerworkCallback
        )
        self.server = Server(manipulater_PIDConfig, self.dynamic_reconfigure_callback)

    def dynamic_reconfigure_callback(self, config: dict, level: int):
        self.ros_rate = config["control_frequency"]
        self.desired_cube_pos_in_cam[0] = config["desired_cube_x"]
        self.desired_cube_pos_in_cam[1] = config["desired_cube_y"]
        self.pid_P = config["Kp"]
        self.pid_I = config["Ki"]
        self.pid_D = config["Kd"]
        self.desired_tag_pos_in_cam[0] = config["desired_tag_x"]
        self.desired_tag_pos_in_cam[1] = config["desired_tag_y"]
        self.xy_seperate_I_threshold = [
            config["x_seperate_I_threshold"],
            config["y_seperate_I_threshold"],
        ]

        self.update_pid_params([0, 0])
        return config

    def update_pid_params(self, setpoint: list):

        self.pos_x_pid.sample_time = 1 / self.ros_rate
        self.pos_y_pid.sample_time = 1 / self.ros_rate
        self.pos_x_pid.setpoint = setpoint[0]
        self.pos_y_pid.setpoint = setpoint[1]
        self.pos_x_pid.Kp = self.pid_P
        self.pos_x_pid.Ki = self.pid_I
        self.pos_x_pid.Kd = self.pid_D
        self.pos_y_pid.Kp = self.pid_P
        self.pos_y_pid.Ki = self.pid_I
        self.pos_y_pid.Kd = self.pid_D

        # rospy.loginfo(f"update_pid_params: {self.pos_x_pid}")

        self.pos_x_pid.reset()
        self.pos_y_pid.reset()

    def marker_pose_cb(self, msg: MarkerInfoList):
        """update self.current_marker_poses of self.desired_marker_id"""
        for markerInfo in msg.markerInfoList:
            markerInfo: MarkerInfo
            if markerInfo.id == self.align_marker_id:
                self.current_marker_poses = markerInfo.pose
                self.marker_time_now = rospy.get_time()

        # ############### Test ###############
        # self.check_stuck()
        # ############### Test ###############

    def gripper_state_cb(self, msg: Point):
        self.gripper_state = msg.x

        if self.gripper_state != self.gripper_state_old:
            self.gripper_state_old = self.gripper_state

            if self.gripper_state == 1.0:
                rospy.loginfo("gripper is closed")
            elif self.gripper_state == 0.0:
                rospy.loginfo("gripper is open")

    def get_target_pos_ang_in_frame(
        self, pose_in_cam: Pose, frame_id: str = "base_link"
    ):
        if not self.tfBuffer.can_transform(
            frame_id, "camera_aligned_depth_to_color_frame_correct", rospy.Time.now()
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
        posestamped_in_base = self.tfBuffer.transform(posestamped_in_cam, frame_id)
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

    def AlignerworkCallback(self, req: graspsignalRequest):
        self.align_marker_id = req.marker_id
        # Reset the arm
        if req.mode == AlignRequest.Reset:
            resp = self.reset_resp()
            return resp

        # If marker_time is too old, rejects to grasp or place cube, instead just go back for sometime
        initial_time = rospy.get_time()
        while rospy.get_time() - self.marker_time_now > 0.1:
            rospy.loginfo("latency detected!")

            # Go back for 0.5s
            if rospy.get_time() - initial_time > 1.0:
                self.send_vel([-0.2, 0.0, 0.0])
                rospy.sleep(0.1)
                self.send_vel([0.0, 0.0, 0.0])
            if rospy.get_time() - initial_time > 2.0:
                resp = graspsignalResponse()
                resp.res = False
                error_msg = "Can't get the marker info in time"
                resp.response = error_msg
                rospy.logwarn(error_msg)
                resp.error_code = ErrorCode.Latency
                return resp
            rospy.sleep(0.1)

        rate = rospy.Rate(self.ros_rate)

        if req.mode == AlignRequest.Grasp:
            resp = self.grasp_resp(rate, self.align_marker_id)
            return resp

        else:
            resp = self.place_resp(
                rate,
                req.layer,
                self.align_marker_id,
            )
            return resp

    def reset_resp(self):
        self.arm_idle_pos()
        rospy.sleep(0.2)
        self.gripper_open()
        resp = graspsignalResponse()
        resp.res = True
        resp.response = "reset arm position and open gripper"
        resp.error_code = ErrorCode.Success
        return resp

    def grasp_resp(self, rate, desired_marker_id):
        resp = graspsignalResponse()
        if not 1 <= desired_marker_id <= 6:
            rospy.logwarn(f"Invalid marker_id: {desired_marker_id}")
            resp.res = False
            error_msg = "Invalid marker_id"
            resp.response = error_msg
            rospy.logwarn(error_msg)
            resp.error_code = ErrorCode.InvalidMarkerID
            return resp
        rospy.loginfo("First align then grasp")
        rospy.loginfo("align to the right place")
        self.send_vel([0.0, 0.0, 0.0])
        self.gripper_open()

        self.update_pid_params(self.desired_cube_pos_in_cam)
        cube_faraway_cnt = 0
        output_x = 0.0
        output_y = 0.0
        while not rospy.is_shutdown():
            if rospy.get_time() - self.marker_time_now > 0.1:
                rospy.logwarn("latency detected!")
                continue
            target_marker_pose = self.current_marker_poses
            if target_marker_pose is None:
                continue

            cube_in_base_link_pos, target_angle = self.get_target_pos_ang_in_frame(
                target_marker_pose, "base_link"
            )
            # pos_in_grasp = [pos_in_base_link_pos[0] - 0.145, pos_in_base_link_pos[1], pos_in_base_link_pos[2] - 0.055]
            desired_grasp_pos = [
                cube_in_base_link_pos[0] - 0.145,
                cube_in_base_link_pos[1],
                cube_in_base_link_pos[2] - 0.055,
            ]

            if (
                self.check_grasp_possibility(desired_grasp_pos)
                and abs(output_x) < 0.1
                and abs(output_y) < 0.1
            ):
                print(
                    f"Near desired pos, cube_in_base_link_pos: {cube_in_base_link_pos}, desired_grasp_pos: {desired_grasp_pos}"
                )
                self.send_vel([0.0, 0.0, 0.0])
                rospy.sleep(0.5)

                self.grasp_process(desired_grasp_pos)
                rospy.sleep(1.0)
                if self.gripper_state == 1.0:
                    rospy.loginfo("Successfully Grasp")
                    resp.res = True
                    resp.response = "Successfully Grasp"
                    resp.error_code = ErrorCode.Success
                else:
                    rospy.logwarn("Fail to grasp")
                    resp.res = False
                    error_msg = "Fail to grasp"
                    resp.response = error_msg
                    rospy.logwarn(error_msg)
                    resp.error_code = ErrorCode.Fail

                break
            elif self.target_pos_not_too_faraway(desired_grasp_pos):
                if not self.movement_process(
                    resp,
                    cube_in_base_link_pos,
                    self.desired_cube_pos_in_cam,
                    False,
                    output_x,
                    output_y,
                ):
                    break
            else:
                cube_faraway_cnt += 1
                if cube_faraway_cnt > 30:
                    rospy.logwarn(
                        "Cube is too far away, cannot grasp just by pid control"
                    )
                    resp.res = False
                    error_msg = "Cube is faraway, cannot grasp just by pid control"
                    resp.response = "Cube is faraway, cannot grasp just by pid control"
                    rospy.logwarn(error_msg)
                    resp.error_code = ErrorCode.TargetTooFaraway
                    self.send_vel([0.0, 0.0, 0.0])
                    rospy.sleep(0.5)
                    break

            rate.sleep()
        return resp

    def target_pos_not_too_faraway(self, desired_grasp_pos):
        return desired_grasp_pos[0] >= -0.5 and abs(desired_grasp_pos[1]) <= 2.0

    def check_grasp_possibility(self, desired_grasp_pos: list):
        if desired_grasp_pos is None:
            return False
        elif abs(desired_grasp_pos[1]) > 0.025:
            return False
        elif 0.18 <= desired_grasp_pos[0] <= 0.22 and desired_grasp_pos[2] >= -0.10:
            return True
        elif 0.09 <= desired_grasp_pos[0] < 0.18 and desired_grasp_pos[2] > 0.08:
            return True
        else:
            return False

    def grasp_process(self, desired_grasp_pos: list):
        self.send_vel([0.0, 0.0, 0.0])
        self.arm_grasp_pos(desired_grasp_pos)
        rospy.sleep(1.0)
        rospy.loginfo("Place: reach the goal for placing.")

        self.gripper_close()
        rospy.sleep(0.5)

        self.gripper_close()
        rospy.sleep(0.5)
        self.arm_idle_pos()

        self.send_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.5)
        self.send_vel([0.0, 0.0, 0.0])

    def place_resp(self, rate, place_layer: int, align_marker_id: int):
        resp = graspsignalResponse()
        set_point = [0, 0]
        if 1 <= align_marker_id <= 6:
            # TODO: Refine the cube align place
            rospy.logwarn(f"Aligning to cube now!")
            set_point = [
                self.desired_cube_pos_in_cam[0] + 0.015,
                self.desired_cube_pos_in_cam[1],
            ]
        elif 7 <= align_marker_id <= 9:
            set_point = self.desired_tag_pos_in_cam
        else:
            rospy.logwarn(f"Invalid marker_id: {align_marker_id}")
            resp.res = False
            error_msg = "Invalid marker_id"
            resp.response = error_msg
            rospy.logwarn(error_msg)
            resp.error_code = ErrorCode.InvalidMarkerID
            return resp

        self.update_pid_params(set_point)
        self.send_vel([0.0, 0.0, 0.0])
        rospy.loginfo("First align then place")
        # TODO: Test whether need to wait for the arm to be stable
        if 1 <= place_layer <= 4:
            self.arm_place_pos(place_layer)
        else:
            error_msg = "Invalid layer"
            rospy.logwarn(error_msg)
            resp.res = False
            resp.response = error_msg
            resp.error_code = ErrorCode.InvalidLayer
            return resp
        rospy.sleep(1.0)

        align_y_finisned = False
        first_align_xy = True
        while not rospy.is_shutdown():
            if rospy.get_time() - self.marker_time_now > 0.1:
                rospy.logwarn("latency detected!")
                continue

            target_marker_pose = self.current_marker_poses
            if target_marker_pose is None:
                continue

            target_pos, target_angle = self.get_target_pos_ang_in_frame(
                target_marker_pose
            )
            if self.is_near_desired_position(
                target_pos,
                [self.pos_x_pid.setpoint, self.pos_y_pid.setpoint],
                self.tag_goal_tolerance,
            ):
                self.place_process(place_layer)

                resp.res = True
                resp.response = "Successfully Place"
                resp.error_code = ErrorCode.Success
                break

            elif self.target_pos_not_too_faraway(target_pos):
                can_move = True

                if not align_y_finisned:
                    # align y
                    output_y = 0.0
                    can_move = self.movement_process(
                        resp, target_pos, set_point, True, output_y=output_y
                    )
                    align_y_finisned = (
                        self.is_near_desired_position(
                            [0, target_pos[1]],
                            [0, self.pos_y_pid.setpoint],
                            self.tag_goal_tolerance,
                        )
                        and abs(output_y) < 0.02
                    )
                # First align y, then align x

                else:
                    # y is aligned, then align both of x and y
                    if first_align_xy:
                        # reset x pid
                        self.pos_x_pid.reset()
                        first_align_xy = False
                        self.send_vel([0.0, 0.0, 0.0])
                        rospy.sleep(0.5)
                    can_move = self.movement_process(resp, target_pos, set_point, False)

                if not can_move:
                    break
            else:
                rospy.logwarn(
                    "Marker is too far away, cannot go there just by pid control"
                )
                resp.res = False
                error_msg = "Marker is faraway, cannot go there just by pid control"
                resp.response = "Marker is faraway, cannot go there just by pid control"
                rospy.logwarn(error_msg)
                resp.error_code = ErrorCode.TargetTooFaraway
                self.send_vel([0.0, 0.0, 0.0])
                break

            rate.sleep()
        return resp

    # return False if stuck
    def movement_process(
        self,
        resp,
        measured_pos: list,
        set_pos: list,
        only_move_y=False,
        output_x=0.0,
        output_y=0.0,
    ):
        output_x, output_y = self.cal_pid_val(measured_pos, set_pos)

        output_x = 0.0 if only_move_y else output_x

        # only check stuck when the output is not zero
        output_x_near_zero = abs(output_x) <= 0.2
        output_y_near_zero = abs(output_y) <= 0.2
        output_near_zero = output_x_near_zero and output_y_near_zero

        if not output_near_zero and self.check_stuck():
            resp.res = False
            error_msg = "Robot is stuck, goint backward!"
            resp.response = error_msg
            rospy.logwarn(error_msg)
            resp.error_code = ErrorCode.Stuck
            reverse_vel_x = int(not output_x_near_zero) * -np.sign(output_x) * 0.3
            reverse_vel_y = int(not output_y_near_zero) * -np.sign(output_y) * 0.3
            self.send_vel([reverse_vel_x, reverse_vel_y, 0.0])
            rospy.sleep(0.5)
            self.send_vel([0.0, 0.0, 0.0])
            return False

        cmd_vel = [output_x, output_y, 0.0]
        self.send_vel(cmd_vel)

        return True

    def place_process(self, place_layer: int):
        rospy.loginfo("Place: reach the goal for placing.")
        self.send_vel([0.0, 0.0, 0.0])
        rospy.sleep(0.5)
        self.gripper_open()
        rospy.sleep(0.5)

        self.arm_idle_pos()
        self.send_vel([-0.3, 0.0, 0.0])
        rospy.sleep(0.5)
        self.send_vel([0.0, 0.0, 0.0])

    def is_near_desired_position(
        self, target_pos: list, desired_pos: list, tolerance: list
    ):
        x_diff_satisfy = abs(target_pos[0] - desired_pos[0]) <= tolerance[0]
        y_diff_satisfy = abs(target_pos[1] - desired_pos[1]) <= tolerance[1]
        return x_diff_satisfy and y_diff_satisfy

    def cal_pid_val(self, measured_pos, set_poos: list):

        if abs(measured_pos[0] - set_poos[0]) < self.xy_seperate_I_threshold[0]:
            self.pos_x_pid.Ki = self.pid_I
        else:
            self.pos_x_pid.Ki = 0.0

        if abs(measured_pos[1] - set_poos[1]) < self.xy_seperate_I_threshold[1]:
            self.pos_y_pid.Ki = self.pid_I
        else:
            self.pos_y_pid.Ki = 0.0

        output_x = -self.pos_x_pid(measured_pos[0])
        output_y = -self.pos_y_pid(measured_pos[1])
        # rospy.loginfo(f"output_x: {output_x}, output_y: {output_y}")
        return output_x, output_y

    def check_stuck(self):
        if rospy.get_time() - self.check_stuck_time_old > 1.0:
            self.check_stuck_time_old = rospy.get_time()
            base_link_tf_stamped: TransformStamped = self.tfBuffer.lookup_transform(
                "base_link", "map", rospy.Time(0), rospy.Duration(1.0)
            )
            base_link_pos_vec: Vector3 = base_link_tf_stamped.transform.translation
            base_link_pos = np.array([base_link_pos_vec.x, base_link_pos_vec.y])

            if np.linalg.norm(base_link_pos - self.base_link_pos_old) < 0.05:
                rospy.logwarn("robot is stuck")
                return True

            self.base_link_pos_old = base_link_pos
            return False
        else:
            return False

    def send_vel(self, vel: list):
        twist = Twist()
        twist.linear.z = 0.0
        twist.linear.x = vel[0]
        twist.linear.y = vel[1]
        twist.angular.z = vel[2]
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        self.cmd_vel_puber.publish(twist)

    def gripper_open(self):
        open_gripper_msg = Point()
        open_gripper_msg.x = 0.0
        open_gripper_msg.y = 0.0
        open_gripper_msg.z = 0.0
        rospy.loginfo("open the gripper")
        self.arm_gripper_pub.publish(open_gripper_msg)

    def gripper_close(self):
        close_gripper_msg = Point()
        close_gripper_msg.x = 1.0
        close_gripper_msg.y = 0.0
        close_gripper_msg.z = 0.0
        rospy.loginfo("close the gripper")
        self.arm_gripper_pub.publish(close_gripper_msg)

    def arm_idle_pos(self):
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

    def arm_grasp_pos(self, desired_grasp_pos: list):
        pose = Pose()
        # limit the grasp position 0.09 <= x <= 0.22, -0.08 <= y
        pose.position.x = max(min(desired_grasp_pos[0], 0.22), 0.09)
        pose.position.y = max(desired_grasp_pos[2], -0.08)
        rospy.loginfo(f"moving to grasp position: {pose.position.x}, {pose.position.y}")
        self.arm_position_pub.publish(pose)

    def arm_place_pos(self, layer=1):
        rospy.loginfo("now prepare to place (first layer)")
        pose = Pose()
        pose.position.x = 0.21
        pose.position.y = -0.04 + 0.055 * (layer - 1)
        # if layer == 1:
        #     pose.position.y = -0.04
        # elif layer == 2:
        #     pose.position.y = 0.03
        # elif layer == 3:
        #     pose.position.y = 0.08
        self.arm_position_pub.publish(pose)


if __name__ == "__main__":
    rospy.init_node("manipulater_node", anonymous=True)
    rter = manipulater()

    rospy.spin()
