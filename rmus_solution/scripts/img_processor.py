#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union
import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, Point, TransformStamped, Quaternion
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image, CameraInfo
from rmus_solution.srv import switch, switchResponse
from rmus_solution.msg import MarkerInfo, MarkerInfoList
from enum import IntEnum
import cv2
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from rtabmap_msgs.msg import RGBDImage
from detect import marker_detection


coord_cam = "camera_aligned_depth_to_color_frame_correct"
coord_glb = "map"


class ModeRequese(IntEnum):
    DoNothing = 0  ## disable marker_detection to save resources (160% cpu -> 60% cpu)
    BlockInfo = (
        1  ## actually not in use, blockinfo and gameinfo are detected simultaneously
    )

    GameInfo = 2

    End = 3


def pose_aruco_2_ros(rvec, tvec):
    tvec = tvec.flatten()
    aruco_pose_msg = Pose()
    aruco_pose_msg.position.x = tvec[0]
    aruco_pose_msg.position.y = tvec[1]
    aruco_pose_msg.position.z = tvec[2]
    rotation_matrix = cv2.Rodrigues(rvec)
    r_quat = R.from_matrix(rotation_matrix[0]).as_quat()
    aruco_pose_msg.orientation.x = r_quat[0]
    aruco_pose_msg.orientation.y = r_quat[1]
    aruco_pose_msg.orientation.z = r_quat[2]
    aruco_pose_msg.orientation.w = r_quat[3]
    return aruco_pose_msg


class Processor:
    def __init__(self, initial_mode=ModeRequese.DoNothing, verbose=True) -> None:
        self.current_mode = initial_mode
        self.collapsed = False
        self.verbose = verbose
        self.start_time = int(rospy.get_time() - 3)
        self.bridge = CvBridge()
        self.current_visualization_image = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_message("/camera/color/image_raw", Image, timeout=5.0)
                rospy.loginfo("Get topic /camera/color/image_raw.")
                break
            except:
                rospy.logwarn("Waiting for message /camera/color/image_raw.")
                continue

        while not rospy.is_shutdown():
            try:
                camerainfo = rospy.wait_for_message(
                    "/camera/color/camera_info", CameraInfo, timeout=5.0
                )
                rospy.loginfo("Get topic /camera/color/camera_info.")
                self.camera_matrix = np.array(camerainfo.K, "double").reshape((3, 3))
                rospy.loginfo("camera_matrix :\n {}".format(self.camera_matrix))
                break
            except:
                rospy.logwarn("Waiting for message /camera/color/camera_info.")
                continue

        try:
            if self.verbose:
                self.vis_thread = Thread(target=self.visualization)
                self.vis_thread.start()
        except:
            while True:
                self.collapsed = False
                rospy.logerr("The visualization window has collapsed!")
        self.br = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(
            "/camera/color/image_raw", Image, self.imageCallback, queue_size=1
        )
        rospy.Service("/img_processor/mode", switch, self.modeCallBack)
        self.pub_p = rospy.Publisher("/get_gameinfo", UInt8MultiArray, queue_size=1)
        self.pub_b = rospy.Publisher("/get_blockinfo", MarkerInfoList, queue_size=1)
        self.detected_gameinfo = None
        self.blocks_info = [None] * 9

    def imageCallback(self, image: Image):
        self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        # self.depth_img = self.bridge.imgmsg_to_cv2(image.depth, "32FC1")
        self.this_image_time_ms = int(image.header.stamp.nsecs / 1e6) + int(
            1000 * (image.header.stamp.secs - self.start_time)
        )
        locked_current_mode = self.current_mode

        id_list, tvec_list, rvec_list = [], [], []
        if locked_current_mode != ModeRequese.DoNothing:
            id_list, _, _, tvec_list, rvec_list = marker_detection(
                self.image, self.camera_matrix, self.verbose
            )
        ## update gameinfo (and publish)
        self.update_gameinfo(id_list, tvec_list)
        ## update blocks_info (and publish)
        self.update_blocks_info(id_list, tvec_list, rvec_list)
        self.current_visualization_image = self.image

        ## send tf for visualization
        for i in range(len(self.blocks_info)):
            if self.blocks_info[i] is None:
                continue
            self.send_block_tf(
                i + 1, pose=self.blocks_info[i][0], gpose=self.blocks_info[i][1]
            )
        return

    def depthCallback(self, image):
        self.depth_img = self.bridge.imgmsg_to_cv2(image, "32FC1")

    def modeCallBack(self, req):
        if ModeRequese.DoNothing <= ModeRequese(req.mode) < ModeRequese.End:
            self.current_mode = ModeRequese(req.mode)
            return switchResponse(self.current_mode)
        else:
            return switchResponse(req.mode)

    def send_block_tf(
        self, block_id, pose: Union[Pose, None] = None, gpose: Union[Pose, None] = None
    ):
        tf_pose = TransformStamped()
        tf_pose.header.stamp = rospy.Time.now()

        if gpose is not None:
            tf_pose.header.frame_id = coord_glb
            tf_pose.child_frame_id = f"gpose_block_{block_id}"
            tf_pose.transform.translation.x = gpose.position.x
            tf_pose.transform.translation.y = gpose.position.y
            tf_pose.transform.translation.z = gpose.position.z
            tf_pose.transform.rotation = gpose.orientation
            self.br.sendTransform(tf_pose)
        if pose is not None:
            tf_pose.header.frame_id = coord_cam
            tf_pose.child_frame_id = f"pose_block_{block_id}"
            tf_pose.transform.translation.x = pose.position.x
            tf_pose.transform.translation.y = pose.position.y
            tf_pose.transform.translation.z = pose.position.z
            tf_pose.transform.rotation = pose.orientation
            self.br.sendTransform(tf_pose)

    def update_gameinfo(self, id_list, tvec_list):
        gameinfo = [0, 0, 0]
        ## quads of gameinfo are high, so y component in camera frame is small. (x, y axis
        ##  of camera frame points right and downwards respectively), typical value is around -0.26.
        digit_list = [
            (id_list[i], t[0]) for i, t in enumerate(tvec_list) if t[1] < -0.15
        ]
        if len(digit_list) != 3:
            # print(f"detected {len(digit_list)} digits in gameinfo board, not 3")
            return
        digit_list.sort(key=lambda pair: pair[1])
        gameinfo = [id for (id, x) in digit_list]
        if self.detected_gameinfo is None:
            self.detected_gameinfo = gameinfo
        if tuple(self.detected_gameinfo) != tuple(gameinfo):
            rospy.logerr(
                f"detected different gameinfo: {gameinfo}, expected: {self.detected_gameinfo}"
            )
        ## publish gameinfo
        self.pub_p.publish(UInt8MultiArray(data=self.detected_gameinfo))
        print(f"gameinfo: {gameinfo}")
        return

    def P2A(self, point: Point) -> np.ndarray:
        return np.array([point.x, point.y, point.z])

    def A2P(self, array) -> Point:
        point = Point()
        point.x, point.y, point.z = array
        return point

    def Q2A(self, quat: Quaternion) -> np.ndarray:
        return np.array([quat.x, quat.y, quat.z, quat.w])

    def A2Q(self, array):
        quat = Quaternion()
        quat.x, quat.y, quat.z, quat.w = array
        return quat

    def sLPF(self, last_quat, new_quat, inertia=0.9):
        """low pass filter with slerp (e.g. quaternion)"""
        Rs = R.from_quat([new_quat, last_quat])
        Ts = [0, 1]
        slerp = Slerp(Ts, Rs)
        r_slerp = slerp([inertia])
        return r_slerp.as_quat()[0]

    def LPF(self, last_pos, new_pos, inertia=0.9) -> np.array:
        """low pass filter"""
        return inertia * last_pos + (1 - inertia) * new_pos

    def update_blocks_info(self, id_list, tvec_list, rvec_list):
        """update blocks_info (block 1-6 and B, O, X) and publish it"""
        ## filter out gameinfo quads (t[1] < -0.15)
        for i in reversed(range(len(id_list))):
            if tvec_list[i][1] < -0.15:
                id_list.pop(i)
                tvec_list.pop(i)
                rvec_list.pop(i)
        ## get transform of coord_cam and coord_glb
        pose_list = [pose_aruco_2_ros(r, t) for t, r in zip(tvec_list, rvec_list)]
        gpose_list = []
        try:
            trans = self.tfBuffer.lookup_transform(
                coord_glb, coord_cam, rospy.Time(), rospy.Duration(0.2)
            )
            inv_trans = self.tfBuffer.lookup_transform(
                coord_cam, coord_glb, rospy.Time(), rospy.Duration(0.2)
            )
        except Exception as e:
            print(f"Failed to get transform: {e}")
            return

        for pose in pose_list:
            ## define pose stamped in camera_link
            pose_stamp = tf2_geometry_msgs.PoseStamped()
            pose_stamp.header.stamp = rospy.Time.now()
            pose_stamp.header.frame_id = coord_cam
            pose_stamp.pose = pose
            ## transform from pose_stamp (camera_link) to gpose (map)
            gpose_stamp = tf2_geometry_msgs.do_transform_pose(pose_stamp, trans)
            gpose_list.append(gpose_stamp.pose)
        for i in range(len(self.blocks_info)):
            id = i + 1
            if id in id_list:
                ## block `id` is detected in image, pose ==> gpose
                idx = id_list.index(id)
                ############ for debug ############
                if self.blocks_info[i] is not None:
                    gp1 = self.P2A(self.blocks_info[i][1].position)
                    gp2 = self.P2A(gpose_list[idx].position)
                    dist = np.linalg.norm(gp1 - gp2)
                    if np.linalg.norm(gp1 - gp2) > 0.2:
                        rospy.logwarn(
                            f"Block {id} has moved a lot ({dist:.2f}) from {gp1} to {gp2}. Maybe misdetection."
                        )
                ###################################
                if self.blocks_info[i] is None:
                    self.blocks_info[i] = [
                        pose_list[idx],
                        gpose_list[idx],
                        self.this_image_time_ms,
                        True,
                    ]
                ## low pass filter
                p2 = self.P2A(pose_list[idx].position)
                q2 = self.Q2A(pose_list[idx].orientation)
                gp2 = self.P2A(gpose_list[idx].position)
                gq2 = self.Q2A(gpose_list[idx].orientation)
                p1 = self.P2A(self.blocks_info[i][0].position)
                q1 = self.Q2A(self.blocks_info[i][0].orientation)
                gp1 = self.P2A(self.blocks_info[i][1].position)
                gq1 = self.Q2A(self.blocks_info[i][1].orientation)
                self.blocks_info[i][0].position = self.A2P(self.LPF(p1, p2, inertia=0))
                self.blocks_info[i][0].orientation = self.A2Q(
                    self.sLPF(q1, q2, inertia=0)
                )
                self.blocks_info[i][1].position = self.A2P(
                    self.LPF(gp1, gp2, inertia=0.9)
                )
                self.blocks_info[i][1].orientation = self.A2Q(
                    self.sLPF(gq1, gq2, inertia=0)
                )
                self.blocks_info[i][2] = self.this_image_time_ms
                self.blocks_info[i][3] = True  ## in_cam is True if id in id_list
            elif self.blocks_info[i] is not None:
                ## update pose with last gpose (pose_in_cam is out-of-date). gpose => pose
                ## Assumption: block's not moving
                gpose = self.blocks_info[i][1]
                gpose_stamp = tf2_geometry_msgs.PoseStamped()
                gpose_stamp.header.stamp = rospy.Time.now()
                gpose_stamp.header.frame_id = coord_glb
                gpose_stamp.pose = gpose
                pose = tf2_geometry_msgs.do_transform_pose(gpose_stamp, inv_trans).pose
                self.blocks_info[i] = [pose, gpose, self.this_image_time_ms, False]
        ## publish blocks info
        self.publish_blocks_info()
        return

    def publish_blocks_info(self):
        marker_list = MarkerInfoList()
        for i in range(len(self.blocks_info)):
            if self.blocks_info[i] is None:
                continue
            marker = MarkerInfo()
            marker.id = i + 1
            marker.pose = self.blocks_info[i][0]
            marker.gpose = self.blocks_info[i][1]
            marker.in_cam = self.blocks_info[i][3]
            marker_list.markerInfoList.append(marker)
        self.pub_b.publish(marker_list)

    def visualization(self):
        while not rospy.is_shutdown():
            try:
                if self.current_visualization_image is not None:
                    cv2.imshow("visualization", self.current_visualization_image)
                    cv2.waitKey(33)
            except:
                self.collapsed = False
                rospy.logerr("The visualization window has collapsed!")


if __name__ == "__main__":
    rospy.init_node("image_node", anonymous=True)
    rter = Processor(initial_mode=ModeRequese.GameInfo, verbose=True)
    rospy.loginfo("Image thread started")
    rospy.spin()
