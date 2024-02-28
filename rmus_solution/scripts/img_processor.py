#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image, CameraInfo
from rmus_solution.srv import switch, switchResponse
from rmus_solution.msg import MarkerInfo, MarkerInfoList
from enum import IntEnum
import cv2
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation as R

from detect import marker_detection


class ModeRequese(IntEnum):
    DoNothing = 0
    One = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    B = 7
    O = 8
    X = 9
    GameInfo = 10

    End = 11


def pose_aruco_2_ros(rvec, tvec):
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
        rospy.Subscriber(
            "/camera/color/image_raw", Image, self.imageCallback, queue_size=1
        )
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self.depthCallback,
            queue_size=1,
        )
        rospy.Service("/image_processor_switch_mode", switch, self.modeCallBack)
        self.pub_p = rospy.Publisher("/get_gameinfo", UInt8MultiArray, queue_size=1)
        self.pub_b = rospy.Publisher("/get_blockinfo", MarkerInfoList, queue_size=1)
        self.detected_gameinfo = None
        self.blocks_info = [None] * 9

    def imageCallback(self, image):
        self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.this_image_time_ms = int(image.header.stamp.nsecs / 1e6) + int(
            1000 * (image.header.stamp.secs - self.start_time)
        )
        locked_current_mode = self.current_mode

        if locked_current_mode == ModeRequese.DoNothing:
            ## only try detection, don't update
            marker_detection(self.image, self.camera_matrix, self.verbose)
        else:
            ## update blocks_info (block 1-6 and B, O, X) and publish it
            self.update_blocks_info(self.image)
            self.publish_blocks_info()

        # detect 3 wanted blocks from gameinfo board
        if locked_current_mode == ModeRequese.GameInfo:
            detected_gameinfo = self.get_gameinfo(self.image)
            if detected_gameinfo is not None:
                if self.detected_gameinfo is None:
                    self.detected_gameinfo = detected_gameinfo
                assert tuple(self.detected_gameinfo) == tuple(detected_gameinfo)
            self.pub_p.publish(UInt8MultiArray(data=self.detected_gameinfo))
        self.current_visualization_image = self.image
        return 

    def depthCallback(self, image):
        self.depth_img = self.bridge.imgmsg_to_cv2(image, "32FC1")

    def modeCallBack(self, req):
        if ModeRequese.DoNothing <= ModeRequese(req.mode) < ModeRequese.End:
            self.current_mode = ModeRequese(req.mode)
            return switchResponse(self.current_mode)
        else:
            return switchResponse(req.mode)

    def get_gameinfo(self, image):
        gameinfo = [0, 0, 0]
        id_list, quads_list, area_list, tvec_list, rvec_list = marker_detection(
            image,
            camera_matrix=self.camera_matrix,
            verbose=self.verbose,
            exchange_station=True,
        )

        digits_list = []
        for id, quads in zip(id_list, quads_list):
            x = (quads[0][0][0] + quads[1][0][0] + quads[2][0][0] + quads[3][0][0]) / 4
            digits_list.append((id, x))
        if len(digits_list) != 3:
            print(f"detected {len(digits_list)} digits in gameinfo board, not 3")
            return None
        digits_list.sort(key=lambda pair: pair[1])
        gameinfo = [id for (id, x) in digits_list]
        print(gameinfo)
        return gameinfo

    def update_blocks_info(self, image):
        id_list, quads_list, area_list, tvec_list, rvec_list = marker_detection(
            image,
            camera_matrix=self.camera_matrix,
            verbose=self.verbose,
            height_range=(-0.2, 10.0),
        )

        pose_list = [pose_aruco_2_ros(r, t) for t, r in zip(tvec_list, rvec_list)]
        gpose_list = []
        coord_cam = "camera_aligned_depth_to_color_frame_correct"
        coord_glb = "map"
        trans = self.tfBuffer.lookup_transform(
            coord_glb, coord_cam, rospy.Time(), rospy.Duration(0.2)
        )
        inv_trans = self.tfBuffer.lookup_transform(
            coord_cam, coord_glb, rospy.Time(), rospy.Duration(0.2)
        )
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
                ## block `id` is detected in image
                idx = id_list.index(id)
                block_info = [pose_list[idx], gpose_list[idx], self.this_image_time_ms]
                self.blocks_info[i] = block_info
            elif self.blocks_info[i] is not None:
                ## update pose_in_cam with last gpose (last pose_in_cam is out-of-date)
                ## Assumption: block's not moving
                ## TODO: test if it's working when lose target temporarily
                gpose = self.blocks_info[i][1]
                gpose_stamp = tf2_geometry_msgs.PoseStamped()
                gpose_stamp.header.stamp = rospy.Time.now()
                gpose_stamp.header.frame_id = coord_glb
                gpose_stamp.pose = gpose
                pose = tf2_geometry_msgs.do_transform_pose(gpose_stamp, inv_trans).pose
                block_info_makeup = [pose, gpose, self.this_image_time_ms]
                self.blocks_info[i] = block_info_makeup
                ## if not working, just set it to None
                # this.blocks_info[i] = None
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
            marker_list.markerInfoList.append(marker)
        self.pub_b.publish(marker_list)

    ## not used yet, but may be useful when sim2real due to noise
    def get_current_depth(self, quads):
        locked_depth = self.depth_img
        new_contour = quads.copy()
        new_contour[0, 0, 0], new_contour[2, 0, 0] = int(
            quads[0, 0, 0] * 0.75 + 0.25 * quads[2, 0, 0]
        ), int(quads[0, 0, 0] * 0.25 + 0.75 * quads[2, 0, 0])
        new_contour[0, 0, 1], new_contour[2, 0, 1] = int(
            quads[0, 0, 1] * 0.75 + 0.25 * quads[2, 0, 1]
        ), int(quads[0, 0, 1] * 0.25 + 0.75 * quads[2, 0, 1])
        new_contour[1, 0, 0], new_contour[3, 0, 0] = int(
            quads[1, 0, 0] * 0.75 + 0.25 * quads[3, 0, 0]
        ), int(quads[1, 0, 0] * 0.25 + 0.75 * quads[3, 0, 0])
        new_contour[1, 0, 1], new_contour[3, 0, 1] = int(
            quads[1, 0, 1] * 0.75 + 0.25 * quads[3, 0, 1]
        ), int(quads[1, 0, 1] * 0.25 + 0.75 * quads[3, 0, 1])

        mask = np.zeros(locked_depth.shape, np.uint8)
        cv2.drawContours(mask, [new_contour], -1, 255, -1)
        depth = int(cv2.mean(locked_depth, mask=mask)[0] * 1000)
        if self.verbose:
            cv2.putText(
                mask,
                str(depth) + "mm",
                (new_contour[0, 0, 0], new_contour[0, 0, 1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            self.current_visualization_image = (
                np.expand_dims(mask, axis=-1) * 0.5 + self.image * 0.5
            ).astype(np.uint8)
        return depth

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
    rter = Processor(initial_mode=ModeRequese.DoNothing, verbose=True)
    rospy.loginfo("Image thread started")
    rospy.spin()
