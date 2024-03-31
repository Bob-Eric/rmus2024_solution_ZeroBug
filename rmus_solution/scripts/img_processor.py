#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, Point
from std_msgs.msg import UInt8MultiArray
from sensor_msgs.msg import Image, CameraInfo
from rmus_solution.srv import switch, switchResponse
from rmus_solution.msg import MarkerInfo, MarkerInfoList
from enum import IntEnum
import cv2
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation as R
from rtabmap_msgs.msg import RGBDImage
from detect import marker_detection


class ModeRequese(IntEnum):
    DoNothing = 0       ## disable marker_detection to save resources (160% cpu -> 60% cpu)
    BlockInfo = 1       ## actually not in use, blockinfo and gameinfo are detected simultaneously
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
                rospy.loginfo(
                    "camera_matrix :\n {}".format(self.camera_matrix)
                )
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
        rospy.Service("/image_processor_switch_mode", switch, self.modeCallBack)
        self.pub_p = rospy.Publisher("/get_gameinfo", UInt8MultiArray, queue_size=1)
        self.pub_b = rospy.Publisher("/get_blockinfo", MarkerInfoList, queue_size=1)
        self.pub_gpose_raw = rospy.Publisher("/gpose_raw", PoseArray, queue_size=10)
        self.pub_gpose_lpf = rospy.Publisher("/gpose_lpf", PoseArray, queue_size=10)
        self.pub_pose_raw = rospy.Publisher("/pose_raw", PoseArray, queue_size=10)
        self.pub_pose_lpf = rospy.Publisher("/pose_lpf", PoseArray, queue_size=10)
        self.detected_gameinfo = None
        self.blocks_info = [None] * 9
        self.blocks_info_lpf = [None] * 9   ## low pass filtered blocks_info

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
                self.image,
                self.camera_matrix,
                self.verbose
            )
        ## update gameinfo (and publish)
        self.update_gameinfo(id_list, tvec_list)
        ## update blocks_info (and publish)
        self.update_blocks_info(id_list, tvec_list, rvec_list)
        self.current_visualization_image = self.image

        ## publish gpose for visualization in rviz
        pose_msg = PoseArray()
        pose_msg.header.frame_id = "map"
        pose_msg.poses = [blockinfo[1] for blockinfo in self.blocks_info if blockinfo is not None]
        self.pub_gpose_raw.publish(pose_msg)
        pose_msg.poses = [blockinfo[1] for blockinfo in self.blocks_info_lpf if blockinfo is not None]
        self.pub_gpose_lpf.publish(pose_msg)

        pose_msg.header.frame_id = "camera_aligned_depth_to_color_frame_correct"
        pose_msg.poses = [blockinfo[0] for blockinfo in self.blocks_info if blockinfo is not None]
        self.pub_pose_raw.publish(pose_msg)
        pose_msg.poses = [blockinfo[0] for blockinfo in self.blocks_info_lpf if blockinfo is not None]
        self.pub_pose_lpf.publish(pose_msg)
        return

    def depthCallback(self, image):
        self.depth_img = self.bridge.imgmsg_to_cv2(image, "32FC1")

    def modeCallBack(self, req):
        if ModeRequese.DoNothing <= ModeRequese(req.mode) < ModeRequese.End:
            self.current_mode = ModeRequese(req.mode)
            return switchResponse(self.current_mode)
        else:
            return switchResponse(req.mode)

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
            rospy.logerr(f"detected different gameinfo: {gameinfo}, expected: {self.detected_gameinfo}")
        ## publish gameinfo
        self.pub_p.publish(UInt8MultiArray(data=self.detected_gameinfo))
        print(f"gameinfo: {gameinfo}")
        return
    
    def point2array(self, point:Point) -> np.ndarray:
        return np.array([point.x, point.y, point.z])
    
    def array2point(self, array) -> Point:
        point = Point()
        point.x, point.y, point.z = array
        return point

    def low_pass_filter(self, last_pos, new_pos, inertia=0.95) -> np.array:
        return inertia * last_pos + (1-inertia) * new_pos

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
        coord_cam = "camera_aligned_depth_to_color_frame_correct"
        coord_glb = "map"
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
                ## block `id` is detected in image
                idx = id_list.index(id)
                ############ for debug ############
                if self.blocks_info[i] is not None:
                    gp1 = self.point2array(self.blocks_info[i][1].position)
                    gp2 = self.point2array(gpose_list[idx].position)
                    dist = np.linalg.norm(gp1-gp2)
                    if np.linalg.norm(gp1 - gp2) > 0.2:
                        rospy.logwarn( f"Block {id} has moved a lot ({dist:.2f}) from {gp1} to {gp2}. Maybe misdetection.")
                ###################################
                self.blocks_info[i] = [pose_list[idx], gpose_list[idx], self.this_image_time_ms, True]  ## in_cam is True if id in id_list
            elif self.blocks_info[i] is not None:
                ## update pose_in_cam with last gpose (last pose_in_cam is out-of-date)
                ## Assumption: block's not moving
                gpose = self.blocks_info[i][1]
                gpose_stamp = tf2_geometry_msgs.PoseStamped()
                gpose_stamp.header.stamp = rospy.Time.now()
                gpose_stamp.header.frame_id = coord_glb
                gpose_stamp.pose = gpose
                pose = tf2_geometry_msgs.do_transform_pose(gpose_stamp, inv_trans).pose
                block_info_makeup = [pose, gpose, self.this_image_time_ms, False]
                self.blocks_info[i] = block_info_makeup
            ## no matter if blocks_info is observed or made up, it contains noise
            ## therefore, apply low pass filter
            if self.blocks_info[i] is not None:
                if self.blocks_info_lpf[i] is None:
                    self.blocks_info_lpf[i] = self.blocks_info[i]
                p1 = self.point2array(self.blocks_info_lpf[i][0].position)
                p2 = self.point2array(self.blocks_info[i][0].position)
                gp1 = self.point2array(self.blocks_info_lpf[i][1].position)
                gp2 = self.point2array(self.blocks_info[i][1].position)
                self.blocks_info_lpf[i][0].position = self.array2point(self.low_pass_filter(p1, p2))
                self.blocks_info_lpf[i][1].position = self.array2point(self.low_pass_filter(gp1, gp2))
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
    rter = Processor(initial_mode=ModeRequese.GameInfo, verbose=True)
    rospy.loginfo("Image thread started")
    rospy.spin()
