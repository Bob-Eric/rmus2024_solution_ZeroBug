#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import UInt8MultiArray
from rmus_solution.srv import switch, setgoal, graspsignal, graspsignalResponse
from manipulater import AlignerworkRequest
from navi_control import PointName, router
from img_processor import ModeRequese
from rmus_solution.msg import MarkerInfoList, MarkerInfo
from geometry_msgs.msg import Point
import math


class gamecore:
    mining_area_coord = [(0, 1.0), (0.65, 3.3), (2.4, -0.15)]

    def __init__(self):
        self.wait_for_services()

        rospy.loginfo("Get all rospy sevice!")
        self.navigation = rospy.ServiceProxy("/set_navigation_goal", setgoal)
        self.aligner = rospy.ServiceProxy("/let_manipulater_work", graspsignal)
        self.img_switch_mode = rospy.ServiceProxy(
            "/image_processor_switch_mode", switch
        )
        rospy.sleep(1)

        self.align_res = self.aligner(AlignerworkRequest.Reset, 0, "")
        self.response = self.img_switch_mode(ModeRequese.GameInfo)
        self.navigation_result = self.navigation(PointName.Noticeboard, "")

        while not rospy.is_shutdown():
            try:
                self.gameinfo: UInt8MultiArray = rospy.wait_for_message(
                    "/get_gameinfo", UInt8MultiArray, timeout=7
                )
                break
            except:
                rospy.logwarn("Waiting for gameinfo message.")
            rospy.sleep(0.5)

        self.block_mining_area = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1}
        self.blockinfo_list = MarkerInfoList()
        rospy.Subscriber("/get_blockinfo", MarkerInfoList, self.update_block_info)

        """ gamecore state params: """
        self.observing = True          ## if self.observing == True, classify the block to mining areas

        """ gamecore logic: """
        # self.test_navigation()
        self.observation()
        self.grasp_and_place()
        self.navigation_result = self.navigation(PointName.Park, "")

    def wait_for_services(self):
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/set_navigation_goal", 1.0)
                break
            except:
                rospy.logwarn("Waiting for set_navigation_goal Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/let_manipulater_work", 1.0)
                break
            except:
                rospy.logwarn("Waiting for let_manipulater_work Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/image_processor_switch_mode", 1.0)
                break
            except:
                rospy.logwarn("Waiting for image_processor_switch_mode Service")
                rospy.sleep(0.5)

    def observation(self):
        print("----------observing----------")
        self.navigation_result = self.navigation(PointName.MiningArea_0_Vp_1, "")
        # self.navigation_result = self.navigation(PointName.MiningArea_0_Vp_2, "")
        self.navigation_result = self.navigation(PointName.MiningArea_1_Vp_1, "")
        # self.navigation_result = self.navigation(PointName.MiningArea_1_Vp_2, "")
        self.navigation_result = self.navigation(PointName.MiningArea_2_Vp_1, "")
        # self.navigation_result = self.navigation(PointName.MiningArea_2_Vp_2, "")
        self.observation = False
        print("----------done observing----------")

    def update_block_info(self, blockinfo_list: MarkerInfoList):
        self.blockinfo_list = blockinfo_list
        blockinfo: MarkerInfo
        # for blockinfo in blockinfo_list.markerInfoList:
        #     if blockinfo.id != 4 and blockinfo.id != 6:
        #         if blockinfo.in_cam:
        #             rospy.logwarn(f"Block {blockinfo.id} is in the cam.")
        if self.observing:
            self.classify_block(blockinfo_list)
        return

    def classify_block(self, blockinfo_list: MarkerInfoList):
        blockinfo: MarkerInfo
        for blockinfo in blockinfo_list.markerInfoList:
            if 1 <= blockinfo.id <= 6:
                mining_area_id = self.check_near_mining_area(blockinfo.gpose.position)
                self.block_mining_area[blockinfo.id] = mining_area_id
        # print(self.block_mining_area)

    def check_near_mining_area(self, pos: Point):
        dist_mining_area = [100.0] * 3

        for i in range(3):

            MiningAreaCtrPos = self.mining_area_coord[i]
            dist_mining_area[i] = math.sqrt(
                (MiningAreaCtrPos[0] - pos.x) ** 2 + (MiningAreaCtrPos[1] - pos.y) ** 2
            )
        if min(dist_mining_area) < 0.75:
            mining_area_id = dist_mining_area.index(min(dist_mining_area))
        else:
            mining_area_id = 0

        return mining_area_id
        ...

    def test_navigation(self):
        self.response = self.img_switch_mode(ModeRequese.DoNothing)
        for i in range(0, 3):
            self.navigation_result = self.navigation(
                PointName.MiningArea_0_Vp_1 + i, ""
            )
            for j, target in enumerate(self.gameinfo.data):
                if j < i:
                    continue
                self.response = self.img_switch_mode(target)
                aligner_response: graspsignalResponse = self.aligner(
                    AlignerworkRequest.Grasp, ""
                )
                if aligner_response.res == True:
                    if aligner_response.response == "Successfully Grasp fake":
                        continue
                    else:
                        break

            self.response = self.img_switch_mode(ModeRequese.DoNothing)
            self.navigation_result = self.navigation(PointName.Station_1 + i, "")
            self.response = self.img_switch_mode(ModeRequese.B + i)

            aligner_response = self.aligner(AlignerworkRequest.Place, "")

            self.response = self.img_switch_mode(ModeRequese.DoNothing)
        self.navigation_result = self.navigation(PointName.Park, "")
        self.response = self.img_switch_mode(ModeRequese.DoNothing)
        ...

    def grasp_and_place(self):
        print("----------grasping three basic blocks----------")
        for i, target in enumerate(self.gameinfo.data):
            print(f"----------grasping No.{i} block(id={target})----------")
            mining_area_id = self.block_mining_area[target]
            rospy.loginfo(f"target: {target}, mining_area_id: {mining_area_id}")
            if mining_area_id == 0:
                self.navigation_result = self.navigation(
                    PointName.MiningArea_0_Vp_2, ""
                )
            elif mining_area_id == 1:
                self.navigation_result = self.navigation(
                    PointName.MiningArea_1_Vp_2, ""
                )
            elif mining_area_id == 2:
                self.navigation_result = self.navigation(
                    PointName.MiningArea_2_Vp_1, ""
                )
            else:
                rospy.logwarn(f"Block {target} is not in any mining area.")
                continue
            self.align_res = self.aligner(AlignerworkRequest.Grasp, target, "")
            self.navigation_result = self.navigation(PointName.Station_1 + i, "")
            self.align_res = self.aligner(AlignerworkRequest.Place, 7 + i, "")
            print(f"----------done grasping No.{i} block(id={target})----------")
        print("----------done grasping three basic blocks----------")
        


if __name__ == "__main__":
    rospy.init_node("gamecore_node")
    node = gamecore()
    rospy.spin()
