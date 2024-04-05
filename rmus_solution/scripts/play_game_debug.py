#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import UInt8MultiArray
from rmus_solution.srv import switch, setgoal, graspsignal
from manipulator import AlignRequest
from navi_control import PointName
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
        self.aligner = rospy.ServiceProxy("/let_manipulator_work", graspsignal)
        self.img_switch_mode = rospy.ServiceProxy(
            "/image_processor_switch_mode", switch
        )
        rospy.Subscriber("/get_gameinfo", UInt8MultiArray, self.update_game_info)
        rospy.sleep(1)

        self.align_res = self.aligner(AlignRequest.Reset, 0, 0)
        self.response = self.img_switch_mode(ModeRequese.GameInfo)
        self.navigation_result = self.navigation(PointName.Noticeboard, "")

        while not rospy.is_shutdown():
            if self.gameinfo is not None and self.gameinfo.data is not None:
                break
            else:
                rospy.logwarn("Waiting for gameinfo message.")
            rospy.sleep(0.5)

        self.block_mining_area = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1}
        self.blockinfo_list = MarkerInfoList()
        rospy.Subscriber("/get_blockinfo", MarkerInfoList, self.update_block_info)

        """ gamecore state params: """
        self.observing = (
            True  ## if self.observing == True, classify the block to mining areas
        )

        """ gamecore logic: """
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
                rospy.wait_for_service("/let_manipulator_work", 1.0)
                break
            except:
                rospy.logwarn("Waiting for let_manipulator_work Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/image_processor_switch_mode", 1.0)
                break
            except:
                rospy.logwarn("Waiting for image_processor_switch_mode Service")
                rospy.sleep(0.5)

    def update_game_info(self, gameinfo: UInt8MultiArray):
        if gameinfo.data is not None:
            self.gameinfo = gameinfo

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

    def go_get_block(self, block_id: int):
        area_idx = self.block_mining_area[block_id]
        if area_idx == -1:
            rospy.logwarn(f"block {block_id} is not in the mining area")
            return False
        navi_areas = [
            PointName.MiningArea_0_Vp_2,
            PointName.MiningArea_1_Vp_2,
            PointName.MiningArea_2_Vp_1,
        ]
        dest = navi_areas[area_idx]
        print(f"fetching block {block_id} from area No.{area_idx}")
        self.navigation_result = self.navigation(dest, "")
        self.align_res = self.aligner(AlignRequest.Grasp, block_id, 0)
        return True

    def stack(self):
        """stack above highest block in sight"""
        if self.blockinfo_list is None or len(self.blockinfo_list.markerInfoList) == 0:
            return
        block_list = [
            blockinfo
            for blockinfo in self.blockinfo_list.markerInfoList
            if blockinfo.in_cam
        ]
        ## frame "map" x,y,z points forwards, right and upwards respectively
        block_list.sort(key=lambda blockinfo: blockinfo.gpose.position.z, reverse=True)
        highest_block = block_list[0]
        height = highest_block.gpose.position.z

        print(f"highest block id: {highest_block.id}, gpose height: {height}")

        ## blocks in first layer is 0.035m above the base
        height_base = 0.035 - 0.08
        block_size = 0.05
        layers = round((height - height_base) / block_size)
        ## stack on the left (by default, align with "B" instead of highest block to eliminate systematic error)
        self.align_res = self.aligner(AlignRequest.Place, 7, 1 + layers + 1)
        ## TODO: failure logic
        return

    def check_stacked_blocks(self):
        pass

    def grasp_and_place(self):
        print("----------grasping three basic blocks----------")
        # for i, target in enumerate(self.gameinfo.data):
        #     print(f"----------grasping No.{i} block(id={target})----------")
        #     done = self.go_get_block(target)
        #     if not done:
        #         ## TODO: failure logic
        #         continue
        #     self.navigation_result = self.navigation(PointName.Station_1 + i, "")
        #     self.align_res = self.aligner(AlignRequest.Place, 7 + i, 1)
        #     print(f"----------done grasping No.{i} block(id={target})----------")
        print("----------done grasping three basic blocks----------")
        blocks_left = range(1, 6 + 1)
        print(f"stacking the rest of the blocks: {blocks_left}")
        for i, target in enumerate(blocks_left):
            print(f"----------grasping No.{i} block(id={target})----------")
            done = self.go_get_block(target)
            if not done:
                ## TODO: failure logic
                continue
            self.navigation_result = self.navigation(PointName.Station_1, "")
            self.stack()
            print(f"----------done stacking No.{i} block(id={target})----------")
        print("----------done stacking three blocks----------")


if __name__ == "__main__":
    rospy.init_node("gamecore_node")
    node = gamecore()
    rospy.spin()
