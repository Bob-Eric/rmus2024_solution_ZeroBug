#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import UInt8MultiArray
from rmus_solution.srv import switch, setgoal, graspsignal, graspsignalResponse
from navi_control import PointName, router
from img_processor import ModeRequese
from manipulator import AlignRequest
from rmus_solution.msg import MarkerInfoList, MarkerInfo
from geometry_msgs.msg import Point
import math


prefix = "[gamecore]"
sysprint = print


def print(*args, **kwargs):
    sysprint(prefix, end="")
    sysprint(*args, **kwargs)


class gamecore:
    mining_area_coord = [(0, 1.0), (0.65, 3.3), (2.4, -0.15)]

    def __init__(self):
        self.wait_for_services()

        rospy.loginfo(prefix + "Get all rospy sevice!")
        self.navigation = rospy.ServiceProxy("/set_navigation_goal", setgoal)
        self.aligner = rospy.ServiceProxy("/let_manipulater_work", graspsignal)
        self.img_switch_mode = rospy.ServiceProxy(
            "/image_processor_switch_mode", switch
        )
        self.gameinfo = None
        rospy.Subscriber("/get_gameinfo", UInt8MultiArray, self.update_game_info)
        rospy.sleep(2)

        self.align_res = self.aligner(AlignRequest.Reset, 0, 0)
        self.response = self.img_switch_mode(ModeRequese.GameInfo)
        self.navigation_result = self.navigation(PointName.Noticeboard_1, "")
        rospy.sleep(2)
        self.navigation_result = self.navigation(PointName.Noticeboard_2, "")
        rospy.sleep(2)

        while not rospy.is_shutdown():
            if self.gameinfo is not None:
                break
            else:
                rospy.logwarn(prefix + "Waiting for gameinfo message.")
            rospy.sleep(0.5)

        self.block_mining_area = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1}
        self.blockinfo_dict = {}
        rospy.Subscriber("/get_blockinfo", MarkerInfoList, self.update_block_info)

        """ gamecore state params: """
        self.observing = (
            True  ## if self.observing == True, classify the block to mining areas
        )

        """ gamecore logic: """
        self.observation()
        self.grasp_and_place()
        self.align_res = self.aligner(AlignRequest.Reset, 0, 0)
        self.observation()
        self.navigation_result = self.navigation(PointName.Park, "")

    def wait_for_services(self):
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/set_navigation_goal", 1.0)
                break
            except:
                rospy.logwarn(prefix + "Waiting for set_navigation_goal Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/let_manipulater_work", 1.0)
                break
            except:
                rospy.logwarn(prefix + "Waiting for let_manipulater_work Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/image_processor_switch_mode", 1.0)
                break
            except:
                rospy.logwarn(
                    prefix + "Waiting for image_processor_switch_mode Service"
                )
                rospy.sleep(0.5)

    def update_game_info(self, gameinfo: UInt8MultiArray):
        if gameinfo.data is not None:
            self.gameinfo = gameinfo

    def observation(self):
        print("----------observing----------")
        targets = [
            PointName.MiningArea_0_Vp_1, PointName.MiningArea_0_Vp_2, 
            PointName.MiningArea_1_Vp_1, PointName.MiningArea_1_Vp_2, 
            PointName.MiningArea_2_Vp_1, PointName.MiningArea_2_Vp_2]
        for target in targets:
            self.navigation_result = self.navigation(target, "")
            rospy.sleep(3)
        self.observing = False
        print("----------done observing----------")

    def update_block_info(self, blockinfo_list: MarkerInfoList):
        for blockinfo in blockinfo_list.markerInfoList:
            self.blockinfo_dict[blockinfo.id] = blockinfo
        # for blockinfo in blockinfo_list.markerInfoList:
        #     if blockinfo.id != 4 and blockinfo.id != 6:
        #         if blockinfo.in_cam:
        #             rospy.logwarn(prefix + f"Block {blockinfo.id} is in the cam.")
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
            rospy.logwarn(prefix + f"block {block_id} is not in the mining area")
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

    def stack(self, block_id: int, slot: int, layer: int):
        """stack the block to the given slot and layer"""
        max_attempt = 5
        hbias_allow = 0.018  # 1.8cm horizontal bias is allowed
        for i in range(max_attempt):
            print(
                f"Attempt {i}: stack block {block_id} to layer {layer} of slot {slot}."
            )
            # try place
            self.align_res = self.aligner(AlignRequest.Place, slot, layer)
            # go to the front to check
            self.navigation_result = self.navigation(PointName.Station_Front, "")
            # if not in sight from the front, it must falls to the back
            if not self.blockinfo_dict[block_id].in_cam:
                print(
                    f"Block {block_id} is not in sight from the front. Now go to the back."
                )
                self.navigation_result = self.navigation(PointName.Station_Back, "")
            # if not in sight from the back... How could it possible?! just return false
            if not self.blockinfo_dict[block_id].in_cam:
                print(
                    f"Block {block_id} is not in sight from the back. What the fuck?! I quit."
                )
                return False
            # check if block's stacked well by horizontal bias
            hbias = self.get_hbias(block_id, slot)
            if self.get_layer(block_id) == layer and hbias < hbias_allow:
                print(
                    f"Success: block {block_id} is in layer {layer} of slot {slot} with hbias of {hbias}."
                )
                return True
            else:
                print(f"Result: hbias of {hbias}.")
            # won't pick up the block at the last attempt
            if i < max_attempt - 1:
                self.align_res = self.aligner(AlignRequest.Grasp, block_id, 0)
        print(f"Max attempt reached. stack failed.")
        return False

    def get_layer(self, block_id: int):
        """calc given block's layer, assuming block is in exchange spot"""
        if (
            block_id not in self.blockinfo_dict
            or not self.blockinfo_dict[block_id].in_cam
        ):
            return -1
        block_info = self.blockinfo_dict[block_id]
        # block in layer 1 is at height of height_base
        block_size, height_base = 0.05, -0.045
        layer = round((block_info.gpose.position.z - height_base) / block_size) + 1
        return layer

    def get_hbias(self, block_id: int, slot: int):
        """calc horizontal bias of given block to given slot,
        return math.inf if block or slot not in sight"""
        if (
            block_id not in self.blockinfo_dict
            or not self.blockinfo_dict[block_id].in_cam
        ):
            return math.inf
        if (
            block_id not in self.blockinfo_dict
            or not self.blockinfo_dict[block_id].in_cam
        ):
            return math.inf
        ## x, y, z axis of "map frame" point forwards, left and upwards respectively
        block_info = self.blockinfo_dict[block_id]
        slot_info = self.blockinfo_dict[slot]
        return abs(block_info.gpose.position.y - slot_info.gpose.position.y)

    def check_stacked_blocks(self, stackinfo: dict):
        """check if blocks are stacked as stackinfo

        `stackinfo`: a dict like {block_id: (slot, layer)}
        """
        for block_id, (slot, layer) in stackinfo.items():
            # if block_id not in self.blockinfo_dict or not self.blockinfo_dict[block_id].in_cam:
            #     print(f"Bad view: block {block_id} is not in sight.")
            #     return False
            if self.get_layer(block_id) != layer:
                print(f"Bad stacking: block {block_id} is not in layer {layer}.")
                return False
            hbias = self.get_hbias(block_id, slot)
            if hbias > 0.02:
                ## bias of center is more than half of block size
                print(
                    f"Bad stacking: block {block_id} is not in slot {slot}. \
                    horizontal bias is {hbias * 100} > 2cm."
                )
                return False
        return True

    def grasp_and_place(self):
        print("----------grasping three basic blocks----------")
        for i, target in enumerate(self.gameinfo.data):
            print(f"----------grasping No.{i} block(id={target})----------")
            done = self.go_get_block(target)
            if not done:
                ## TODO: failure logic
                continue
            self.navigation_result = self.navigation(PointName.Station_1 + i, "")
            self.stack(target, 7 + i, 1)
            print(f"----------done grasping No.{i} block(id={target})----------")
        print("----------done grasping three basic blocks----------")
        blocks_left = [i for i in range(1, 6 + 1) if i not in self.gameinfo.data]
        print(f"stacking the rest of the blocks: {blocks_left}")
        slots_order = [7, 7, 8]
        layers_order = [2, 3, 2]
        for i, target in enumerate(blocks_left):
            print(f"----------grasping No.{i} block(id={target})----------")
            done = self.go_get_block(target)
            if not done:
                ## TODO: failure logic
                continue
            self.navigation_result = self.navigation(PointName.Station_2, "")
            self.stack(target, slots_order[i], layers_order[i])
            print(f"----------done stacking No.{i} block(id={target})----------")
        print("----------done stacking three blocks----------")

        ## check stacking
        b1, b2, b3 = self.gameinfo.data
        b4, b5, b6 = blocks_left
        self.check_stacked_blocks(
            {b1: (7, 1), b2: (8, 1), b3: (9, 1), b4: (7, 2), b5: (7, 3), b6: (8, 2)}
        )


if __name__ == "__main__":
    rospy.init_node("gamecore_node")
    node = gamecore()
    rospy.spin()
