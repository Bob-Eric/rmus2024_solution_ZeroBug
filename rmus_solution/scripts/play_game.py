#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import UInt8MultiArray
from rmus_solution.srv import switch, setgoal, graspsignal, graspconfig, keepoutmode
from navi_control import PointName, router
from img_processor import ModeRequese
from manipulator import AlignRequest
from rmus_solution.msg import MarkerInfoList, MarkerInfo
from geometry_msgs.msg import Point, PoseArray
import math
from navi_control import KeepOutMode, KeepOutArea


class gamecore:
    mining_area_coord = [(0, 1.0), (0.65, 3.3), (2.4, -0.15)]

    def __init__(self):
        self.wait_for_services()

        rospy.loginfo("Get all rospy sevice!")
        self.navigation = rospy.ServiceProxy("/navigation/goal", setgoal)
        self.aligner = rospy.ServiceProxy("/manipulator/grasp", graspsignal)
        self.img_switch_mode = rospy.ServiceProxy("/img_processor/mode", switch)
        self.swtch_align_mode = rospy.ServiceProxy("/manipulator/grasp_config", graspconfig)
        self.keep_out_mode = rospy.ServiceProxy("/keep_out_layer/mode", keepoutmode)
        """ gamecore state params: """
        self.observing = (
            True  ## if self.observing == True, classify the block to mining areas
        )
        """ gamecore record data (global): """
        self.gameinfo = None
        self.block_mining_area = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1}
        self.blockinfo_dict = (
            {}
        )  ## {block_id: MarkerInfo}, stores latest block info (maybe incomplete if never detected by img_processor)
        ## subscribe to gameinfo and blockinfo
        rospy.Subscriber("/get_gameinfo", UInt8MultiArray, self.update_gameinfo)
        rospy.Subscriber("/get_blockinfo", MarkerInfoList, self.update_blockinfo)
        ## switch to PID control (no angle alignment)
        self.swtch_align_mode(2, 0)
        """ start gamecore logic """
        self.navigation(PointName.Home, "")
        self.aligner(AlignRequest.Reset, 0, 0)
        rospy.sleep(2)
        """ gamecore logic: """
        self.img_switch_mode(ModeRequese.BlockInfo)
        self.observation()
        self.grasp_and_place()
        self.aligner(AlignRequest.Reset, 0, 0)
        self.navigation(PointName.Park, "")

    def wait_for_services(self):
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/navigation/goal", 1.0)
                break
            except:
                rospy.logwarn("Waiting for navigation/goal Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/manipulator/grasp", 1.0)
                break
            except:
                rospy.logwarn("Waiting for /manipulator/grasp Service")
                rospy.sleep(0.5)

        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service("/img_processor/mode", 1.0)
                break
            except:
                rospy.logwarn("Waiting for /img_processor/mode Service")
                rospy.sleep(0.5)

    def update_gameinfo(self, gameinfo: UInt8MultiArray):
        if gameinfo.data is not None:
            self.gameinfo = gameinfo

    def observation(self):
        self.keep_out_mode(KeepOutMode.AddAll, 0)
        print("----------observing----------")
        self.img_switch_mode(ModeRequese.BlockInfo)
        self.navigation(PointName.MiningArea_0_Vp_1, "")
        rospy.sleep(2)
        self.navigation(PointName.MiningArea_0_Vp_2, "")
        rospy.sleep(2)
        self.img_switch_mode(ModeRequese.GameInfo)
        self.navigation(PointName.Noticeboard_2, "")
        rospy.sleep(2)
        self.img_switch_mode(ModeRequese.BlockInfo)
        self.navigation(PointName.MiningArea_1_Vp_1, "")
        rospy.sleep(2)
        self.navigation(PointName.MiningArea_1_Vp_2, "")
        rospy.sleep(2)
        self.navigation(PointName.MiningArea_2_Vp_1, "")
        rospy.sleep(2)
        self.navigation(PointName.MiningArea_2_Vp_2, "")
        rospy.sleep(2)
        self.observing = False
        print("----------done observing----------")
        self.keep_out_mode(KeepOutMode.RemoveAll, 0)

    def update_blockinfo(self, blockinfo_list: MarkerInfoList):
        for blockinfo in blockinfo_list.markerInfoList:
            self.blockinfo_dict[blockinfo.id] = blockinfo
        if self.observing:
            self.assign_area(blockinfo_list)
        return

    def assign_area(self, blockinfo_list: MarkerInfoList):
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

    def go_get_block(self, block_id: int, retry: int = 0):
        """ 
        go to mining_area if target block's not in sight, try grasp the block once.
            Assertion: target block must in sight in current pose, or spot0 or spot1 in mining area.
            return True iff the block's not in sight after grasp action.
        """
        area_idx = self.block_mining_area[block_id]
        if area_idx == -1:
            rospy.logwarn(f"panic in go_get_block(): block {block_id} is not in the mining area.")
            return False
        navi_areas = [
            PointName.MiningArea_0_Vp_2,
            PointName.MiningArea_0_Vp_1,
            PointName.MiningArea_1_Vp_2,
            PointName.MiningArea_1_Vp_1,
            PointName.MiningArea_2_Vp_1,
            PointName.MiningArea_2_Vp_2,
        ]
        print(f"fetching block {block_id} from area No.{area_idx}")
        self.img_switch_mode(ModeRequese.BlockInfo)
        if not self.blockinfo_dict[block_id].in_cam:
            print(f"block {block_id} not found in sight, go to spot0...")
            self.navigation(navi_areas[2*area_idx], "")
            rospy.sleep(0.5)
        if not self.blockinfo_dict[block_id].in_cam:
            print(f"block {block_id} not found in spot0, go to spot1...")
            self.navigation(navi_areas[2*area_idx+1], "")
            rospy.sleep(0.5)
        if not self.blockinfo_dict[block_id].in_cam:
            rospy.logwarn(f"panic in go_get_block(): block {block_id} is not in sight. cannot find it.")
            return False
        self.aligner(AlignRequest.Grasp, block_id, 0)
        ## because arm pos will be reset when grasp done, target block shouldn't be sight.
        for i in range(1, retry+1):
            if self.blockinfo_dict[block_id].in_cam:
                print(f"----------retry {i}----------")
                self.aligner(AlignRequest.Grasp, block_id, 0)
                print(f"----------retry {i} done----------")
        return not self.blockinfo_dict[block_id].in_cam

    def stack(self, block_id: int, slot: int, layer: int):
        """stack the block to the given slot and layer"""
        self.navigation(slot, "")
        self.aligner(AlignRequest.Place, slot, layer)
        return False

    def get_layer(self, block_id: int):
        """calc given block's layer, assuming block is in exchange spot"""
        if (
            block_id not in self.blockinfo_dict
            or not self.blockinfo_dict[block_id].in_cam
        ):
            return -1
        block_info = self.blockinfo_dict[block_id]
        ## TODO: height base is not correct
        ## block in layer 1 is at height of height_base
        block_size, height_base = 0.05, -0.045
        block_height = block_info.gpose.position.z
        layer = round((block_height - height_base) / block_size) + 1
        print(f"block_height:{block_height:.3f}; layer:{layer}")
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
            done = self.go_get_block(target, retry=1)
            if not done:
                continue
            self.navigation(PointName.Station_1 + i, "")
            self.stack(target, 7 + i, 1)
            print(f"----------done grasping No.{i} block(id={target})----------")
        print("----------done grasping three basic blocks----------")
        blocks_left = [i for i in range(1, 6 + 1) if i not in self.gameinfo.data]
        print(f"stacking the rest of the blocks: {blocks_left}")
        self.swtch_align_mode(1, 1)
        slots_order = [7, 7, 8]
        layers_order = [2, 3, 2]
        for i, target in enumerate(blocks_left):
            print(f"----------grasping No.{i} block(id={target})----------")
            done = self.go_get_block(target, retry=1)
            if not done:
                continue
            self.navigation(slots_order[i], "")
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
