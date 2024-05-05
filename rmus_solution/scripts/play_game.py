#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import UInt8MultiArray
from rmus_solution.srv import (
    switch,
    setgoal,
    setgoalcoord,
    graspsignal,
    graspconfig,
    keepoutmode,
    graspsignalResponse,
)
from navi_control import PointName, router
from img_processor import ModeRequese
from manipulator import AlignRequest, ErrorCode
from rmus_solution.msg import MarkerInfoList, MarkerInfo
from geometry_msgs.msg import Point, Pose, PoseArray
import math
from navi_control import KeepOutMode, KeepOutArea


class gamecore:
    def __init__(self):
        self.wait_for_services()
        rospy.loginfo("Get all rospy sevice!")
        self.navigation = rospy.ServiceProxy("/navigation/goal", setgoal)
        self.navigation_coord = rospy.ServiceProxy("/navigation/goal/coord", setgoalcoord)
        self.aligner = rospy.ServiceProxy("/manipulator/grasp", graspsignal)
        self.img_switch_mode = rospy.ServiceProxy("/img_processor/mode", switch)
        self.swtch_align_mode = rospy.ServiceProxy("/manipulator/grasp_config", graspconfig)
        self.keep_out_mode = rospy.ServiceProxy("/keep_out_layer/mode", keepoutmode)
        """ gamecore state params: """
        # self.observing = True  ## if self.observing == True, classify the block to mining areas
        """ gamecore record data (global): """
        self.gameinfo = []
        self.block_mining_area = {1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1}
        self.blockinfo_dict = {}  ## {block_id: MarkerInfo}, stores latest block info (maybe incomplete if never detected by img_processor)
        self.stackinfo = [[], [], []]
        # self.areas_empty = [False, False, False]
        ## subscribe to gameinfo and blockinfo
        rospy.Subscriber("/get_gameinfo", UInt8MultiArray, self.update_gameinfo)
        rospy.Subscriber("/get_blockinfo", MarkerInfoList, self.update_blockinfo)
        ## switch to PID control (with angle alignment)
        self.swtch_align_mode(2, 1)
        """ start gamecore logic """
        ## initial pose
        self.aligner(AlignRequest.Reset, 0, 0)
        self.navigation(PointName.Home)
        ## get gameinfo
        self.keep_out_mode(KeepOutMode.AddAll, 0)
        self.img_switch_mode(ModeRequese.GameInfo)
        self.navigation(PointName.Noticeboard)
        self.keep_out_mode(KeepOutMode.RemoveAll, 0)
        assert len(self.gameinfo) == 3
        rospy.sleep(0.5)
        ## go get blocks
        self.img_switch_mode(ModeRequese.BlockInfo)
        self.cruise()
        self.keep_out_mode(KeepOutMode.AddAll, 0)
        self.aligner(AlignRequest.Reset, 0, 0)
        self.navigation(PointName.Park)

    def blks_in_sight(self):
        return [blk for blk in self.blockinfo_dict if self.blockinfo_dict[blk].in_cam]

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
            self.gameinfo = gameinfo.data

    def nearest_block(self) -> int:
        """ return nearest block id among visible blocks, return 0 if no block in sight """
        dists = []
        for id, blkinfo in self.blockinfo_dict.items():
            if not blkinfo.in_cam:
                continue
            p:Pose = blkinfo.pose
            dist = np.linalg.norm([p.position.x, p.position.z])
            dists.append((id, dist))
        dists.sort(key=lambda x: x[1])
        print(dists)
        return dists[0][0] if dists else 0

    def cruise(self):
        print("----------observing----------")
        self.img_switch_mode(ModeRequese.BlockInfo)
        spots = [PointName.MiningArea0, PointName.MiningArea1, PointName.MiningArea2]
        blks_stash = []
        ## traverse 3 areas to grasp and stack blocks
        for i in range(3):
            cnt_debug = 0
            while 1:
                cnt_debug += 1
                if cnt_debug > 4:
                    print("PANIC!")
                    break
                self.navigation(spots[i])
                blk_id = self.nearest_block()
                if not blk_id:
                    break
                print(f'----------grasping and stacking block {blk_id}----------')
                if not self.grasp(blk_id, retry=1):
                    ## panic: blk_id is the nearest block in sight, but cannot be grasped?!
                    ## break while loop and go to next area
                    print(f"PANIC: {blk_id} is the nearest block in sight, but cannot be grasped?!")
                    break
                ## deliver it to exchange spot or stash area
                ret = self.check_placeable(blk_id)
                if not ret:
                    ## to stash
                    ## stash_dst is (x, y, angle), in meter and rad
                    stash_dst = (0.9, 0.3, 0) if not hasattr(self, "stash_lst") else (0.9, self.stash_lst[1] - 0.25, 0)
                    blks_stash.append(blk_id)
                    self.navigation_coord(*stash_dst)
                    self.aligner(AlignRequest.Drop, 0, 0)
                    self.stash_lst = stash_dst
                else:
                    slot, layer = ret
                    ## to exchange station
                    self.navigation(PointName.Station_1 + slot - 7)
                    self.stack(blk_id, slot, layer)
                    self.stackinfo[slot-7].append(blk_id)
                    print(">>>>>>>>>>>>>>>>>>>>")
                    print(self.stackinfo)
                    print("<<<<<<<<<<<<<<<<<<<<")
        stash_vp = (0.5, 0.05, 0)
        slots_order = [7, 8, 7]
        layers_order = [2, 2, 3]
        for i, blk_id in enumerate(blks_stash):
            print(f'----------grasping and stacking block {blk_id}----------')
            self.navigation_coord(0.75, 0.3 - 0.25*i, 0)
            self.grasp(blk_id, retry=1)
            self.navigation(PointName.Station_1 + slots_order[i] - 7)
            self.stack(blk_id, slots_order[i], layers_order[i])

    def check_placeable(self, blk_id: int):
        """ 
        check if target block is in sight and can be stacked to exchange station.
            return (slot, layer) if target block can be stacked, else None
        """
        ret = None
        if blk_id in self.gameinfo:
            ret = (7 + self.gameinfo.index(blk_id), 1)
            return ret
        ## else blk is non gameinfo block, check if it's stackable (slot has 1 or 2 block)
        for i, stacked in enumerate(self.stackinfo):
            if not 1 <= len(stacked) <= 2:
                continue
            ret = (7 + i, len(stacked) + 1)
            if len(stacked) == 2:
                break
        return ret

    def update_blockinfo(self, blockinfo_list: MarkerInfoList):
        for blockinfo in blockinfo_list.markerInfoList:
            self.blockinfo_dict[blockinfo.id] = blockinfo
        return

    def grasp(self, block_id: int, retry: int = 0):
        """ grasp target block with retry given times.
                Assertion: target block's in sight.
                return True iff the block's not in sight after grasp action.
        """
        resp:graspsignalResponse = self.aligner(AlignRequest.Grasp, block_id, 0)
        ## because arm pos will be reset when grasp done, target block shouldn't be in sight.
        for i in range(1, retry + 1):
            if not self.blockinfo_dict[block_id].in_cam and resp.error_code == ErrorCode.Success:
                return True
            print(f"----------retry {i}----------")
            resp = self.aligner(AlignRequest.Grasp, block_id, 0)
            print(f"----------retry {i} done----------")
        return not self.blockinfo_dict[block_id].in_cam and resp.error_code == ErrorCode.Success

    def stack(self, block_id: int, slot: int, layer: int):
        """
        stack the block to the given slot and layer
        assertion: given slot is in sight
        """
        self.aligner(AlignRequest.Place, slot, layer)
        return False

    def get_layer(self, block_id: int):
        """ calc given block's layer, assuming block is in exchange spot """
        if (block_id not in self.blockinfo_dict or not self.blockinfo_dict[block_id].in_cam):
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
        """ calc horizontal bias of given block to given slot,
                return math.inf if block or slot not in sight
        """
        if (block_id not in self.blockinfo_dict or not self.blockinfo_dict[block_id].in_cam):
            return math.inf
        if (block_id not in self.blockinfo_dict or not self.blockinfo_dict[block_id].in_cam):
            return math.inf
        ## x, y, z axis of "map frame" point forwards, left and upwards respectively
        block_info = self.blockinfo_dict[block_id]
        slot_info = self.blockinfo_dict[slot]
        return abs(block_info.gpose.position.y - slot_info.gpose.position.y)

    def check_stacked_blocks(self, stackinfo: dict):
        """ check if blocks are stacked as stackinfo
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

if __name__ == "__main__":
    rospy.init_node("gamecore_node")
    node = gamecore()
    rospy.spin()
