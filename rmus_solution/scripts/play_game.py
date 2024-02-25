#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import UInt8MultiArray
from rmus_solution.srv import switch, setgoal, graspsignal, graspsignalResponse
from manipulater import TrimerworkRequest
from navi_control import MissionResquest


class gamecore:
    def __init__(self):
        self.wait_for_services()

        rospy.loginfo("Get all rospy sevice!")
        self.navigation = rospy.ServiceProxy("/set_navigation_goal", setgoal)
        self.trimer = rospy.ServiceProxy("/let_manipulater_work", graspsignal)
        self.img_switch_mode = rospy.ServiceProxy(
            "/image_processor_switch_mode", switch
        )
        rospy.sleep(2)

        self.trim_res = self.trimer(TrimerworkRequest.Reset, "")
        self.response = self.img_switch_mode(9)
        self.navigation_result = self.navigation(MissionResquest.Noticeboard, "")

        while not rospy.is_shutdown():
            try:
                self.gameinfo: UInt8MultiArray = rospy.wait_for_message(
                    "/get_gameinfo", UInt8MultiArray, timeout=7
                )
                break
            except:
                rospy.logwarn("Waiting for gameinfo message.")
            rospy.sleep(0.5)
        self.test_navigation()

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

    def test_navigation(self):
        self.response = self.img_switch_mode(0)
        for i in range(0, 3):
            self.navigation_result = self.navigation(
                MissionResquest.MiningArea_1 + i, ""
            )
            for j, target in enumerate(self.gameinfo.data):
                if j < i:
                    continue
                self.response = self.img_switch_mode(target)
                trimer_response: graspsignalResponse = self.trimer(
                    TrimerworkRequest.Grasp, ""
                )
                if trimer_response.res == True:
                    if trimer_response.response == "Successfully Grasp fake":
                        continue
                    else:
                        break

            self.response = self.img_switch_mode(0)
            self.navigation_result = self.navigation(MissionResquest.Station_1 + i, "")
            station = min(max(7, 6 + i), 8)
            self.response = self.img_switch_mode(station)

            if i == 1:
                trimer_response = self.trimer(TrimerworkRequest.PlaceHighly, "")
            else:
                trimer_response = self.trimer(TrimerworkRequest.Place, "")

            self.response = self.img_switch_mode(0)
        self.navigation_result = self.navigation(MissionResquest.Park, "")
        self.response = self.img_switch_mode(0)
        ...


if __name__ == "__main__":
    rospy.init_node("gamecore_node")
    node = gamecore()
