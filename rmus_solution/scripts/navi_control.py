#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist, PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from rmus_solution.srv import setgoal, setgoalResponse, setgoalRequest
import tf2_ros
from tf_conversions import transformations
from enum import IntEnum


class MissionResquest(IntEnum):
    Home = 0
    MiningArea_1 = 1
    MiningArea_2 = 2
    MiningArea_3 = 3
    Station_1 = 4
    Station_2 = 5
    Station_3 = 6
    Noticeboard = 7
    Park = 8

    End = 9


class router:
    """
    brief
    ----
    发布move_base目标
    """

    def __init__(self) -> None:
        self.M_reach_goal = False

        # 所有观察点的索引、名称、位置(posi_x,pose_y,yaw)
        self.mission_point = {
            MissionResquest.Home: (0.00, 0.00, 0.00),
            MissionResquest.MiningArea_1: (0.5, 0.5, 3 * pi / 4),
            MissionResquest.MiningArea_2: (1.3, 3.0, 3 * pi / 4),
            MissionResquest.MiningArea_3: (1.96, 0.00768, 0.00),
            MissionResquest.Station_1: (1.18, 1.91, 0.00),
            MissionResquest.Station_2: (1.18, 1.80, 0.00),
            MissionResquest.Station_3: (1.18, 1.65, 0.00),
            MissionResquest.Noticeboard: (0.00, 0.00, pi / 4),
            MissionResquest.Park: (3.16, -0.795, 0.00),
        }

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.cmd_vel_puber = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        while not rospy.is_shutdown():
            try:
                self.tfBuffer.lookup_transform(
                    "map", "base_link", rospy.Time(), timeout=rospy.Duration(2)
                )
                rospy.loginfo("Get tf from map to base_link")
                break
            except:
                rospy.logwarn("Waiting for tf from map to base_link")
            rospy.sleep(0.5)

        rospy.Subscriber(
            "/move_base/result", MoveBaseActionResult, self.MoveBaseResultCallback
        )
        self.goal_puber = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=10
        )
        self.cancel_goal_puber = rospy.Publisher(
            "/move_base/cancel", GoalID, queue_size=10
        )

        self.service = rospy.Service(
            "/set_navigation_goal", setgoal, self.setgoalCallback
        )
        self.mission = MissionResquest.End

    def MoveBaseResultCallback(self, msg: MoveBaseActionResult):
        if msg.status.status == 3:
            self.M_reach_goal = True

    def pubMovebaseMissionGoal(self):
        simple_goal = PoseStamped()
        simple_goal.header.stamp = rospy.Time.now()

        simple_goal.header.frame_id = "map"
        simple_goal.pose.position.x = self.mission_point[self.mission][0]
        simple_goal.pose.position.y = self.mission_point[self.mission][1]
        simple_goal.pose.position.z = 0.0
        quat = transformations.quaternion_from_euler(
            0.0, 0.0, self.mission_point[self.mission][2]
        )
        simple_goal.pose.orientation.x = quat[0]
        simple_goal.pose.orientation.y = quat[1]
        simple_goal.pose.orientation.z = quat[2]
        simple_goal.pose.orientation.w = quat[3]
        self.goal_puber.publish(simple_goal)

    def setgoalCallback(self, req: setgoalRequest):
        resp = setgoalResponse()
        rospy.loginfo(">>>>>>>>>>>>>>>>>>>>>>>>>")
        rospy.loginfo("req: call = {} point = {}".format(req.call, req.point))

        if 0 <= req.point < MissionResquest.End:
            self.mission = MissionResquest(req.point)
            self.pubMovebaseMissionGoal()
            self.M_reach_goal = False

            r = rospy.Rate(10)
            while not rospy.is_shutdown():
                if self.M_reach_goal:
                    rospy.loginfo(
                        "Reach Goal {}!".format(self.mission_point[self.mission][0])
                    )
                    resp.res = True
                    resp.response = "Accomplish!"
                    self.mission = MissionResquest.End
                    break

                r.sleep()

        else:
            rospy.loginfo("Invalid request!")
            resp.res = False
            resp.response = "Invalid request!"

        return resp


if __name__ == "__main__":
    rospy.init_node("router", anonymous=True)
    rospy.loginfo(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    rter = router()
    rospy.spin()
