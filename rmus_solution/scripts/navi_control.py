#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from math import pi
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from move_base_msgs.msg import MoveBaseActionResult
from rmus_solution.srv import setgoal, setgoalResponse, setgoalRequest
from rmus_solution.srv import keepoutmode, keepoutmodeResponse, keepoutmodeRequest
from rmus_solution.srv import setgoalcoord, setgoalcoordRequest, setgoalcoordResponse
from keep_out_layer.srv import keepOutZone, keepOutZoneRequest, keepOutZoneResponse
import tf2_ros
from tf_conversions import transformations
from enum import IntEnum


class KeepOutMode(IntEnum):
    AddAll = 0
    RemoveAll = 1
    AddByArea = 2
    RemoveByArea = 3


class KeepOutArea(IntEnum):
    MiningArea_0 = 0
    MiningArea_1 = 1
    MiningArea_2 = 2
    TempArea = 3


class PointName(IntEnum):
    Home = 0
    MiningArea0 = 1
    MiningArea1 = 2
    MiningArea2 = 3
    Station_1 = 4
    Station_2 = 5
    Station_3 = 6
    Noticeboard = 7
    Park = 8
    Station_Front = 9
    Station_Back = 10
    MiningArea1_Vertical = 11

    End = 12


class router:
    """
    brief
    ----
    发布move_base目标
    """

    Points = {
        PointName.Home: (0.00, 0.00, 0.00),
        PointName.MiningArea0: (0.82, 0.80, pi),
        PointName.MiningArea1: (1.25, 2.90, 3/4 * pi),
        PointName.MiningArea2: (2.10, 0.60, -pi/3),
        PointName.Station_1: (1.18, 1.91, 0.00),
        PointName.Station_2: (1.18, 1.80, 0.00),
        PointName.Station_3: (1.18, 1.65, 0.00),
        PointName.Noticeboard: (0, 1.6, 0),
        PointName.Park: (3.16, -0.795, 0.00),
        PointName.Station_Front: (1.05, 1.55, pi/12),
        PointName.Station_Back: (2.48, 1.80, pi),
    }

    def __init__(self) -> None:
        self.M_reach_goal = False
        self.getKeepOutAreaPoints()

        # 所有观察点的索引、名称、位置(posi_x,pose_y,yaw)

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

        self.mission = PointName.End

        rospy.sleep(2.0)

        self.__set_goal_service = rospy.Service(
            "/navigation/goal", setgoal, self.setgoalCallback
        )
        self.__keep_out_mode_service = rospy.Service(
            "/keep_out_layer/mode", keepoutmode, self.keepoutmodeCallback
        )
        self.__xju_service_global = rospy.ServiceProxy(
            "/move_base/global_costmap/keep_out_layer/xju_zone", keepOutZone
        )
        self.__xju_service_local = rospy.ServiceProxy(
            "/move_base/local_costmap/keep_out_layer/xju_zone", keepOutZone
        )
        self.__set_goal_coord_service = rospy.Service(
            "/navigation/goal/coord", setgoalcoord, self.setgoalcoordCallback
        )

    def getKeepOutAreaPoints(self):
        mining_area_center = [(-0.15, 0.8), (0.7, 3.4), (2.55, -0.1)]
        temp_area_center = (1.40, 0.15)

        def getPoints(center: "tuple[float,float]", size: "tuple[float,float]"):
            points: list[PointStamped] = []
            for i in range(2):
                for j in range(2):
                    x = center[0] + size[0] * (i - 0.5)
                    y = center[1] + size[1] * (j - 0.5)
                    poseStamped = PointStamped()
                    poseStamped.header.frame_id = "map"
                    poseStamped.point.x = x
                    poseStamped.point.y = y

                    points.append(poseStamped)
            # Swap points[2] and points[3]
            pnt_tmp = points[2]
            points[2] = points[3]
            points[3] = pnt_tmp
            return points

        mining_area_size = (0.30, 0.30)
        temp_area_size = (0.10, 0.6)
        self.KeepOutPoints = {
            KeepOutArea.MiningArea_0: {
                "pose": getPoints(mining_area_center[0], mining_area_size),
                "id": None,
            },
            KeepOutArea.MiningArea_1: {
                "pose": getPoints(mining_area_center[1], mining_area_size),
                "id": None,
            },
            KeepOutArea.MiningArea_2: {
                "pose": getPoints(mining_area_center[2], mining_area_size),
                "id": None,
            },
            KeepOutArea.TempArea: {
                "pose": getPoints(temp_area_center, temp_area_size),
                "id": None,
            },
        }
        return self.KeepOutPoints

    def keepoutmodeCallback(self, req: keepoutmodeRequest):
        mode = req.mode
        area = KeepOutArea(req.area)
        cost = 0
        resp = keepoutmodeResponse()

        if mode == KeepOutMode.AddAll:
            for area in self.KeepOutPoints:
                if self.KeepOutPoints[area]["id"] is None:
                    xju_resp: keepOutZoneResponse = self.__xju_service_global(
                        0, cost, self.KeepOutPoints[area]["pose"], 0
                    )
                    self.__xju_service_local(
                        0, cost, self.KeepOutPoints[area]["pose"], 0
                    )
                    self.KeepOutPoints[area]["id"] = xju_resp.id
                else:
                    rospy.loginfo("KeepOutArea_{} already exists!".format(area))
            message = "Add All KeepOutArea!"
            resp.success = True
            resp.message = message
            rospy.loginfo(message)
        elif mode == KeepOutMode.RemoveAll:
            self.__xju_service_global(2, cost, [], 0)
            self.__xju_service_local(2, cost, [], 0)
            message = "Delete All KeepOutArea!"
            resp.success = True
            resp.message = message
            rospy.loginfo(message)
            for _, temp_area in self.KeepOutPoints.items():
                temp_area["id"] = None

        elif mode == KeepOutMode.AddByArea:
            if self.KeepOutPoints[area]["id"] is None:
                xju_resp: keepOutZoneResponse = self.__xju_service_global(
                    0, cost, self.KeepOutPoints[area]["pose"], 0
                )
                self.__xju_service_local(0, cost, self.KeepOutPoints[area]["pose"], 0)
                self.KeepOutPoints[area]["id"] = xju_resp.id
            else:
                rospy.loginfo("KeepOutArea_{} already exists!".format(area))
            resp.success = True
            message = "Add KeepOutArea_{}!".format(area)
            resp.message = message
            rospy.loginfo(message)
        elif mode == KeepOutMode.RemoveByArea:
            if self.KeepOutPoints[area]["id"] is not None:
                self.__xju_service_global(1, cost, [], self.KeepOutPoints[area]["id"])
                self.KeepOutPoints[area]["id"] = None
            else:
                rospy.loginfo("KeepOutArea_{} does not exist!".format(area))
            message = "Delete KeepOutArea_{}!".format(area)
            resp.success = True
            resp.message = message
            rospy.loginfo(message)
        else:
            rospy.loginfo("Invalid mode!")
            resp.success = False

        return resp

    def MoveBaseResultCallback(self, msg: MoveBaseActionResult):
        if msg.status.status == 3:
            self.M_reach_goal = True

    def pubMovebaseMissionGoal(self, x: float, y: float, yaw: float):
        simple_goal = PoseStamped()
        simple_goal.header.stamp = rospy.Time.now()

        simple_goal.header.frame_id = "map"
        simple_goal.pose.position.x = x
        simple_goal.pose.position.y = y
        simple_goal.pose.position.z = 0.0
        quat = transformations.quaternion_from_euler(0.0, 0.0, yaw)
        simple_goal.pose.orientation.x = quat[0]
        simple_goal.pose.orientation.y = quat[1]
        simple_goal.pose.orientation.z = quat[2]
        simple_goal.pose.orientation.w = quat[3]
        self.goal_puber.publish(simple_goal)

    def setgoalCallback(self, req: setgoalRequest):
        resp = setgoalResponse()
        rospy.loginfo("req: point = {}".format(req.point))

        if 0 <= req.point < PointName.End:
            self.mission = PointName(req.point)
            self.pubMovebaseMissionGoal(
                self.Points[self.mission][0],
                self.Points[self.mission][1],
                self.Points[self.mission][2],
            )
            self.M_reach_goal = False

            r = rospy.Rate(10)
            while not rospy.is_shutdown():
                if self.M_reach_goal:
                    if req.point == PointName.Park:
                        ## send velocity
                        cmd_vel = Twist()
                        cmd_vel.linear.x = 0.1
                        cmd_vel.linear.y = -0.1
                        self.cmd_vel_puber.publish(cmd_vel)
                        rospy.sleep(3)
                        cmd_vel.linear.x = 0.0
                        cmd_vel.linear.y = 0.0
                        self.cmd_vel_puber.publish(cmd_vel)
                    rospy.loginfo("Reach Goal {}!".format(self.Points[self.mission][0]))
                    resp.res = True
                    resp.response = "Accomplish!"
                    self.mission = PointName.End
                    break

                r.sleep()

        else:
            rospy.loginfo("Invalid request!")
            resp.res = False
            resp.response = "Invalid request!"

        return resp

    def setgoalcoordCallback(self, req: setgoalcoordRequest):
        resp = setgoalcoordResponse()
        rospy.loginfo(
            "navigation to x: {:.4f}, y: {:.4f}, yaw: {:.4f}".format(
                req.x, req.y, req.yaw
            )
        )
        self.pubMovebaseMissionGoal(req.x, req.y, req.yaw)
        self.M_reach_goal = False

        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.M_reach_goal:
                rospy.loginfo(
                    "Reach goal x: {:.4f}, y: {:.4f}, yaw: {:.4f}".format(
                        req.x, req.y, req.yaw
                    )
                )
                resp.res = True
                break
            r.sleep()

        return resp


if __name__ == "__main__":
    rospy.init_node("router", anonymous=True)
    rter = router()
    rospy.spin()
