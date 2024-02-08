import rospy
from nav_msgs.msg import Odometry
import tf


# TODO: 用robot_localization下的ekf_node来融合IMU和编码器数据
def PublishTF(odom: Odometry):
    br = tf.TransformBroadcaster()
    br.sendTransform(
        (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ),
        (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ),
        odom.header.stamp,
        "base_link",
        "odom",
    )
    ...


if __name__ == "__main__":
    rospy.init_node("odom_publisher")
    sub = rospy.Subscriber("/ep/odom", Odometry, PublishTF)
    rospy.spin()
