import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
import tf


# TODO: 用robot_localization下的ekf_node来融合IMU和编码器数据
def OdomTransfer(
    odom: Odometry,
):
    odom_new = odom
    odom_new.pose.covariance = [
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
    ]
    odom_new.twist.covariance = [
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0,
        0,
        0,
        0.01,
    ]
    odom_transfer_pub.publish(odom_new)


def ImuTransfer(
    imu: Imu,
):
    imu_new = imu
    imu_new.angular_velocity_covariance = [
        0.01,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0.01,
    ]
    imu_new.linear_acceleration_covariance = [
        0.01,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0.01,
    ]
    imu_new.orientation_covariance = [
        0.01,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0.01,
    ]
    imu_transfer_pub.publish(imu_new)


def EkfOdomTransfer(ekf_odom: PoseWithCovarianceStamped):
    odom = Odometry()
    odom.header = ekf_odom.header
    odom.header.frame_id = "rtab/odom"
    odom.child_frame_id = "base_link"
    odom.pose = ekf_odom.pose
    ekf_odom_pub.publish(odom)
    ...


if __name__ == "__main__":
    rospy.init_node("odom_publisher")
    rospy.loginfo("odom_publisher node started")
    odom_transfer_pub = rospy.Publisher(
        "/rtabmap/transfer/odom", Odometry, queue_size=10
    )
    imu_transfer_pub = rospy.Publisher("/rtabmap/transfer/imu", Imu, queue_size=10)
    odom_transfer = rospy.Subscriber("/ep/odom", Odometry, OdomTransfer)
    imu_transfer = rospy.Subscriber("/imu/data_raw", Imu, ImuTransfer)
    ekf_odom_pub = rospy.Publisher("/rtabmap/odom", Odometry, queue_size=10)
    ekf_odom_transfer = rospy.Subscriber(
        "/rtabmap/odom_combined", PoseWithCovarianceStamped, EkfOdomTransfer
    )
    rospy.spin()
