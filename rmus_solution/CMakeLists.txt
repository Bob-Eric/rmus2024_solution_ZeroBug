cmake_minimum_required(VERSION 3.0.2)
project(rmus_solution)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  message_generation
  dynamic_reconfigure
)

add_message_files(
  FILES
  MarkerInfo.msg
  MarkerInfoList.msg
)

add_service_files(
  FILES
  setgoal.srv
  switch.srv
  graspsignal.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)


generate_dynamic_reconfigure_options(cfg/manipulater_PID.cfg)

catkin_package(
 CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime dynamic_reconfigure
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)