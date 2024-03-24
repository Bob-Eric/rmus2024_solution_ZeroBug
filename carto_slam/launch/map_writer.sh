#!/bin/bash
rosservice call /finish_trajectory 0
rosservice call /write_state "filename: '/opt/ep_ws/src/rmus_solution/carto_slam/maps/map_real.pbstream'"
rosrun cartographer_ros cartographer_pbstream_to_ros_map -map_filestem=/opt/ep_ws/src/rmus_solution/navigation/maps/map_real \
         -pbstream_filename=/opt/ep_ws/src/rmus_solution/carto_slam/maps/map_real.pbstream -resolution=0.05
