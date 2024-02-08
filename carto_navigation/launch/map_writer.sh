#!/bin/bash
rosservice call /finish_trajectory 0
rosservice call /write_state "filename: '/opt/ep_ws/src/rmus_solution/carto_navigation/maps/map.pbstream'"
rosrun cartographer_ros cartographer_pbstream_to_ros_map -map_filestem=/opt/ep_ws/src/rmus_solution/navigation/maps \
         -pbstream_filename=/opt/ep_ws/src/rmus_solution/carto_navigation/maps/map.pbstream -resolution=0.05
