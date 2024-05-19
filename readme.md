# RMUS2024 Solution by ZeroBug
- [RMUS2024](#rmus2024-solution-by-zerobug)
  - [〇、成员信息](#〇成员信息)
  - [一、介绍](#一介绍)
  - [二、安装](#二安装)
  - [三、用法](#三用法)
  - [四、比赛视频](#四比赛视频)
  - [五、改进方向](#五改进方向)
  - [六、License](#六license)
  - [七、Code Base](#七code-base)

## 〇、成员信息

| 成员           | 学校           | 联系方式          |
| :------------- | -------------- | :---------------- |
| 曹鸿钰（队长） | 天津大学       | 1274653465@qq.com |
| 杨铭           | 中国科学院大学 | 1308592371@qq.com |
| 赵礼轩         | 天津大学       | 2196680698@qq.com |
| 张宸睿         | 天津大学       | 2594881464@qq.com |


## 一、介绍

本仓库为`ZeroBug`队伍参加[RMUS 2024](https://github.com/AIR-DISCOVER/ICRA2024-Sim2Real-RM)比赛的代码开源。

该仓库记录了2月底以来、对官方demo project的所有更改，其中，3月2日的v1.0.4-simulation-final是仿真最终版本，5月6日的v1.0.13-sim2real-final是实车环境下的最终版本。

二者的区别主要在于sim2real阶段实车的视频处理更符合成像特性，控制算法参数更为保守，对齐算法逻辑更为鲁棒完善，游戏主逻辑用时更优且具有更少的corner cases。v1.0.4于仿真阶段拿到38分，v1.0.13于实车环境分别拿到38分、32分和24分。

受线上赛限制，官方反馈-调试只进行了8轮，错误和疏漏再所难免，从最后一次比赛反馈中我们得到的改进方向已总结至 [五、改进方向](#五改进方向)。

- ### carto_slam

  该文件夹为`Cartographer`配置文件目录。

  ```
  carto_slam
  ├── launch
  │   ├── cartographer.launch
  │   ├── cartographer_localization.launch
  │   ├── map_writer.sh
  │   └── mapping.launch
  ├── maps
  │   ├── map.pbstream
  │   └── map_real.pbstream
  └── param
      ├── cartographer_2d.lua
      └── cartographer_2d_localization.lua
  ```

- ### navigation

  该文件夹负责导航算法的选择及启动。

  ```
  navigation
  ├── launch
  │   └── navigation.launch
  └── maps
      ├── map.pgm
      └── map.yaml
  ```

- ### rmus_solution

  该文件夹负责比赛相关功能：导航、抓取、识别和主逻辑。

  ```
  rmus_solution
  └── scripts
      ├── arm_ctrl.py
      ├── detect.py
      ├── img_processor.py
      ├── manipulator.py
      ├── navi_control.py
      ├── play_game.py
      └── simple_digits_classification
          ├── model.pth
          └── simple_digits_classify.py
  	...
  ```

  - 导航`(navi_control.py)`

    该文件管理导航点以及禁行区，存储了导航点和禁行区的位置信息，提供了前往导航点和编辑禁行区的`ROS Service`接口。

  - 抓取`(manipulator.py)`

    该文件负责矿石的抓取与叠放，提供`ROS Service`接口。矿石对准算法具有3种：`Open Loop`、`PID`和`State Space`。

    然而`Open Loop`误差过大，`PID`位置控制较为精确，却无法对准角度，`State Space`能对准角度，但无法处理速度死区问题。经多次测试，选择了`PID + State Space`的方案。在角度误差较大时，采取`State Space`方案对准角度，而当角度误差收敛到容许值以内时，切换成`PID`方案对准位置。

  - 识别`(img_processor.py)`

    该文件负责矿石以及兑换站标签的识别，并发布`ROS Topic`话题。识别算法分为负责标签角点检测的前端`(ArUco)`和负责标签分类的后端`(CNN)`。`ArUco`是一种基于二维码的视觉标记系统，已被集成在`OpenCV`中。通过编写自定义字典，`ArUco`算法可以识别场地中`1-6`字样的矿石标签以及`B` 、`O`和` X`字样的兑换站标签，并给出标签的`id`值以及角点。利用角点信息，通过`PNP`解算即可获取矿石中心位姿，在仿真环境中测试，距标签`0.5m`以内时，精度可达`1mm`；由于`ArUco`存在标签误识别和无效框选问题，接入后端`CNN`进行分类或拒识。

    获得标签在相机系`(camera_aligned_depth_to_color_frame_correct)`下的位姿后，通过`tf`树，可以转换为在全局系`(map)`下的坐标，由此获得了标签的全局位姿。当相机未观察到该标签时，以全局位姿进行补间。

  - 主逻辑`(play_game.py)`

    该文件负责比赛主逻辑。

    - 机器人首先对3个矿区添加禁行区（防止导航算法穿越矿区与矿石发生碰撞），并前往`Noticeboard`点位观察`Game Info`，接着删除禁行区，前往`MiningArea0`点位（启动区附近的矿区）进行观测，并选取距机器人最近的矿石进行抓取。

    - 在抓取过程中，若矿石与墙壁距离较近，为避免碰撞，选取与墙壁夹角最小的矿面进行角度对准。在抓取完毕后，判断是否可以放置在兑换站，若是，则放置于兑换站，若否，则放置于启动区前方`I`字型墙前的暂存区，待最后统一抓取。

    
    - 放完第一个矿石后，回到`MiningArea0`点位观察是否有遗留矿石，进行第二次抓取，直到回到`MiningArea0`点位未发现矿石。
    

    - 前往`MiningArea1`点位（`L`型墙壁附近的矿区）和`MiningArea2`点位（停车区附近的矿区）进行相同操作。

    
    - 机器人将暂存区的矿石叠放至兑换站，至此，矿石抓取完毕。
    
    
    - 最后，对3个矿区添加禁行区，并驱使机器人前往`Park`点位进行停车。

- ### ros_motion_planning

  该程序包提取自[ROS Motion Planning](https://github.com/ai-winter/ros_motion_planning)，针对EP机器人配置了导航参数（机器人尺寸、速度和恢复行为等），并在代价地图中添加了禁行区（见[XJU robot project](https://github.com/Mr-Tony921/xju-robot/tree/class_8/src/costmap_plugins)）。

  ```
  ros_motion_planning
  └── src
      ├── sim_env
      │   └── config
      │       ├── amcl_params.yaml
      │       ├── costmap
      │       │   ├── global_costmap_params.yaml
      │       │   ├── global_costmap_plugins.yaml
      │       │   ├── local_costmap_params.yaml
      │       │   └── local_costmap_plugins.yaml
      │       ├── ep_robot
      │       │   └── costmap_common_params_ep_robot.yaml
      │       ├── move_base_params.yaml
      │       └── planner
      │           ├── rpp_planner_params.yaml
      │           └── teb_planner_params.yaml
      └── third_party
          └── map_plugins
              └── keep_out_layer
  ```

- ### rotate_recovery

  该程序包提取自[ROS Navigation Stack](https://github.com/ros-planning/navigation/tree/noetic-devel/rotate_recovery)，对`rotate_recovery`增加了超时功能。

- ### rtab_slam

  该文件夹为`RTAB-Map`配置文件目录。`RTAB-Map`是一套经典的视觉`SLAM`算法，它可支持`2D LIDAR + RGB-D + Odometry + IMU`多传感器融合，并在有限资源情况下精度较高，在仿真环境中`CPU`占用情况和定位精度均好于`Cartographer`。受限于测试反馈数据中地图数据的缺失，在真实环境中使用仿真地图进行定位时，`RTAB-Map`算法无法匹配特征点，定位不可用，后续改用`Cartographer`算法。

- ### [straf_recovery](https://github.com/Bob-Eric/straf_recovery)

  当机器人卡死，触发该恢复行为时，算法在代价地图中寻找最近障碍物点，驱使机器人远离该障碍物点，使机器人脱离卡死状态，恢复正常导航功能。

## 二、安装

```shell
git clone https://github.com/Bob-Eric/rmus_solution.git --recursive
rosdep install --from-paths src --ignore-src -r -y
catkin build keep_out_layer
catkin build
```

## 三、用法

```shell
roslaunch rmus_solution start_game.launch
```

## 四、比赛视频

请见release

## 五、改进方向

- `Cartographer`小概率出现定位漂的现象（5次测试中出现1次），由此可能导致定位标签（全局）位姿估计错误。
  > 考虑到该问题频次不高，可以忽略，或更换建图定位算法。
- 机器人完成一次抓矿动作后，将以`0.3m/s`速度后退`0.7s`，在最左侧矿区的矿石布置在矿区边界时，该过程可能导致于矿区右侧L型墙壁碰撞（e.g. 最终比赛中的24分）；将后退距离改小到10cm左右即可解决该问题，比如以`0.2m/s`速度后退`0.5s`。
  > 后退行为用于观察目标矿石是否仍在视野内以判断抓取动作是否成功，不建议完全取消。

## 六、License

源代码根据GPL 2.0许可发布。

## 七、Code Base

[ICRA2024-Sim2Real-RM](https://github.com/AIR-DISCOVER/ICRA2024-Sim2Real-RM.git)

[ROS Motion Planning](https://github.com/ai-winter/ros_motion_planning)

[XJU robot project](https://github.com/Mr-Tony921/xju-robot.git)

[ROS Navigation Stack](https://github.com/ros-planning/navigation.git)
