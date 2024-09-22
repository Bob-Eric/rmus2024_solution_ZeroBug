# RoboMaster University Sim2real Challenge-ZeroBug Team

## 1. Introduction of us and the competition

This is the official repository of Team ZeroBug, which took first place in 2024 RoboMaster University Sim2real Challenge (RMUS).

<img src="./figure/first_place_ZeroBug.png" alt="certificate" style="zoom:10%;">

RMUS is the mineral searching task of Robotic Sim2real Challenge (refer to http://www.sim2real.net for details), hosted at ICRA 2024.

> The RoboMaster University Sim2Real Challenge (“RMUS”) at its core allows participants to win points by rearranging minerals using fully automated RoboMaster EPs that have been modified officially. The match lasts for five minutes with a sim2real-based format, where robots rearrange minerals based on the information shown on the exchange tags to earn points. The objective of the challenge is to assess how well a program completed on a simulation platform can be operated in real application environments. Participants will be ranked according to their total points won.

Basically, RMUS can be divided into two stages, i.e., simulation and real-world. In simulation stage, a standardized simulator developed on Habitat is provided. Afterwards, each team is required to perform sim-to-real transfer by submitting algorithms to the competition committee via Docker once per week and debugging with test-run logs and videos. After 8 rounds of debugging, the committee conducts three trials in the real robot to evaluate the algorithm of each team and ranks teams according to highest score.

For researchers interested in our paper "Robotic Sim-to-Real Transfer for Long-Horizon Pick-and-Place Tasks in the Robotic Sim2Real Competition", refer to https://github.com/Bob-Eric/rmus2024_solution_ZeroBug/paper/contributions.md for our contributions.

For engineers interested in technical details, refer to https://github.com/Bob-Eric/rmus2024_solution_ZeroBug/techniques.md for explanation of each module.

## 2. Deployment

**Important!** We assume that you have basic knowledge of Docker and ROS. E.g. What are Docker images and containers? How to execute commands inside a Docker container? What are the node, topic and service in ROS? How to launch nodes with launch files? etc. Additionally, it's helpful to take good command of git if you'd like to check PID and open-loop controllers mentioned in our paper, which are used in early stage of the competition but removed later.

To run our algorithms in RMUS simulation environment, you need to follow the installation guide provided by committee (https://github.com/AIR-DISCOVER/ICRA2024-Sim2Real-RM?tab=readme-ov-file#icra-robomaster-sim2real-challenge-client-baseline).

After that, you should have three dockers: the simulator, the client (with a demo project inside), and ros-master.

// todo: write deployment steps here.

For installation:

```bash
git clone https://github.com/Bob-Eric/rmus_solution.git --recursive
rosdep install --from-paths src --ignore-src -r -y
catkin build keep_out_layer
catkin build
```

For running: 

## 3. License and Acknowledgement

The project is released under GPL 2.0 license. During the development of our algorithms, we draw inspirations from these projects:

[ICRA2024-Sim2Real-RM](https://github.com/AIR-DISCOVER/ICRA2024-Sim2Real-RM.git)

[ROS Motion Planning](https://github.com/ai-winter/ros_motion_planning)

[XJU robot project](https://github.com/Mr-Tony921/xju-robot.git)

[ROS Navigation Stack](https://github.com/ros-planning/navigation.git)

## 4. Contact

| Member        | Affiliation                                          | E-mail            |
| ------------- | ---------------------------------------------------- | ----------------- |
| Hongyu Cao    | Tianjin University                                   | 1274653465@qq.com |
| Ming Yang     | Institute of Automation, Chinese Academy of Sciences | 1308592371@qq.com |
| Lixuan Zhao   | Tianjin University                                   | 2196680698@qq.com |
| Chenrui Zhang | Tianjin University                                   | 2594881464@qq.com |

