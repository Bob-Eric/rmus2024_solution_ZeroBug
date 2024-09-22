#  Techniques

## 1. Architecture

The algorithms can be divided into four modules, navigation, task planner, visual perception and servo control.

<img src="https://github.com/Bob-Eric/rmus2024_solution_ZeroBug/figure/architecture.png" alt="architecture" style="zoom:50%;" />

The navigation module comprises `navi_control.py`, which is a ROS node and calls `cartographer` to navigate.

The task planner comprises `play_game.py`, which is a ROS node and contains game logic.

The visual perception module comprises `img_processor.py`, `detect.py` and `simple_digits_classification/`. `img_processor.py` is a ROS node that subscribes sensor data (RGBD image) and publishes pose estimates. `detect.py` contains utility functions. `simple_digits_classification/` contains classifier's weight `model.pth`, training set `templates/` and training program `simple_digits_classify.py`.

The servo module comprises `manipulator.py` and `arm_ctrl.py`. `manipulator.py` is a ROS node that subscribe pose estimates and publish velocity to control the robot. `arm_ctrl.py` contains utility functions of arm control and chassis alignment.

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
          ├── simple_digits_classify.py
          └── templates/
```