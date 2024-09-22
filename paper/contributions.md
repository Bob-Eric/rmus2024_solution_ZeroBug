## Abstract

This paper presents a fully autonomous robotic system that performs sim-to-real transfer in complex long-horizon tasks involving navigation, recognition, grasping, and stacking in an environment with multiple obstacles.

The key feature of the system is the ability to overcome typical sensing and actuation discrepancies during sim-to-real transfer and to achieve consistent performance without any algorithmic modifications. To accomplish this, a lightweight noise-resistant visual perception system and a nonlinearity-robust servo system are adopted.

We conduct a series of tests in both simulated and real-world environments. The visual perception system achieves the speed of 11 ms per frame due to its lightweight nature, and the servo system achieves sub-centimeter accuracy with the proposed controller. Both exhibit high consistency during sim-to-real transfer.

Our robotic system took first place in the mineral searching task of the Robotic Sim2Real Challenge hosted at ICRA 2024. The simulator is available from the competition committee at https://github.com/AIR-DISCOVER/ICRA2024-Sim2Real-RM, and all code and competition videos can be accessed via our GitHub repository at https://github.com/Bob-Eric/rmus2024_solution_ZeroBug.

## Contributions

the main propose of our paper is to demonstrate that **it is both possible and feasible to achieve consistent performance in long-horizon pick-and-place tasks across simulated and real-world environments without any modification to the algorithm.**

<img src="./figure/key feature.png" width="500">

Our main contributions are summarized as follows:

### 1. High-Consistency Visual Perception System

We design Sequential Motion-Blur Mitigation Strategy for our visual perception system to handle typical sensing errors, delivering consistent performance in both simulated and real-world environments.

### 2.  Nonlinearity-Robust Servo System

We introduce Design Function to our feedback-linearized servo system to mitigate actuation discrepancies, demonstrating strong robustness to nonlinearities.

### 3. Modular System Architecture

The robotic system features a modular architecture, offering a flexible and adaptable platform for researchers.