# Student-conference-sumbission
Simulation Framework for a Collision Avoidance Inverse Kinematics model of a 7 degrees of freedom Ultrasound Imaging Robot


In this project, a simulation framework for a collision-avoidance inverse kinematics model for a 7-DoF KUKA
robot is being developed. The robot is intended to be used for ultrasound imaging of patients at the Institute of
Robotics. A Proximal Policy Optimization (PPO) Reinforcement Learning (RL) algorithm is utilized to train the
desired model, using a CoppeliaSim simulation as the environment and PyTorch for the implementation of the PPO
method. Observations from the environment are obtained through a camera attached to the robot’s end effector,
with the captured image data serving as input for the algorithm. While existing inverse kinematics models focus on
guiding robot movements, there is insufficient research on models for preventing unwanted collisions with patients
during ultrasound imaging procedures. To address this challenge and avoid the need for ground-truth data, a RL
approach was chosen. This allows the robot to learn how to avoid collisions in real time, even in the presence of
patient movements.

Libraries used:
gym                             0.17.1
gym-notices                     0.0.8
torch                           2.5.1
torchaudio                      2.0.1+cu118
torchrl                         0.6.0
torchvision                     0.20.1
