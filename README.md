# Particle Filter SLAM

## Overview

This project implements a **Particle Filter–based SLAM system** for a TurtleBot3 operating in a landmark-rich Gazebo environment using ROS2.

The SLAM pipeline is inspired by **FastSLAM**, where:
- a particle filter maintains multiple hypotheses of the robot pose,
- each particle carries its own landmark map estimate,
- probabilistic motion and measurement updates are applied,
- resampling is used to focus on high-likelihood hypotheses.

This project implements all core SLAM components explicitly:
- particle propagation
- measurement likelihood computation
- landmark initialization and update
- resampling
- evaluation against ground truth
   
<p align="center">
  <img src="media/slam_demo.gif" width="600">
</p>

---

## Demo Videos

### SLAM with Default Landmark Configuration
📺 https://www.youtube.com/watch?v=2_0moIchwtc

- Landmarks placed at default locations:
  - Red Landmark: 8.5 -5 0.25
  - Green Landmark: 8.5 5 0.25
  - Yellow Landmark: -11.5 5 0.25
  - Magenta Landmark: -11.5 -5 0.25
  - Cyan Landmark: 0 0 0.25

### SLAM with Randomized Landmark Configuration
📺 https://www.youtube.com/watch?v=G2JGwp-YjP0
- Landmarks placed at default locations:
  - Red Landmark: 2.5 -2 0.25
  - Green Landmark: 4.5 2 0.25
  - Yellow Landmark: -7.5 2 0.25 0 0 0</pose>
  - Magenta Landmark: -4.5 -2 0.25
  - Cyan Landmark: -5.5 4 0.25

## Algorithm Description

### FastSLAM based Particle Filter SLAM
This implementation follows the FastSLAM 1.0 framework proposed by Thrun et al., which factorizes the SLAM posterior into a particle filter over robot poses and independent landmark estimators conditioned on each particle’s trajectory.
Each particle represents a hypothesis of the robot pose:

$$
x_t[i] = (x, y, \theta)
$$

and maintains its own map consisting of landmark means and covariances. Conditioned on a particle’s pose history, landmarks are assumed independent and are estimated using individual EKFs.

### Motion update
Robot motion is obtained from an EKF-based odometry estimate. Each particle is propagated forward using a velocity-based motion model with additive Gaussian noise:

$$
x_t[i] \sim p(x_{t-1}[i], u_t)
$$

Angle normalization is applied to maintain consistency.

### Measurement Update
Particle degeneracy is monitored using the effective sample size:

$$
N_{\text{eff}} = \frac{1}{\sum_i w_i^2}
$$

Systematic resampling is triggered when $N_{\text{eff}} < \alpha N$, with & \alpha = 0.5&, allowing unlikely hypotheses to be discarded while preserving multimodal pose distributions. 

### Covergence and Loop Closure
As the robot revisits previously observed landmarks, inconsistent particle hypotheses receive low likelihood and are eliminated during resampling. This process enables both robot pose and landmark map estimates to converge to a consistent solution, achieving loop closure without maintaining a full joint covariance.



