# Particle Filter SLAM

## Overview

This repository contains a **from-scratch implementation of FastSLAM (FastSLAM 1.0)** for a TurtleBot3 platform in ROS 2. The system performs simultaneous localization and mapping using a particle filter over robot pose and independent EKF landmark estimators per particle, following the formulation introduced by **Thrun, Montemerlo, and colleagues**.

The implementation integrates vision-based landmark observations, EKF-filtered odometry, and particle resampling to estimate both the robot trajectory and a sparse landmark map in a simulated Gazebo environment.

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
- Landmarks placed at randomly chosen locations:
  - Red Landmark: 2.5 -2 0.25
  - Green Landmark: 4.5 2 0.25
  - Yellow Landmark: -7.5 2 0.25 0 0 0</pose>
  - Magenta Landmark: -4.5 -2 0.25
  - Cyan Landmark: -5.5 4 0.25

**Note:** The RQT Plots show the following: (1,1): Robot Position x value against ground truth x value | (1,2): Robot Position y value against ground truth y value | (2,1): SLAM computed Landmark x values | (2,2): SLAM computed landmark y values
 
---

## Algorithm Description

### FastSLAM Factorization
The SLAM posterior is factorized according to the FastSLAM principle:

$$
p(x_{1:t}, m \mid z_{1:t}, u_{1:t}) = p(x_{1:t} \mid z_{1:t}, u_{1:t}) \prod_i p(m_i \mid x_{1:t}, z_{1:t})
$$

This allows:
- a **particle filter** to represent the robot pose distribution, and  
- **independent EKFs** to estimate landmark positions conditioned on each particle’s pose history.

### Particle Representation

Each particle represents a complete SLAM hypothesis and stores:
- robot pose: $(x, y, \theta)$
- particle weight
- a per-landmark map:
  - landmark mean $\mu_i \in \mathbb{R}^2$
  - landmark covariance  $\Sigma_i \in \mathbb{R}^{2\times2}$

Landmarks are indexed by color, providing known data association.

### Motion update
Robot motion is obtained from an EKF-based odometry estimate. Each particle is propagated forward using a velocity-based motion model with additive Gaussian noise:

$$
x_t[i] \sim p(x_{t-1}[i], u_t)
$$

Angle normalization is applied to maintain consistency.

### Measurement Update

Landmark observations are obtained from a vision pipeline and provide **range and bearing** measurements derived from camera intrinsics and detected landmark corners.

For each particle and observed landmark:
- the expected measurement is computed
- the innovation is formed and angle-normalized
- an EKF update is applied to the landmark estimate
- the particle weight is updated using the Gaussian measurement likelihood

New landmarks are initialized using the inverse measurement model when first observed.

Data association is assumed known via color-based landmark identification.

### Resampling
Particle degeneracy is monitored using the effective sample size:

$$
N_{\text{eff}} = \frac{1}{\sum_i w_i^2}
$$

Systematic resampling is triggered when $N_{\text{eff}} < \alpha N$, with $\alpha = 0.5$, allowing unlikely hypotheses to be discarded while preserving multimodal pose distributions. 

### Covergence and Loop Closure
As the robot revisits previously observed landmarks, inconsistent particle hypotheses receive low likelihood and are eliminated during resampling. This process enables both robot pose and landmark map estimates to converge to a consistent solution, achieving loop closure without maintaining a full joint covariance.

---

## Observed Behavior (as shown in videos)

Particles are uniformly initialized across the environment. Early landmark observations result in multiple valid pose hypotheses due to limited information. As the robot moves and observes additional landmarks:
  - particle hypotheses cluster
  - map consistency improves
After repeated landmark observations and loop closure:
  - inconsistent particles are eliminated
  - both robot pose and landmark map converge to a stable solution

These behaviors are visualized in RViz and supported plots.

**Plot 1:** Particles are initialized uniformly accross map to perform Global Localization. RViz also visualizes estimate pose of 3 landmarks that are in Robot cameras viewframe initially. 

<img width="546" height="618" alt="image" src="https://github.com/user-attachments/assets/4d11b04f-5bbc-4afb-a075-04e21241ff28" /> <img width="642" height="385" alt="image" src="https://github.com/user-attachments/assets/26c20d9e-8bab-40b8-9a33-ccb2a60d014e" />
The time-series plots of the robot pose (x and y in the map frame, obtained from odometry) and landmark positions show highly irregular behavior during the initial phase of operation. This is due to the particle filter being initialized with particles uniformly distributed across the map, causing frequent changes in the selected best particle for visualization. Consequently, both robot and landmark estimates exhibit large discontinuities and high variance, reflecting the lack of convergence and the presence of multiple competing pose hypotheses.

**Plot 2:** After driving the robot and accumulating odometry and sensor measurements, the particles form a multimodal belief, with distinct clusters corresponding to multiple plausible robot poses.

<img width="546" height="618" alt="image" src="https://github.com/user-attachments/assets/57a4db0e-3b72-4648-bbf5-bddeda4fab15" /> <img width="648" height="388" alt="image" src="https://github.com/user-attachments/assets/62aeb2ce-0469-4d0e-a0e7-89c1d6e77a9b" />

**Plot 3:** Particles have converged to unimodal belief, with single cluster of plausible robot poses.

<img width="546" height="618" alt="image" src="https://github.com/user-attachments/assets/d3ccde77-dd3b-4a4c-a545-a833e2c3b5d3" /> <img width="645" height="378" alt="image" src="https://github.com/user-attachments/assets/20b2477a-91b3-47f5-b782-6bd5b2c49793" />
The time-series plots of the robot pose (x and y in the map frame) and landmark positions show smooth and stable trajectories after the particle filter has converged. As sensor measurements and odometry are accumulated, the particle set concentrates around a single consistent hypothesis, resulting in reduced variance and continuous estimates of both robot and landmark poses.

**Note:** In the video it can also be observed that the belief sometimes diverges when the robot is driven around for a while without collecting measurement inforation. But as soon as a landmark is seen, the pose converges to a very narrow belief again. 


---

## Global Offset and Error Interpretation

The SLAM pose and map are estimated **in the odometry frame**, not in the Gazebo world frame.

As expected for SLAM:
- the recovered map is correct **up to a rigid-body transformation (SE(2) gauge freedom)**
- absolute alignment with the Gazebo world is unobservable without an external reference

This explains the small, persistent offsets visible in:
- robot pose error plots
- landmark position error plots

Despite this offset, **relative geometry and map consistency are preserved**, which is the defining criterion for successful SLAM.

---

## System Execution

**1. Launch Gazebo Environment with Landmarks**
```
ros2 launch prob_rob_labs turtlebot3_among_landmarks_launch.py
```
**2. Launch EKFNode publishing Odometry**
```
ros2 launch prob_rob_labs EKFNode_launch.py
```
**3. Launch Particle Filter SLAM Module**
```
ros2 launch prob_rob_labs pf_slam_launch.py
```
**4. Launch Ground Truth Publisher**
```
ros2 launch prob_rob_labs lab4_assign1_launch.py
```

---

## References

- Thrun, S., Burgard, W., Fox, D.  
  **FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem.**  
  *Proceedings of the AAAI Conference on Artificial Intelligence*, 2002.  
  https://www.aaai.org/Papers/AAAI/2002/AAAI02-089.pdf

- Thrun, S., Burgard, W., Fox, D.  
  **Probabilistic Robotics.**  
  MIT Press, 2005.  
  https://mitpress.mit.edu/9780262201629/probabilistic-robotics/


