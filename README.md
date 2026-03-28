# Committor-Based Optimal Control for Stochastic Saddle-Point Systems

This repository contains the simulation code for evaluating committor-based reach-avoid control near stochastic saddle equilibria. It implements both an exact model-based (oracle) controller and a data-enabled tracking procedure (Whitened SVD) that estimates the necessary control parameters directly from passive trajectory observations.

## Overview

Control is often most valuable precisely where autonomous dynamics are most fragile, such as neighborhoods of saddle equilibria where small perturbations redirect trajectories toward qualitatively different futures. This codebase evaluates:

1. **Model-based Optimal Control:** A deterministic, one-sided linear (ReLU-like) law derived from the forward committor limit.
2. **Data-enabled Control:** An open-loop controller that identifies the unstable mode, growth rate, and diffusion from an initial observation period using Whitened SVD.
3. **Closed-loop Adaptive Control:** An online estimator that continually refines parameters using temporal differencing to bypass unknown fixed-point offsets while control is actively applied.

### Evaluated Dynamical Systems
The controllers are evaluated across three distinct stochastic dynamical systems:
* **2D Linear Saddle:** A non-normal linear system.
* **Curved Double Well:** A nonlinear system representing decision-making circuits with a curved stable manifold.
* **Inverted Pendulum:** An undamped nonlinear pendulum representing bipedal locomotion stance phases.

## Installation & Usage

This project requires Python 3.8+ and standard scientific computing libraries. 

1. Clone or download this repository.
2. Install the required dependencies and run the main script:

```bash
pip install -r requirements.txt
python committor_control.py