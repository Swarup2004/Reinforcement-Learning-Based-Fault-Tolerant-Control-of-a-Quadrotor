# Reinforcement Learning Based Fault-Tolerant Control of a Quadrotor

Implementation of the [MathWorks Challenge Project](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/tree/main/projects/Reinforcement%20Learning%20Based%20Fault%20Tolerant%20Control%20of%20a%20Quadrotor) using MATLAB R2025b.

## Results

| Scenario | Steps Survived | Final Altitude | Status |
|---|---|---|---|
| Healthy (no fault) | 1000/1000 | 1.12m | PASS |
| Motor 1 - 50% LoE | 1000/1000 | 1.18m | PASS |
| Motor 2 - 50% LoE | 1000/1000 | 1.14m | PASS |
| Motor 3 - 50% LoE | 1000/1000 | 1.05m | PASS |
| Motor 4 - 50% LoE | 1000/1000 | 1.13m | PASS |
| Any motor - 70% LoE | < 200 | - | FAIL (physically limited) |

## Algorithm: DDPG (Deep Deterministic Policy Gradient)

Actor network: 9 -> 128 -> 128 -> 4 (tanh)
Critic network: (9->64) + (4->64) -> 128 -> 1
Training time: ~15 minutes on CPU (R2025b, Parallel Computing Toolbox)

## Fault Model
Each motor i has a Loss-of-Effectiveness (LoE) factor lambda in [0,1]:
F_eff_i = lambda_i * kT * omega_i^2
lambda=1 means healthy, lambda=0 means complete failure.

## Reward Function
R = -5(z_err^2) - 1(vz^2) - 0.5(phi^2+theta^2) - 0.01(norm(a)^2) + bonus

## Requirements
- MATLAB R2025b
- Reinforcement Learning Toolbox
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional, speeds up training)

## Project Structure
01_params/    Physical constants (mass, inertia, kT, kQ)
02_plant/     Simulink 6-DOF dynamics model
03_rl_env/    MATLAB RL environment classes
04_agent/     DDPG actor-critic network builder
05_train/     Training scripts and best trained agents
06_results/   Evaluation and visualization scripts

## How to Run
1. run('01_params/quad_params.m')
2. run('05_train/run_training.m')
3. run('06_results/plot_results.m')

## Training Curriculum
Stage 1: Healthy-only environment (500 episodes, ~3 min)
Stage 2: Mixed fault environment, 50% fault episodes (368 episodes, ~4 min)
Best agent: Checkpoint Agent26 from Stage 2 training

## Key Findings
- Single DDPG agent handles all 4 motor faults simultaneously
- No explicit fault detection needed - agent infers fault from state
- Catastrophic forgetting occurs with sequential fine-tuning
- Best results from checkpoint scanning across all training runs
