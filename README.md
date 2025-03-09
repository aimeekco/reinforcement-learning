# Reinforcement Learning

This repository contains my projects for Reinforcement Learning (CSCI181V).

## Projects

- **Q-Learning**: Model-free reinforcement learning algorithm that seeks to find the best action to take given the current state. It does this by learning a Q-value function that estimates the expected utility of taking a given action in a given state.
  - **Qtable**: A table that stores Q-values for each state-action pair.
  - **Qnetwork**: A neural network that approximates the Q-value function.
  - **QNN with buffer**: A Q-network with experience replay buffer to store and reuse past experiences.
- **PPO**: Proximal Policy Optimization is a policy gradient method for reinforcement learning. It uses a surrogate objective function to enable multiple updates per data sample while ensuring the updates do not deviate too much from the previous policy. This helps in improving training stability and performance.
  - **Simple PO**: A basic implementation of Policy Optimization that aims to improve the policy by directly optimizing the expected reward. Updates the policy parameters using gradient ascent on the expected reward.
  - **REINFORCE**: Simple PO with "rewards to-go" for more accurate loss.

