## PPO Agent (PPO-Clip)

This code implements a Proximal Policy Optimization (PPO) agent using the PPO-Clip algorithm for reinforcement learning.

### Overview

PPO is an on-policy RL algorithm designed to address challenges faced by policy gradient methods, such as high variance, long horizons, and continuous action spaces. It offers a balance between sample complexity, computational efficiency, and ease of implementation.

### Components

* **Policy Network:** A neural network that predicts the probability distribution over possible actions given the current state. Implemented as a simple feedforward network with two hidden layers.
* **Get Action Probability:** Function that calculates the action probability for a given state using the policy network.
* **Training:** Function that updates the policy network based on collected experiences. PPO-Clip utilizes a surrogate objective function for this purpose.

### Surrogate Objective Function

The core of PPO-Clip is the surrogate objective function used for training. It aims to minimize the difference between the new and old policies while encouraging actions with high advantage values.

```
L(θ) = E_t [ min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t) ]
```

**Key components:**

* `θ`: Policy network parameters
* `t`: Time step
* `r_t(θ)` (probability ratio): π_θ(a_t|s_t) / π_θ_old(a_t|s_t) - Ratio of new and old policy probabilities for action `a_t` in state `s_t`.
* `A_t`: Advantage function at time step `t`.
* `ε`: Clip ratio (hyperparameter) - Controls update step size.

**Function Breakdown:**

* The first term, `r_t(θ) A_t`, encourages taking actions with higher advantage values.
* The second term, `clip(r_t(θ), 1-ε, 1+ε) A_t`, penalizes large policy updates by clipping the probability ratio within a range defined by `ε`.

This clipped objective helps maintain stability during training by preventing drastic policy changes.

### Training Process

1. Collect a set of trajectories using the current policy.
2. Calculate advantages for each state-action pair in the trajectories.
3. Update the policy network using the clipped surrogate objective function.

PPO-Clip leverages the clipped objective to achieve sample efficiency and stability compared to traditional policy gradient methods.
