[//]: # (Image References)

[image1]: score_vs_episodes.png "Performance"

# Project 1: Navigation

## Learning Algorithm

The training procedure leverages the DQN architecture in conjunction with a fixed target network and experience replay. At a high level, the training loop involves:

1. Environment interaction: the agent interacts with the environment using an epsilon-greedy policy with respect to the current action-value function (encoded by the Q-network) and stores the outcome of the interactions in memory (the experience replay buffer).

2. Network update: Every few interactions (controlled by the hyperparameter `UPDATE_EVERY` in `dqn_agent.py`) the agent draws experiences from memory (uniformly at random) and trains the DQN using a fixed (or slow-moving, in this case) target network.

### Architecture

The function approximator network employed in this solution is a simple multi-layer perceptron (MLP). The implementation provides options for the activation types, as well as the number and size of the hidden layers. The default parameters are:
- `sizes = [state_space_size, 64, 64, action_space_size]`, which defines a MLP with two hidden layers with 64 units each (number of input and output units are matched to the environment).
- `activation = torch.nn.ReLU()`, which sets the activation functions between layers as rectified linear units (ReLU).

### Agent

The agent object contains both the main DQN (`agent.qnetwork_local`), the target DQN (`agent.qnetwork_target`), and the memory (the experience replay buffer, `agent.memory`). The agent's hyperparameters control the two steps of the training procedure outlined above.
- `BUFFER_SIZE = 1e5` controls the size of the experience replay buffer
- `BATCH_SIZE = 64` controls the number of experiences drawn from the replay buffer whenever a DQN update is triggered
- `GAMMA = .99` is the discont factor used when computing returns (between 0 and 1)
- `TAU = 1e-3` controls the rate at which the target DQN is updated (between 0 and 1: 0 means the target DQN is never updated, 1 means the target network is always equal to the main network)
- `LR = 5e-4` is the learning rate used when updating the weights of the local network
- `UPDATE_EVERY = 4` controls how frequently to trigger a network update step (1 means the network weights are updated after every interaction with the environment, 4 means the weights are updated every 4 interactions, etc)

### Training

- `n_episodes = 2000` controls the total number of episodes used for training
- `max_t = 1000` controls the maximum duration of an episode
- `eps_start = 1.0`, `eps_end=0.01` and `eps_decay=0.995` control the epsilon parameter of the epsilon-greedy policy. Specifically, the epsilon parameters is set to `eps_start` at the beggining of training and decays by `eps_decay` every episode until it reaches `eps_end`, after which it stays constant.

## Learning Performance

Bellow is the result of one training run, showing the average total rewards over 100 episodes over the course of 2000 episodes. In this instance, the agent needed 527 episodes to solve the environment.

![Learning Performance][image1]

## Ideas for Future Work

While the approach employed here was fairly successful, it represents a fairly straightforward implementation that can be improved in a number of ways. While these improvements might have a limited impact in solving this simple environment, they might prove much more important when tackling more complex environments (for example, learning using pixel-level representation of states):
- Double DQN
- Prioritized Experience Replay
- Dueling DQN
- Further hyper-parameter tuning