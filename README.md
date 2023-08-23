# Pong-DQN

An implementation of the Deep Q-Learning (DQN) algorithm to train an agent to play a simple version of the Pong game. The game's objective is to keep the ball from reaching the top boundary by moving a paddle up or down.

## Files Overview

- `pong_env/pong_env.py`: Contains the environment description for the Pong game. It provides functionalities like `reset()`, `step()`, and `get_state()` to interact with the game.
- `dqn_agent.py`: Contains the main DQN agent, the neural network model (`QNetwork`), and a replay buffer (`ReplayBuffer`). The agent uses the QNetwork to approximate the Q-value function and the replay buffer to store and sample experiences.

- `train.py`: Provides a training loop (`dqn()`) to train the DQN agent on the Pong game. The trained agent's model weights are saved to `checkpoint.pth`.

- `play.py`: Allows you to see the trained agent in action. It loads the trained model from `checkpoint.pth` and plays a few games of Pong.

## Setup and Installation

1. Ensure you have Python 3.7+ installed.
2. Install the required packages:
   ```
   pip install numpy torch
   ```

## Training the Agent

To train the DQN agent on the Pong game, run:

```
python train.py
```

Training will loop through episodes of the game, updating the agent's knowledge as it goes. The process will save the agent's model weights to `checkpoint.pth` once it reaches a specified performance threshold.

## Playing the Game with the Trained Agent

After training, or if you have a pre-trained model, you can watch the agent play the game using:

```
python play.py
```

This will load the model weights from `checkpoint.pth` and play a few episodes of Pong, printing out the game state as it progresses.

## Potential Improvements

- Visualization: Incorporate a GUI or visualization library to watch the agent play in real-time.
- Advanced DQN methods: Explore Double DQNs, Dueling DQNs, or Prioritized Experience Replay to enhance the agent's performance.
- Hyperparameter Tuning: Experiment with various hyperparameters to optimize the training process and agent's performance.

## Acknowledgments

This project is inspired by the foundational work on DQN by DeepMind and leverages the power of PyTorch for neural network implementations.
