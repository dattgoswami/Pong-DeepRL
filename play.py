from pong_env.pong_env import PongEnv
from dqn_agent import DQNAgent
import torch
# import time 

env = PongEnv()
agent = DQNAgent(state_size=3, action_size=2, seed=0)

# Load the weights from the trained model
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

def play(agent, env, num_episodes=5):
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward = env.step(action)
            state = next_state
            score += reward
            print(
                f"Episode {i_episode}, Ball Position: {state[0]:.2f}, Paddle Position: {state[1]:.2f}, Reward: {score}")

            if reward != 0:  # game ends when we get a reward (either -1 or 1)
                break

        if score == 1:
            print(f"Episode {i_episode} Ended: Win!")
        else:
            print(f"Episode {i_episode} Ended: Loss!")

play(agent, env)
