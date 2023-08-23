import numpy as np

class PongEnv:
    def __init__(self):
        self.ball_y = np.random.random()
        self.paddle_y = 0.5
        self.ball_velocity = np.random.choice([-1, 1]) * 0.05

    def reset(self):
        self.__init__()
        return self.get_state()

    def get_state(self):
        return np.array([self.ball_y, self.paddle_y, self.ball_velocity])

    def step(self, action):
        if action == 0:  # move up
            self.paddle_y = min(1, self.paddle_y + 0.05)
        elif action == 1:  # move down
            self.paddle_y = max(0, self.paddle_y - 0.05)

        self.ball_y += self.ball_velocity

        if self.ball_y <= 0 or self.ball_y >= 1:
            self.ball_velocity = -self.ball_velocity

        reward = 0
        if self.ball_y > 0.95:
            if self.paddle_y > 0.85:
                reward = 1
            else:
                reward = -1
            self.ball_velocity = -self.ball_velocity

        return self.get_state(), reward
