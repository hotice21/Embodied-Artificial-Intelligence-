import random
import time

class RandomAgent:
    def __init__(self, keys):
        self.keys = keys

    def act(self):
        # 返回一个随机动作（按键名）
        return random.choice(self.keys)


class AgentInterface:
    def __init__(self, controller):
        self.controller = controller

    def step(self, action):
        # 控制器应实现 press(key, duration)
        self.controller.press(action, 0.06)
        time.sleep(0.01)
