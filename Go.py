import os
import time
import keyboard
import numpy as np

class Go():
    def __init__(self):
        pass

    def draw(self, board, active):
        pass

    def left(self, board, actions):
        pass

    def right(self, board, actions):
        pass

    def win(self):
        pass

    def lose(self):
        pass

    def cls(self):
        os.system('cls' if os.name=='nt' else 'clear')

    def manual_policy(self, observation, action_mask):

        return 0
