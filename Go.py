import os
import time
import keyboard
import numpy as np

class Go():
    def __init__(self):
        self.selected = [0, 0]

    def draw(self, board, active):
        self.cls()
        print(3*'\n', end='')
        for row in range(len(board[0])):
            print(10*' ', end='')
            for column in range(len(board)):
                if active and column == self.selected[0] and row == self.selected[1]:
                    print(' ^', end='')
                elif active:
                    print(' x' if board[column][row][0] == 1 else ' o' if board[column][row][1] == 1 else ' _', end='')
                else:
                    print(' o' if board[column][row][0] == 1 else ' x' if board[column][row][1] == 1 else ' _', end='')
                if column == len(board) - 1:
                    print('\n', end='')

    def left(self, board, actions):
        self.selected[0] = (self.selected[0] - 1) % len(board)
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.left(board, actions)
        self.draw(board, True)

    def right(self, board, actions):
        self.selected[0] = (self.selected[0] + 1) % len(board)
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.right(board, actions)
        self.draw(board, True)

    def up(self, board, actions):
        self.selected[1] = (self.selected[1] - 1) % len(board[0])
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.up(board, actions)
        self.draw(board, True)

    def down(self, board, actions):
        self.selected[1] = (self.selected[1] + 1) % len(board[0])
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.down(board, actions)
        self.draw(board, True)

    def win(self):
        print(2*'\n' + 10*' ' + 'You won!' + '\n\n' + 10*' ' + 'Press Space to continue', end='')
        while not keyboard.is_pressed('space'):
            time.sleep(0.01)

    def lose(self):
        print(2*'\n' + 10*' ' + 'You lost.' + '\n\n' + 10*' ' + 'Press Space to continue', end='')
        while not keyboard.is_pressed('space'):
            time.sleep(0.01)

    def cls(self):
        os.system('cls' if os.name=='nt' else 'clear')

    def manual_policy(self, observation, action_mask):
        for _ in range(len(observation[0])):
            for _ in range(len(observation)):
                if action_mask[self.selected[0] * len(observation) + self.selected[1]] == 0:
                    self.selected[0] = (self.selected[0] + 1) % len(observation)
            if action_mask[self.selected[0] * len(observation) + self.selected[1]] == 0:
                self.selected[1] = (self.selected[1] + 1) % len(observation[0])

        self.draw(observation, True)

        block_left = True
        block_right = True
        block_up = True
        block_down = True
        block_enter = True
        while True:
            if not block_left and keyboard.is_pressed('left'):
                self.left(observation, action_mask)
            elif not block_right and keyboard.is_pressed('right'):
                self.right(observation, action_mask)
            elif not block_up and keyboard.is_pressed('up'):
                self.up(observation, action_mask)
            elif not block_down and keyboard.is_pressed('down'):
                self.down(observation, action_mask)
            elif keyboard.is_pressed('enter'):
                break

            block_left = keyboard.is_pressed('left')
            block_right = keyboard.is_pressed('right')
            block_up = keyboard.is_pressed('up')
            block_down = keyboard.is_pressed('down')
            block_enter = keyboard.is_pressed('enter')

            time.sleep(0.001)

        return self.selected[0] * len(observation) + self.selected[1]
