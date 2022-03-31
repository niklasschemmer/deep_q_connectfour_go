import os
import time
import keyboard
import numpy as np

class Connect_four():
    def __init__(self):
        self.selected = 0

    def draw(self, board, active):
        self.cls()
        print(3*'\n', end='')
        for row in range(6):
            print(10*' ', end='')
            for column in range(7):
                if active:
                    print('|' if column == self.selected else ' ', end='')
                    print('x' if board[row][column][0] == 1 else 'o' if board[row][column][1] == 1 else '_' if row == 5 else ' ', end='')
                    print('|' if column == self.selected else ' ', end='')
                else:
                    print(' o ' if board[row][column][0] == 1 else ' x ' if board[row][column][1] == 1 else ' _ ' if row == 5 else '   ', end='')
                if column == 6:
                    print('\n', end='')

    def left(self, board, actions):
        self.selected = (self.selected - 1) % 7
        if (actions[self.selected] == 0):
            self.left(board, actions)
        self.draw(board, True)

    def right(self, board, actions):
        self.selected = (self.selected + 1) % 7
        if (actions[self.selected] == 0):
            self.right(board, actions)
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
        for _ in range(6):
            if action_mask[self.selected] == 0:
                self.selected = (self.selected + 1) % 7
            else:
                break

        self.draw(observation, True)

        block_left = True
        block_right = True
        block_enter = True
        while True:
            if not block_left and keyboard.is_pressed('left'):
                self.left(observation, action_mask)
            elif not block_right and keyboard.is_pressed('right'):
                self.right(observation, action_mask)
            elif keyboard.is_pressed('enter'):
                break

            block_left = keyboard.is_pressed('left')
            block_right = keyboard.is_pressed('right')
            block_enter = keyboard.is_pressed('enter')

            time.sleep(0.001)

        return self.selected
