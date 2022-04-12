"""
Module to wrap functions around the pettingzoo go game

This module create a wrapper around go helper functions to control it from the console.

Authors: Dominik Brockmann, Niklas Schemmer
"""
import os
import time
import keyboard
import numpy as np

class Go():
    """
    A go wrapper.

    This creates a wrapper that interacts with the console to let the human player play against the bot.
    It provides some helpful functions to create a console UI.
    """
    def __init__(self):
        """
        Init the wrapper.
        """
        self.selected = [0, 0]

    def draw(self, board: np.array, active: bool):
        """
        Draw the input column into the console with all possible inputs.

        Parameter board: A matrix containing the whole board with own and opponents coins
        Parameter active: Is the manual player currently playing
        """
        # Clear the console to redraw and add padding
        self.cls()
        print(2*'\n', end='')
        print(10*' ' + 'Press Space to pass')
        print('\n', end='')

        # Iterate through each row and column to create the board layout.
        for row in range(len(board[0])):
            print(10*' ', end='')
            for column in range(len(board)):
                # Draw the selected field and where to place the coin
                if active and column == self.selected[0] and row == self.selected[1]:
                    print(' ^', end='')
                elif active:
                    print(' x' if board[column][row][0] == 1 else ' o' if board[column][row][1] == 1 else ' _', end='')
                else:
                    print(' o' if board[column][row][0] == 1 else ' x' if board[column][row][1] == 1 else ' _', end='')
                if column == len(board) - 1:
                    print('\n', end='')

    def left(self, board: np.array, actions: np.array):
        """
        Go into a lefter column to select whether to place a coin or not.

        Parameter board: A matrix contain the board layout
        Parameter actions: Action mask containing possible actions
        """
        # Make sure player is always in the selectable space
        self.selected[0] = (self.selected[0] - 1) % len(board)

        # Go one more left if action is not possible
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.left(board, actions)
        self.draw(board, True)

    def right(self, board: np.array, actions: np.array):
        """
        Go into a righter column to select whether to place a coin or not.

        Parameter board: A matrix contain the board layout
        Parameter actions: Action mask containing possible actions
        """
        # Make sure player is always in the selectable space
        self.selected[0] = (self.selected[0] + 1) % len(board)

        # Go one more right if action is not possible
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.right(board, actions)
        self.draw(board, True)

    def up(self, board: np.array, actions: np.array):
        """
        Go into an upper row to select whether to place a coin or not.

        Parameter board: A matrix contain the board layout
        Parameter actions: Action mask containing possible actions
        """
        # Make sure player is always in the selectable space
        self.selected[1] = (self.selected[1] - 1) % len(board[0])

        # Go one more up if action is not possible
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.up(board, actions)
        self.draw(board, True)

    def down(self, board: np.array, actions: np.array):
        """
        Go into an lower row to select whether to place a coin or not.

        Parameter board: A matrix contain the board layout
        Parameter actions: Action mask containing possible actions
        """
        # Make sure player is always in the selectable space
        self.selected[1] = (self.selected[1] + 1) % len(board[0])

        # Go one more down if action is not possible
        if (actions[self.selected[0] * len(board) + self.selected[1]] == 0):
            self.down(board, actions)
        self.draw(board, True)

    def win(self):
        """
        The player wins.

        This prints into console that the player won and then waits for him to press space.
        """
        print(2*'\n' + 10*' ' + 'You won!' + '\n\n' + 10*' ' + 'Press Space to continue', end='')

        # Wait for space press
        while not keyboard.is_pressed('space'):
            time.sleep(0.01)

    def lose(self):
        """
        The player lost.

        This prints into console that the player lost and then waits for him to press space.
        """
        print(2*'\n' + 10*' ' + 'You lost.' + '\n\n' + 10*' ' + 'Press Space to continue', end='')

        # Wait for space press
        while not keyboard.is_pressed('space'):
            time.sleep(0.01)

    def cls(self):
        """
        Clear system console for windows and linux.
        """
        os.system('cls' if os.name=='nt' else 'clear')

    def manual_policy(self, observation: np.array, action_mask: np.array):
        """
        This is the manual policy of the player.

        It gets input from keyboard to see which arrow the player pressed.
        Then draws the board with the new selected row.

        Parameter observation: A matrix containing the board state
        Parameter action_mask: An array containing the possible actions
        """
        for _ in range(len(observation[0])):
            # Check if actions is possible, if not go into one of the dimensions where play is possible
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
        block_space = True
        block_enter = True

        while True:
            # Go one left
            if not block_left and keyboard.is_pressed('left'):
                self.left(observation, action_mask)
            # Go one right
            elif not block_right and keyboard.is_pressed('right'):
                self.right(observation, action_mask)
            # Go one up
            elif not block_up and keyboard.is_pressed('up'):
                self.up(observation, action_mask)
            # Go one down
            elif not block_down and keyboard.is_pressed('down'):
                self.down(observation, action_mask)
            # Pass
            elif not block_space and keyboard.is_pressed('space'):
                pass_action = len(observation) * len(observation[0])
                if action_mask[pass_action] == 1:
                    return pass_action
            # Place coin on the current field
            elif not block_enter and keyboard.is_pressed('enter'):
                break

            block_left = keyboard.is_pressed('left')
            block_right = keyboard.is_pressed('right')
            block_up = keyboard.is_pressed('up')
            block_down = keyboard.is_pressed('down')
            block_space = keyboard.is_pressed('space')
            block_enter = keyboard.is_pressed('enter')

            time.sleep(0.001)

        return self.selected[0] * len(observation) + self.selected[1]
