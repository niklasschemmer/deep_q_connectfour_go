"""
Module to wrap functions around the pettingzoo connect four game

This module create a wrapper around connect four helper functions to control it from the console.

Authors: Dominik Brockmann, Niklas Schemmer
"""
import os
import time
import keyboard
import numpy as np

class Connect_four():
    """
    A connect four wrapper.

    This creates a wrapper that interacts with the console to let the human player play against the bot.
    It provides some helpful functions to create a console UI.
    """
    def __init__(self):
        """
        Init the wrapper.
        """
        self.selected = 0

    def draw(self, board: np.array, active: bool):
        """
        Draw the input column into the console with all possible inputs.

        Parameter board: A matrix containing the whole board with own and opponents coins
        Parameter active: Is the manual player currently playing
        """
        # Clear the console to redraw and add padding
        self.cls()
        print(3*'\n', end='')

        # Iterate through each row and column to create the board layout.
        for row in range(6):
            print(10*' ', end='')
            for column in range(7):
                # Draw coins in column and where to throw own coin
                if active:
                    print('|' if column == self.selected else ' ', end='')
                    print('x' if board[row][column][0] == 1 else 'o' if board[row][column][1] == 1 else '_' if row == 5 else ' ', end='')
                    print('|' if column == self.selected else ' ', end='')
                else:
                    print(' o ' if board[row][column][0] == 1 else ' x ' if board[row][column][1] == 1 else ' _ ' if row == 5 else '   ', end='')
                if column == 6:
                    print('\n', end='')

    def left(self, board: np.array, actions: np.array):
        """
        Go into a lefter column to select whether to throw a coin or not.

        Parameter board: A matrix contain the board layout
        Parameter actions: Action mask containing possible actions
        """
        # Make sure player is always in the selectable space
        self.selected = (self.selected - 1) % 7

        # Go one more left if action is not possible
        if (actions[self.selected] == 0):
            self.left(board, actions)
        self.draw(board, True)

    def right(self, board: np.array, actions: np.array):
        """
        Go into a righter column to select whether to throw a coin or not.

        Parameter board: A matrix contain the board layout
        Parameter actions: Action mask containing possible actions
        """
        # Make sure player is always in the selectable space
        self.selected = (self.selected + 1) % 7

        # Go one more right if action is not possible
        if (actions[self.selected] == 0):
            self.right(board, actions)
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
        for _ in range(6):
            # Check if actions is possible, if not go one right, if yes stay in selected action
            if action_mask[self.selected] == 0:
                self.selected = (self.selected + 1) % 7
            else:
                break

        self.draw(observation, True)

        block_left = True
        block_right = True
        while True:
            # Go one row left
            if not block_left and keyboard.is_pressed('left'):
                self.left(observation, action_mask)
            # Go one row right
            elif not block_right and keyboard.is_pressed('right'):
                self.right(observation, action_mask)
            # If enter is pressed break loop and draw selected action
            elif keyboard.is_pressed('enter'):
                break

            block_left = keyboard.is_pressed('left')
            block_right = keyboard.is_pressed('right')

            time.sleep(0.001)

        return self.selected