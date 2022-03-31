"""
This module contains the console menu of the program and starts on selection the corresponding functions.

At the beginning of the file, the parameters can be changed and additional games can be added. In the menu you can select a game, if you want to train/play/plot and load saves from previous trainings. Be aware that only saves with the same parameters can be loaded.

Authors: Dominik Brockmann, Niklas Schemmer
"""
import base64
import os
import time
import keyboard
import datetime
from DeepQTraining import run_training
from ReplayMemory import ReplayMemory
from Model import DenseModel
import tensorflow as tf
import numpy as np

from ConnectFour import Connect_four
from Go import Go

from Plotting import plot_accuracy
from Testing import test_game

from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import go_v5

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
The configuration dictionary of all games.

Attribute id: Unique identifier
Attribute name: Name of the game to show in the menu
Attribute description: A description of the game to show in the menu
Attribute game_import: The module that defines the game
Attribute play: A class that implements the manual play functionality of the game
"""
games = {
    'connect_four': {
        'id': 'connect_four',
        'name': 'Connect Four',
        'description': 'Connect-Four is a tic-tac-toe-like two-player game in which players alternately place pieces on a vertical board 7 columns across and 6 rows high.', # https://mathworld.wolfram.com
        'game_import': connect_four_v3,
        'play': Connect_four
    },
    'go': {
        'id': 'go',
        'name': 'Go',
        'description': 'Go is an adversarial game with the objective of surrounding a larger total area of the board with one\'s stones than the opponent. As the game progresses, the players position stones on the board to map out formations and potential territories.', # Matthews, Charles (2004). Teach Yourself Go.
        'game_import': go_v5,
        'play': Go
    }
}

"""
The parameters of the training process.

Attribute batch_size: The batch size used by the Replay Memory
Attribute gamma:
Attribute eps_start: The starting epoch of the Epsilon Greedy method
Attribute eps_end: The ending epoch of the Epsilon Greedy method
Attribute eps_decy: The decay of the epsilon variable Epsilon Greedy method
Attribute memory_size: The size of the Replay Memory
Attribute epochs: The goal number of epochs; end if trained for so many epochs
Attribute learning_rate: The learning rate of the used Optimizer
Attribute hidden_units: An Array with each number representing the number of neurons in one hidden layer
"""
parameters = {
    'batch_size': 128,
    'gamma': 0.99,
    'eps_start': 1,
    'eps_end': 0,
    'eps_decay': 0.000001,
    'memory_size': 1000000,
    'epochs': 170000,
    'learning_rate': 0.0001,
    'hidden_units': [100, 75, 50, 25]
}


class Menu():
    """
    A menu holding multiple selection options.
    """
    def __init__(self, title):
        """
        Initialize the menu.

        Parameter title: The title that is shown on top of the menu
        """
        self.title = title
        self.items = []
        self.selected = 0

    def append(self, name, function, attributes, description=''):
        """
        Append a selection element to the list of the menu.

        Parameter name: The name of the element
        Parameter function: The callback that is executed when the user selects this element
        Parameter attributes: The attributes of the callback
        Parameter description: An optional description of the element that is shown when selected
        """
        self.items.append(MenuItem(name, function, attributes, description))

    def draw(self):
        """
        Draw the menu.
        """
        # clear window
        self.cls()

        # print title
        print(2*'\n' + 10*' ', end='')
        size = 100
        c = 0
        for word in self.title.split(' '):
            if c + len(word) > size:
                print('\n' + 10*' ', end='')
                c = 0
            c += len(word) + 1
            print(word + ' ', end='')
        print('\n')

        # print menu list
        for i in range(len(self.items)):
            print(12*' ' + ('{0} ' + self.items[i].name + ' {1}').format('>' if self.selected == i else ' ', '<' if self.selected == i else ' '))

        # print description box
        if self.items[self.selected].description != '':
            size = 48
            print(2*'\n')
            print(12*' ' + '+' + (size+2)*'-' + '+\n' + 12*' ' + '| ', end='')
            c = 0
            for word in self.items[self.selected].description.split(' '):
                if c + len(word) > size:
                    print((size-c+1)*' ' + '|\n' + 12*' ' + '| ', end='')
                    c = 0
                c += len(word) + 1
                print(word + ' ', end='')
            print((size-c+1)*' ' + '|\n' + 12*' ' + '+' + (size+2)*'-' + '+')

    def up(self):
        """
        Select the element above the currently selected. If it is the element on top, select the element at the bottom.
        """
        self.selected = (self.selected - 1) % len(self.items)
        self.draw()

    def down(self):
        """
        Select the element under the currently selected. If it is the element at the b ottom, select the element at top.
        """
        self.selected = (self.selected + 1) % len(self.items)
        self.draw()

    def select(self):
        """
        Execute the function of the currently selected element.
        """
        keyboard.remove_hotkey('up')
        keyboard.remove_hotkey('down')
        keyboard.remove_hotkey('enter')
        self.cls()
        self.items[self.selected].function(*self.items[self.selected].attributes)

    def start(self):
        """
        Start the menu: Add the required hotkeys and draw it.
        """
        keyboard.add_hotkey('up', self.up)
        keyboard.add_hotkey('down', self.down)
        keyboard.add_hotkey('enter', self.select)
        self.draw()

    def cls(self):
        """
        Clear system console for windows and linux.
        """
        os.system('cls' if os.name=='nt' else 'clear')

class MenuItem():
    """
    A menu item representing one element in a menu.
    """
    def __init__(self, name, function, attributes, description):
        """
        Initialize the Menu Item.

        Parameter name: The name of the element
        Parameter function: The callback that is executed when the user selects this element
        Parameter attributes: The attributes of the callback
        Parameter description: An optional description of the element that is shown when selected
        """
        self.name = name
        self.function = function
        self.attributes = attributes
        self.description = description


def greeting(fnct):
    """
    Display the greetings menu.

    Parameter fnct: Callback executed when pressing Enter
    """

    menu = Menu(f'Welcome to this reinforcement learning approach to classical board games. It uses the Deep Q-Learning algorithm and currently includes the games {" and ".join([games[game]["name"] for game in games])}. This program was created as part of the "Implementing Artificial Neural Networks with Tensorflow" course at the University of Osnabrück in the winter semester 2021/22 by Niklas Schemmer and Dominik Brockmann.')
    menu.append('Press Enter to continue', fnct, [])
    menu.start()

def select_game():
    """
    Display the select game menu in which the user selects the game to train on/play.
    """

    menu = Menu('Which game do you want to play?')
    for game, info in games.items():
        menu.append(info['name'], select_play, [game], info['description'])
    menu.start()

def select_play(game):
    """
    Display the select play menu in which the user selects if to train/play/plot accuracy.

    Parameter game: The game configuration holding name, description, etc.
    """

    menu = Menu('Do you want to train or play ' + games[game]['name'] + '?')
    menu.append('Train Agent', train_load, [game], f'Train the agent in {games[game]["name"]} to become an unbeatable AI!')
    menu.append('Play against Agent', play_load, [game], f'Play against the agent in {games[game]["name"]} and get defeated by your own creation!')
    menu.append('Plot the accuracy of a saved training', plot_load, [game], f'Plot the accuracy of the quality measurement of a previous training. Note: you can only load saved trainings that used the same parameters that you defined.')
    menu.append('<- Back', select_game, [])
    menu.start()

def train_load(game):
    """
    The menu to load a previous training.

    Parameter game: The game configuration holding name, description, etc.
    """

    # build save path from parameters
    parameters_str = base64.b64encode(str(list(parameters.values())).encode('ascii')).decode('ascii')
    folder_path = 'checkpoints/' + games[game]['id'] + '/' + parameters_str + '/'

    # build menu from found savings
    sessions = [f for f in os.listdir(folder_path) if os.path.isdir(folder_path + f)] if os.path.isdir(folder_path) else []
    menu = Menu(f'Do you want to load a checkpoint from a previous training of {games[game]["name"]}?') \
        if len(sessions) > 0 else \
        Menu(f'There are no checkpoints for the current parameters. But you can start a new training of {games[game]["name"]}.')
    menu.append('New Training', train, [game, folder_path + str(int(time.time()))])
    for i in range(len(sessions)):
        menu.append(str(datetime.datetime.fromtimestamp(int(sessions[i]))), train, [game, folder_path + sessions[i]])
    menu.append('<- Back', select_play, [game])
    menu.start()

def play_load(game):
    """
    The menu to load a previous training to play on.

    Parameter game: The game configuration holding name, description, etc.
    """

    # build save path from parameters
    parameters_str = base64.b64encode(str(list(parameters.values())).encode('ascii')).decode('ascii')
    folder_path = 'checkpoints/' + games[game]['id'] + '/' + parameters_str + '/'

    # build menu from found savings
    sessions = [f for f in os.listdir(folder_path) if os.path.isdir(folder_path + f)] if os.path.isdir(folder_path) else []
    menu = Menu(f'Do you want to load a checkpoint from a previous training of {games[game]["name"]}?') \
      if len(sessions) > 0 else \
      Menu('There are no checkpoints for the current parameters. To play against a trained agent, please train the agent first. If you already have a trained agent using other parameters, please change these and restart the program.')
    if len(sessions) == 0:
        menu.append('Train Agent first', train_load, [game])
    menu.append('Play against untrained Agent', play, [game, folder_path + str(int(time.time()))])
    for i in range(len(sessions)):
        menu.append(str(datetime.datetime.fromtimestamp(int(sessions[i]))), play, [game, folder_path + sessions[i]])
    menu.append('<- Back', select_play, [game])
    menu.start()

def plot_load(game):
    """
    The menu to load a previous training to plot the accuracy.

    Parameter game: The game configuration holding name, description, etc.
    """

    # build save path from parameters
    parameters_str = base64.b64encode(str(list(parameters.values())).encode('ascii')).decode('ascii')
    folder_path = 'checkpoints/' + games[game]['id'] + '/' + parameters_str + '/'

    # build menu from found savings
    sessions = [f for f in os.listdir(folder_path) if os.path.isdir(folder_path + f)] if os.path.isdir(folder_path) else []
    menu = Menu(f'Which training of {games[game]["name"]} do you want to plot?') \
      if len(sessions) > 0 else \
      Menu('There are no checkpoints for the current parameters. Please train the agent first. If you already have a trained agent using other parameters, please change these and restart the program.')
    if len(sessions) == 0:
        menu.append('Train Agent first', train_load, [game])
    for i in range(len(sessions)):
        menu.append(str(datetime.datetime.fromtimestamp(int(sessions[i]))), start_plot, [game, folder_path + sessions[i]])
    menu.append('<- Back', select_play, [game])
    menu.start()

def start_plot(game, save_path):
    """
    Start the plotting.

    Parameter game: The game configuration holding name, description, etc.
    Parameter save_path: The location of the save
    """
    plot_accuracy(save_path)
    plot_load(game)

def play_pause(game, play_again):
    """
    Pause menu between playing.

    Parameter game: The game configuration holding name, description, etc.
    Parameter play_again: The callback executed when playing again
    """
    menu = Menu(f'Do you want to play {games[game]["name"]} again or return to the menu?')
    menu.append('Play again', play_again, [])
    menu.append('<- Back', play_load, [game])
    menu.start()


def train(game, save_path):
    """
    Start the training.

    Parameter game: The game configuration holding name, description, etc.
    Parameter save_path: The location of the save
    """
    try:
        # initialize the environment
        env = games[game]['game_import'].env()
        env.reset()

        # define action and observation space
        action_space = env.action_space(env.agents[0]).n
        observation_space = np.prod(env.observation_space(env.agents[0])['observation'].shape)

        # create the networks
        policy_net = DenseModel(parameters['hidden_units'], action_space)
        target_net = DenseModel(parameters['hidden_units'], action_space)

        # initialize the replay memory
        memory = ReplayMemory(parameters['memory_size'])

        # create the checkpoint and checkpoint manager
        # the network, step and epoch is saved
        checkpoint = tf.train.Checkpoint(model=policy_net, step=tf.Variable(0), epoch=tf.Variable(0))
        cp_manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=3)
        # load checkpoint if it exists
        if cp_manager.latest_checkpoint:
            checkpoint.restore(cp_manager.latest_checkpoint)

        # start the training process
        run_training(env, parameters, policy_net, target_net, checkpoint, cp_manager, save_path, memory, action_space, observation_space)
    except Exception as e:
        print(e)
        raise e

def play(game, save_path):
    """
    Start playing.

    Parameter game: The game configuration holding name, description, etc.
    Parameter save_path: The location of the save
    """
    try:
        # initialize the environment
        env = games[game]['game_import'].env()
        env.reset()

        # define action and observation space
        action_space = env.action_space(env.agents[0]).n
        observation_space = np.prod(env.observation_space(env.agents[0])['observation'].shape)

        # create the network
        policy_net = DenseModel(parameters['hidden_units'], action_space)

        # create the checkpoint and checkpoint manager
        # the network, step and epoch is saved
        checkpoint = tf.train.Checkpoint(model=policy_net, step=tf.Variable(0), epoch=tf.Variable(0))
        cp_manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=3)
        # load checkpoint if it exists
        if cp_manager.latest_checkpoint:
            checkpoint.restore(cp_manager.latest_checkpoint)

        # start the play loop
        play_loop(env, game, policy_net, observation_space)

    except Exception as e:
        print(e)
        raise e

def play_loop(env, game, policy_net, observation_space):
    """
    Start the play loop, showing the pause menu after every game.

    Parameter game: The game configuration holding name, description, etc.
    Parameter policy_net: The network to play against
    Parameter observation_space: The dimensions of the observation space of the game
    """
    test_game(env, games[game]['play'](), policy_net, observation_space)
    play_pause(game, lambda: play_loop(env, game, policy_net, observation_space))

# show the greeting
greeting(select_game)
keyboard.wait()
