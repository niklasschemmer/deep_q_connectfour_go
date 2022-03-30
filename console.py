import base64
import os
import time
import keyboard
import datetime

from connect_four import Connect_Four
from go import Go

games = {
    Connect_Four: {
        'id': 'connect_four',
        'name': 'Connect Four',
        'description': 'Connect-Four is a tic-tac-toe-like two-player game in which players alternately place pieces on a vertical board 7 columns across and 6 rows high.' # https://mathworld.wolfram.com
    },
    Go: {
        'id': 'go',
        'name': 'Go',
        'description': 'Go is an adversarial game with the objective of surrounding a larger total area of the board with one\'s stones than the opponent. As the game progresses, the players position stones on the board to map out formations and potential territories.' # Matthews, Charles (2004). Teach Yourself Go.
    }
}

parameters = {
    'batch_size': 64,
    'gamma': 0.99,
    'eps_start': 1,
    'eps_end': 0,
    'eps_decay': 0.000001,
    'memory_size': 1000000,
    'epochs': 170000,
    'lr': 0.0001,
    'hidden_units': [100, 75, 50, 25]
}


class Menu():
    def __init__(self, title):
        self.title = title
        self.items = []
        self.selected = 0
        self.exit = False

    def append(self, name, function, attributes, description=''):
        self.items.append(MenuItem(name, function, attributes, description))

    def draw(self):
        self.cls()
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

        for i in range(len(self.items)):
            print(12*' ' + ('{0} ' + self.items[i].name + ' {1}').format('>' if self.selected == i else ' ', '<' if self.selected == i else ' '))

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
        self.selected = (self.selected - 1) % len(self.items)
        self.draw()

    def down(self):
        self.selected = (self.selected + 1) % len(self.items)
        self.draw()

    def select(self):
        keyboard.remove_hotkey('up')
        keyboard.remove_hotkey('down')
        keyboard.remove_hotkey('enter')
        exit = True
        self.cls()
        self.items[self.selected].function(*self.items[self.selected].attributes)

    def start(self):
        keyboard.add_hotkey('up', self.up)
        keyboard.add_hotkey('down', self.down)
        keyboard.add_hotkey('enter', self.select)
        self.draw()
        while not exit:
            pass

    def cls(self):
        os.system('cls' if os.name=='nt' else 'clear')

class MenuItem():
    def __init__(self, name, function, attributes, description):
        self.name = name
        self.function = function
        self.attributes = attributes
        self.description = description


def greeting(fnct):
    menu = Menu('Greetings!')
    menu.append('Next', fnct, [])
    menu.start()

def select_game():
    menu = Menu('Which game do you want to play?')
    for game, info in games.items():
        menu.append(info['name'], select_play, [game], info['description'])
    menu.start()

def select_play(game):
    menu = Menu('Do you want to train or play ' + games[game]['name'] + '?')
    menu.append('Train Agent', train_load, [game], f'Train the agent in {games[game]["name"]} to become an unbeatable AI!')
    menu.append('Play against Agent', play_load, [game], f'Play against the agent in {games[game]["name"]} and get defeated by your own creation!')
    menu.start()

def train_load(game):
    parameters_str = base64.b64encode(str(list(parameters.values())).encode('ascii')).decode('ascii')
    folder_path = 'checkpoints/' + games[game]['id'] + '/' + parameters_str + '/'

    sessions = [f for f in os.listdir(folder_path) if os.path.isdir(folder_path + f)] if os.path.isdir(folder_path) else []
    menu = Menu(f'Do you want to load a checkpoint from a previous training of {games[game]["name"]}?') \
        if len(sessions) > 0 else \
        Menu(f'There are no checkpoints for the current parameters. But you can start a new training of {games[game]["name"]}.')
    menu.append('New Training', train, [game, folder_path + str(int(time.time()))])
    for i in range(len(sessions)):
        menu.append(str(datetime.datetime.fromtimestamp(int(sessions[i]))), train, [game, folder_path + sessions[i]])
    menu.start()

def play_load(game):
    parameters_str = base64.b64encode(str(list(parameters.values())).encode('ascii')).decode('ascii')
    folder_path = 'checkpoints/' + games[game]['id'] + '/' + parameters_str + '/'

    sessions = [f for f in os.listdir(folder_path) if os.path.isdir(folder_path + f)] if os.path.isdir(folder_path) else []
    menu = Menu(f'Do you want to load a checkpoint from a previous training of {games[game]["name"]}?') \
      if len(sessions) > 0 else \
      Menu('There are no checkpoints for the current parameters. To play against a trained agent, please train the agent first. If you already have a trained agent using other parameters, please change these and restart the program.')
    if len(sessions) == 0:
        menu.append('Train Agent first', train_load, [game])
    menu.append('Play against untrained Agent', play, [game, folder_path + str(int(time.time()))])
    menu.append('Exit program to change parameters', quit, [])
    for i in range(len(sessions)):
        menu.append(str(datetime.datetime.fromtimestamp(int(sessions[i]))), play, [game, folder_path + sessions[i]])
    menu.start()

def train_pause(game, save_path):
    print("pause")

def play_pause(game, save_path):
    print("pause")


def train(game, save_path):
    try:
        # start training
        print("start training")
    except KeyboardInterrupt:
        train_pause(game, save_path)
    except Exception:
        print("Error")
        select_game()

def play(game, save_path):
    try:
        # start playing
        print("start playing")
    except KeyboardInterrupt:
        play_pause(game, save_path)
    except Exception:
        print("Error")
        select_game()

greeting(select_game)
