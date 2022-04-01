
# Deep Q-Learning of Classic Games (Connect Four and Go)

This project is a reinforcement learning approach to classical board games. It uses the Deep Q-Learning algorithm and currently includes the games Connect Four and Go. It was created as part of the "Implementing Artificial Neural Networks with Tensorflow" course at the University of Osnabrück in the winter semester 2021/22 by Niklas Schemmer and Dominik Brockmann.

## Install Requirements

```python
pip install -r requirements.txt
```

Note: On Linux, the package keyboard requires root privileges. Consequently, you have to install the package and start the program using root.

## Start Program

```python
python main.py
```

## Usage

After starting a console menu guides through different options. First, you can select a game. Then if you want to train an agent in this game, play against an already trained agent or plot the accuracy of a previous training obtained by the quality measure.

The program automatically saves the training status every 1000 episodes. Along with the data of the trained network, the number of actions and episodes and the accuracy is saved. These checkpoints are can be found in the folder checkpoints/{game}/{base64(parameters)}/{start_time}. This path uses a base64 representation of the parameters to automatically match checkpoints with the parameters that were used.

## CUDA

To decrease training times it is recommended to use a GPU with CUDA. You can find more information about installing CUDA for tensorflow at https://www.tensorflow.org/install/gpu.
