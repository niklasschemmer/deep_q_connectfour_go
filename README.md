
# Deep Q-Learning of Classic Games (Connect Four and Go)

This project is a reinforcement learning approach to classical board games. It uses the Deep Q-Learning algorithm and currently includes the games Connect Four and Go. This program was created as part of the "Implementing Artificial Neural Networks with Tensorflow" course at the University of Osnabrück in the winter semester 2021/22 by Niklas Schemmer and Dominik Brockmann.

## Install Requirements

```python
pip install -r requirements.txt
```

Note: On Linux, the package keyboard requires root privileges. Consequently, you have to install the package and start the program using root.

## Start Program

```python
python main.py
```

## CUDA

To decrease training times it is recommended to use a GPU with CUDA. You can find more information about installing CUDA for tensorflow at https://www.tensorflow.org/install/gpu.
