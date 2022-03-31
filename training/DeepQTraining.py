"""
This module contains the whole training procedure of the deep q learning process.

This contains the deep q learning procedure.
Part of it are the DeepQ Agent that uses the epsilon greedy strategy to find a tradeoff between exploration and exploitation.
The quality is measured against a random policy.

Authors: Dominik Brockmann, Niklas Schemmer
"""
import os
import time
import csv
from time import sleep
import math
import random
import base64
import datetime
from datetime import timedelta
import numpy as np
from statistics import mean
from collections import namedtuple
import tensorflow as tf

from ReplayMemory import ReplayMemory

class EpsilonGreedyStrategy():
    """
    The strategy to calculate the exploration rate of the agent.
    """
    def __init__(self, start: int, end: int, decay: float):
        """
        Initialize the EpsilonGreedyStrategy.

        Parameter start: start of the exponential decay
        Parameter end: end of the exponential decy
        Parameter decay: The decay rate of the exploration
        """
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step: int):
        """
        Calculates the exploration rate of the strategy.

        Parameter current_step: The current step of the training
        """
        # Calculate exponential decay of the rate
        return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)

class DeepQ_Agent():
    """
    The agent used for select random or calculate actions from the policy.
    """
    def __init__(self, strategy: object, num_actions: int):
        """
        Initialize the deep q agent.

        Parameter strategy: The strategy of the exploration rate
        Parameter num_actions: The amount of actions that can be chosen
        """
        self.strategy = strategy
        self.num_actions = num_actions

    def select_greedy_action(self, observation: np.array, policy_net, current_step):
        """
        Select an action from the exploration rate telling us to go exploring or do what the model wants.

        Parameter observation: The current observation made from the enviroment
        Parameter policy_net: The policy controlled by the trained agent
        Parameter current_step: The current cycle of the training
        """
        # Get the current exploration rate from our strategy
        rate = self.strategy.get_exploration_rate(current_step)

        # Choose to explore or to exploit
        if rate > random.random():
            return random.randrange(self.num_actions), rate, True
        else:
            return np.argmax(policy_net(np.atleast_2d(observation).astype('float32'))), rate, False

    def select_random_policy(self, action_mask):
        """
        Select an action from the random policy.

        Parameter action_mask: The mask that tells us which actions are possible
        """
        # Get indices of possible actions
        allowed_actions = np.where(np.array(action_mask) == 1)[0]

        # Return zero if no actions possible
        if len(allowed_actions) == 0:
            return 0
        # Return random index if possible actions exist
        else:
            return np.random.choice(allowed_actions)

def test_against_random_policy(env, model: object, policy_net: object, observation_space: np.array, num_games: int =100):
    """
    Test the trained policy against a random policy to get the win rate as quality measure.

    This tests our trained network.
    The more games to be played, the more accurate is our quality measure.

    Parameter model: The deepQ agent used for training the network
    Parameter policy_net: The trained policy
    Parameter observation_space: A matrix containing the space of observations a player can make
    Parameter num_games: Amount of games to be played
    """
    # The amount of wins our trained policy makes
    policy_wins = 0

    for i in range(num_games):
        env.reset()

        # Select the agent that will be played by the trained policy
        play_agent = env.agents[0]

        for agent in env.agent_iter():
            observation, reward, done, info = env.last()

            # Reshaping observation into flat observation
            observation['observation'] = tf.reshape(observation['observation'], shape=(observation_space))
            action = None

            if agent==play_agent:
                # Getting the amount of wins by adding up the reward which is either 1 or -1 at the end of the game
                policy_wins += reward

                # Exploit maximum Q value from the net
                action = np.argmax(policy_net(np.atleast_2d(observation['observation']).astype('float32')))
            else:
                # Select random action
                action = model.select_random_policy(observation['action_mask'])
            if done == False:
                env.step(action)
            else:
                env.step(None)

    # Calculate win rate from the amount of won games
    return policy_wins/num_games

def run_training(env: object, training_parameters: dict, policy_net: object, target_net: object, checkpoint: tf.train.Checkpoint, cp_manager: tf.train.CheckpointManager, save_path: str, memory: ReplayMemory, action_space: int, observation_space: int):
    """
    This starts the Q training with the given parameters.

    The trained policy, checkpoint manager, memory buffer and action/observation spaces are passed as arguments.
    The training then starts on the GPU and regularily prints the results of the training.

    Parameter env: The game enviroment
    Parameter training_parameters: The parameters configured by the user
    Parameter policy_net: The given policy to be trained
    Parameter target_net: The target policy used to compare against the policy_net. It contains the same weights as policy_net
    Parameter checkpoint: The last checkpoint of the training to start it from again
    Parameter cp_manager: The checkpoint manager of the training, which handles saving of new checkpoints
    Parameter memory: The replay buffer, used to replay game situations to retrain the policy on it
    Parameter action_space: The amount of actions that can be played in the game
    Parameter observation_space: The dimensions of the observation space of the game
    """
    with tf.device("GPU"):
        # Using adam optimizer for the training, others can be chosen as well
        optimizer = tf.keras.optimizers.Adam(
                        learning_rate=training_parameters['learning_rate'],
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07)

        # Setting up the greedy strategy and the deepQ Agent
        strategy = EpsilonGreedyStrategy(training_parameters['eps_start'], training_parameters['eps_end'], training_parameters['eps_decay'])
        model = DeepQ_Agent(strategy, action_space)

        # The layout of the experience
        Experience = namedtuple('Experience', ['observations','actions', 'rewards', 'next_observations', 'dones'])

        # Syncronize weights of policy and target net
        policy_net.copy_weights_to(target_net)

        # Passed time for training
        start_time = time.time()

        # Iterate through the training epochs
        while int(checkpoint.epoch) < training_parameters['epochs']:
            env.reset()
            losses = []

            # Save the last observations of the enviroment to save them as an experience later
            last_state = {}
            for agent in env.agents:
                last_state[agent] = {
                    'observation': None,
                    'action': None,
                    'reward': 0
                }

            for agent in env.agent_iter():
                checkpoint.step.assign_add(1)

                observation, reward, done, _ = env.last()

                # Increase the reward to get a nicer space of Q values later
                reward *= 100

                # Flatten observations
                observation['observation'] = tf.reshape(observation['observation'], shape=(observation_space))

                # Add up the reward for a game
                last_state[agent]['reward'] += reward

                # Check if this is the first training cycle
                if last_state[agent]['action'] != None:
                    # Create a new experience with the current step to the memory
                    memory.add(Experience(last_state[agent]['observation'], last_state[agent]['action'], observation['observation'], reward, done))

                    # If the amount of batch size is reached in the replay memory, start sampling
                    if memory.has_batch_length(training_parameters['batch_size']):
                        experiences = memory.random_sample(training_parameters['batch_size'])
                        batch = Experience(*zip(*experiences))

                        observations, actions, rewards, next_observations, dones = np.asarray(batch[0]),np.asarray(batch[1]),np.asarray(batch[3]),np.asarray(batch[2]),np.asarray(batch[4])

                        # Get the maximum Q value for the next step predicted by the target network
                        q_s_a_prime = np.max(target_net(np.atleast_2d(next_observations).astype('float32')), axis = 1)
                        # In those steps where the game was done, add
                        q_s_a_target = np.where(dones, rewards, rewards+training_parameters['gamma']*q_s_a_prime)
                        # Convert Q values to float32
                        q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')

                        # Sum the Q values from actions predicted by the networks
                        q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(observations).astype('float32')) * tf.one_hot(actions, action_space), axis=1)
                        # Calculate loss from the Q values in the actual step leading to high Q values in the next step, to maximise future rewards
                        loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

                        with tf.GradientTape() as tape:
                            # Apply losses to the trainable variables in the policy network
                            variables = policy_net.trainable_variables
                            gradients = tape.gradient(loss, variables)
                            optimizer.apply_gradients(zip(gradients, variables))

                        losses.append(loss.numpy())

                    else:
                        losses.append(0)

                # Select an action for the current step by the network
                action, rate, flag = model.select_greedy_action(observation['observation'], policy_net, int(checkpoint.step))

                # Remember current step to save it with the next step later in the replay buffer
                last_state[agent]['action'] = action
                last_state[agent]['observation'] = observation['observation']

                if done == False:
                    env.step(action)
                else:
                    env.step(None)

            # Copy the weights of policy network to target network at the end of each game
            policy_net.copy_weights_to(target_net)

            epoch = int(checkpoint.epoch)

            # Format the remaining time for the console print
            passed_time = time.time() - start_time
            formated = "{}".format(str(timedelta(seconds=passed_time * ((training_parameters['epochs']-epoch)/100))))

            # Additional print performance of the model every 1000 steps
            if epoch%1000 == 0:
                # Calculate accuracy against random policy and save in a file
                accuracy = test_against_random_policy(env, model, policy_net, observation_space)
                with open(save_path + '/accuracy.csv', 'a' if os.path.exists(save_path + '/accuracy.csv') else 'w+', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([accuracy])

                print(f"Episode:{epoch} Remaining Time: {formated} Winrate against random policy:{accuracy} Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")

                cp_manager.save()

            # Print remaining time losses exploration rate every 100 steps
            elif epoch%100 == 0:
                print(f"Episode:{epoch} Remaining Time: {formated} Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")

            # Increase epoch in checkpoint
            checkpoint.epoch.assign_add(1)
