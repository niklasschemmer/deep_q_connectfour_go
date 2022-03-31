import os
import time
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

class EpsilonGreedyStrategy():
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)

class DeepQ_Agent():
    def __init__(self, strategy, num_actions):
        self.strategy = strategy
        self.num_actions = num_actions

    def select_greedy_action(self, observation, policy_net, current_step):
        rate = self.strategy.get_exploration_rate(current_step)

        if rate > random.random():
            return random.randrange(self.num_actions), rate, True
        else:
            return np.argmax(policy_net(np.atleast_2d(observation).astype('float32'))), rate, False

    def select_random_policy(self, action_mask):
        allowed_actions = np.where(np.array(action_mask) == 1)[0]
        if len(allowed_actions) == 0:
            return 0
        else:
            return np.random.choice(allowed_actions)

def test_against_random_policy(env, model, policy_net, observation_space, num_games=100):
    policy_wins = 0
    for i in range(num_games):
        env.reset()
        play_agent = env.agents[0]
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            observation['observation'] = tf.reshape(observation['observation'], shape=(observation_space))
            action = None
            if agent==play_agent:
                policy_wins += reward
                action = np.argmax(policy_net(np.atleast_2d(observation['observation']).astype('float32')))
            else:
                action = model.select_random_policy(observation['action_mask'])
            if done == False:
                env.step(action)
            else:
                env.step(None)

    return policy_wins/num_games

def copy_weights(Copy_from, Copy_to):
	variables2 = Copy_from.trainable_variables
	variables1 = Copy_to.trainable_variables
	for v1, v2 in zip(variables1, variables2):
		v1.assign(v2.numpy())

def load_checkpoint(checkpoint, path):
    checkpoint.restore(path)

def run_training(env, parameters, policy_net, target_net, checkpoint, cp_manager, memory, action_space, observation_space):
    with tf.device("GPU"):
        optimizer = tf.keras.optimizers.Adam(
                        learning_rate=parameters['learning_rate'],
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07)

        strategy = EpsilonGreedyStrategy(parameters['eps_start'], parameters['eps_end'], parameters['eps_decay'])
        model = DeepQ_Agent(strategy, action_space)

        Experience = namedtuple('Experience', ['observations','actions', 'rewards', 'next_observations', 'dones'])

        copy_weights(policy_net, target_net)

        start_time = time.time()
        while int(checkpoint.epoch) < parameters['epochs']:
            env.reset()
            losses = []

            last_state = {}
            for agent in env.agents:
                last_state[agent] = {
                    'observation': None,
                    'action': None,
                    'reward': 0
                }

            for agent in env.agent_iter():
                observation, reward, done, _ = env.last()
                observation['observation'] = tf.reshape(observation['observation'], shape=(observation_space))

                checkpoint.step.assign_add(1)

                last_state[agent]['reward'] += reward

                if last_state[agent]['action'] != None:
                    memory.push(Experience(last_state[agent]['observation'], last_state[agent]['action'], observation['observation'], reward, done))

                    if memory.can_provide_sample(parameters['batch_size']):
                        experiences = memory.sample(parameters['batch_size'])
                        batch = Experience(*zip(*experiences))

                        observations, actions, rewards, next_observations, dones = np.asarray(batch[0]),np.asarray(batch[1]),np.asarray(batch[3]),np.asarray(batch[2]),np.asarray(batch[4])

                        q_s_a_prime = np.max(target_net(np.atleast_2d(next_observations).astype('float32')), axis = 1)
                        q_s_a_target = np.where(dones, rewards, rewards+parameters['gamma']*q_s_a_prime)
                        q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')

                        with tf.GradientTape() as tape:
                            q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(observations).astype('float32')) * tf.one_hot(actions, action_space), axis=1)
                            loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

                        variables = policy_net.trainable_variables
                        gradients = tape.gradient(loss, variables)
                        optimizer.apply_gradients(zip(gradients, variables))

                        losses.append(loss.numpy())

                    else:
                        losses.append(0)

                action, rate, flag = model.select_greedy_action(observation['observation'], policy_net, int(checkpoint.step))

                last_state[agent]['action'] = action
                last_state[agent]['observation'] = observation['observation']
                if done == False:
                    env.step(action)
                else:
                    env.step(None)

            copy_weights(policy_net, target_net)

            epoch = int(checkpoint.epoch)
            passed_time = (time.time() - start_time)
            start_time = time.time()
            formated = "{}".format(str(timedelta(seconds=passed_time * ((parameters['epochs']-epoch)/100))))
            if epoch%1000 == 0:
                print(f"Episode:{epoch} Remaining Time: {formated} Winrate against random policy:{test_against_random_policy(env, model, policy_net, observation_space)} Losses:{mean(losses): 0.1f} rate:{rate: 0.8f}")
                cp_manager.save()
            elif epoch%100 == 0:
                print(f"Episode:{epoch} Remaining Time: {formated} Losses:{mean(losses): 0.1f} rate:{rate: 0.8f} flag:{flag}")

            checkpoint.epoch.assign_add(1)
