"""
Module to test a game ploicy with manual play

Authors: Dominik Brockmann, Niklas Schemmer
"""
import random
import tensorflow as tf
import numpy as np
import time

def test_game(env, game: object, policy_net: object, observation_space: np.array):
    """
    Play a game against the trained policy net.

    This functions plays a game against the trained policy net to test the performance.
    It has a manual control for the first player and the policy net controls the second player.

    Parameter game: The game wrapper
    Parameter policy_net: The policy model that was trained
    Parameter observation_space: The dimensions of the observation a player can make
    """
    # Reset enviroment and set player to the first agent
    env.reset()
    player_name = env.agents[0]

    # Iterate through both agents
    for agent in env.agent_iter():

        # Render game in a window
        env.render(mode='human')

        # Get last state of game and draw it into the console
        observation, _, done, _ = env.last()
        observation_reshaped = tf.reshape(observation['observation'], shape=(observation_space))
        game.draw(observation['observation'], agent == player_name)

        if done == False:
            # Let either the human player or the trained policy play
            if agent == player_name:
                action = game.manual_policy(observation['observation'], observation['action_mask'])
            else:
                time.sleep(1)
                action = np.argmax(policy_net(np.atleast_2d(observation_reshaped).astype('float32')))

            env.step(action)
        else:
            # If game is over the actual agent wins
            if agent == player_name:
                game.win()
            else:
                game.lose()
            return

def random_policy_go(action_mask: np.array):
    """
    Select an action from the random policy.

    Parameter action_mask: The mask that tells us which actions are possible
    """
    # Get indices of possible actions
    allowed_actions = np.where(np.array(action_mask) == 1)[0]

    # Return zero if no actions possible
    if len(allowed_actions) == 0 or 0.1 > random.random():
        return len(action_mask) - 1
    # Return random index if possible actions exist
    else:
        return np.random.choice(allowed_actions)

def random_policy_connect_four(action_mask: np.array):
    """
    Select an action from the random policy.

    Parameter action_mask: The mask that tells us which actions are possible
    """
    # Get indices of possible actions
    allowed_actions = np.where(np.array(action_mask) == 1)[0]

    # Return zero if no actions possible
    if len(allowed_actions) == 0:
        return len(action_mask) - 1
    # Return random index if possible actions exist
    else:
        return np.random.choice(allowed_actions)