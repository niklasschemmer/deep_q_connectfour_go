import tensorflow as tf
import numpy as np
import time

def test_game(env, game, policy_net, observation_space):
    env.reset()

    for agent in env.agent_iter():

        env.render(mode='human')
        observation, reward, done, info = env.last()
        observation_reshaped = tf.reshape(observation['observation'], shape=(observation_space))
        game.draw(observation['observation'], agent == 'player_0' if not done else False)

        if done == False:
            if agent == 'player_0':
                action = game.manual_policy(observation['observation'], observation['action_mask'])
            else:
                time.sleep(1)
                action = np.argmax(policy_net(np.atleast_2d(observation_reshaped).astype('float32')))

            env.step(action)
        else:
            if agent == 'player_0':
                game.win()
            else:
                game.lose()
            return
