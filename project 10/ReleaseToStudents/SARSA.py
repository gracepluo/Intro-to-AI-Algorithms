import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle
import math

from collections import defaultdict


EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON_MIN = 0.01  # Minimum epsilon value
EPSILON_MAX = 0.8   # Maximum epsilon value
DECAY_RATE = 0.001  # Decay rate parameter to control decay speed


def default_Q_value():
    return 0


def get_epsilon(episode):
    """
    Calculate epsilon based on the episode number using the formula:
    ε = ε_min + (ε_max - ε_min) * e^(-decay_rate * episode)
    
    Args:
        episode: Current episode number
        
    Returns:
        The epsilon value for the current episode
    """
    return EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * math.exp(-DECAY_RATE * episode)


if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)
    env.reset(seed=1)
    
    for i in range(EPISODES):
        # Get current epsilon for this episode using decay formula
        epsilon = get_epsilon(i)
        
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Implement SARSA with decaying epsilon
        
        while (not done):
            action = env.action_space.sample() # currently only performs a random action.
            obs,reward,terminated,truncated,info = env.step(action)
            episode_reward += reward # update episode reward
            
            done = terminated or truncated
            
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(epsilon) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE_SARSA.pkl' ,'wb')
    pickle.dump([Q_table, get_epsilon(EPISODES-1)], model_file)
    model_file.close()
    #########################