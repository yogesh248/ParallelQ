import gym
import numpy as np
import random
import math
from time import sleep
from time import time
import multiprocessing as mp

## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
mp_arr=mp.Array('d',36)
arr=np.frombuffer(mp_arr.get_obj())
q_table = arr.reshape(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

## Defining the simulation related constants
NUM_EPISODES = 100
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = True

def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate.value:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return mp.Value('d',max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25))))

def get_learning_rate(t):
    return mp.Value('d',max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25))))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

## Instantiating the learning related parameters
learning_rate = get_learning_rate(0)
explore_rate =  get_explore_rate(0)
discount_factor = mp.Value('d',0.99)  # since the world is unchanging

def simulate(begin,end):

    global learning_rate,explore_rate,discount_factor
    num_streaks = 0

    for episode in range(begin,end):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)

        for t in range(MAX_T):
            env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate.value*(reward + discount_factor.value*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate.value)
                print("Learning rate: %f" % learning_rate.value)
                print("Streaks: %d" % num_streaks)

                print("")

            if done:
                print("Episode %d finished"% (episode))
                if (t >= SOLVED_T):
                   num_streaks += 1
                else:
                   num_streaks = 0
                break

            #sleep(0.25)

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


if __name__ == "__main__":
    start=time()

    processes=[mp.Process(target=simulate,args=(x*int((NUM_EPISODES/4)),int((NUM_EPISODES/4))*(x+1))) for x in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()    

    end=time()
    print("Total time taken is {0} seconds".format(end-start))