#import libraries
import gym
import numpy as np
import random
import math
import time

#create environment
environment = gym.make('CartPole-v0')

no_buckets = (1, 1, 6, 3)
no_actions = environment.action_space.n

state_value_bounds = list(zip(environment.observation_space.low, environment.observation_space.high))

action_index = len(no_buckets)

q_value_table = np.zeros(no_buckets + (no_actions,))
min_explore_rate = 0.01
min_learning_rate = 0.1

max_episodes = 1000
max_time_steps = 250
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0


def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = environment.action_space.sample()
    else:
        action = np.argmax(q_value_table[state_value])
    return action


def select_explore_rate(x):
    return max(min_explore_rate, min(1, 1.0 - math.log10((x+1)/25)))


def select_learning_rate(x):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((x+1)/25)))

#You have to Explore this
def bucketize_state_value(state_value):
    bucket_indexes = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = no_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i]-1)*state_value_bounds[i][0]/bound_width
            scaling = (no_buckets[i]-1)/bound_width
            bucket_index = int(round(scaling*state_value[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)

for episode_no in range(max_episodes):
    #Get the exploration rate
    explore_rate = select_explore_rate(episode_no)
    #Get the learning rate
    learning_rate = select_learning_rate(episode_no)
    #Get initial state
    observation = environment.reset()
    #Buckatize the initial state
    start_state_value = bucketize_state_value(observation)

    previous_state_value = start_state_value

    for time_step in range(max_time_steps):
        environment.render()
        #select action
        selected_action = select_action(previous_state_value, explore_rate)
        #Receive next state, reward and done value
        observation, reward_gain, completed, _ = environment.step(selected_action)
        #Bucatize the observation
        state_value = bucketize_state_value(observation)
        #get the Q-value
        best_q_value = np.amax(q_value_table[state_value])
        #Update the table
        q_value_table[previous_state_value + (selected_action,)] += learning_rate * (
                reward_gain + discount * (best_q_value) - q_value_table[previous_state_value + (selected_action,)])

        #Print important parameters
        print('Episode number : %d' % episode_no)
        print('Time step : %d' % time_step)
        print('Selection action : %d' % selected_action)
        print('Current state : %s' % str(state_value))
        print('Reward obtained : %f' % reward_gain)
        print('Best Q value : %f' % best_q_value)
        print('Learning rate : %f' % learning_rate)
        print('Explore rate : %f' % explore_rate)
        print('Streak number : %d' % no_streaks)
        #delay so that we can observe the actions
        time.sleep(0.03)

        if completed:
            print('-----------------------------------')
            print('Episode %d finished after %f time steps' % (episode_no, time_step))
            print('-----------------------------------')
            if time_step >= solved_time:
                no_streaks += 1
            else:
                no_streaks = 0
            break

        previous_state_value = state_value

    if no_streaks > streak_to_end:
        break
