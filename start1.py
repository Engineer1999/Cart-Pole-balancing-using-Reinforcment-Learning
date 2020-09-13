#import libraries
import gym
import time

#create environment
environment = gym.make('CartPole-v0')
#reset environment
environment.reset()

#iterator for rendaring the environment
for _ in range(100):
    time.sleep(0.05)
    environment.render()
    _, _, completed, _ = environment.step(environment.action_space.sample())
    if completed:
        break;