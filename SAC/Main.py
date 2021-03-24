from matplotlib import pyplot as plt
from Car import CarRacing
import tensorflow as tf
import numpy as np
import keyboard
from SAC import Agent

if __name__ == '__main__':
    #=========================Enable GPU usage===========================
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #========Operational Instructions=========
    render = True
    debugging = False
    training = True
    load_checkpoint = True

    #============================Constants================
    n_episodes = 7000
    env = CarRacing(obstacles = False)
    agent = Agent(env)
    total_steps = 0
    if load_checkpoint:
        total_rewards = list(agent.load_rewards('total'))
        avg_rewards = list(agent.load_rewards('avg'))
        episode_offset = len(total_rewards)
        agent.load_models()
        agent.load_buffer()
    else:
        total_rewards = []
        avg_rewards = []

    #================Main loop===============================
    for episode in range(n_episodes):
        if load_checkpoint: episode += episode_offset
        env.reset()
        episode_step = 0
        print('\n','='*10,'Episode ',episode + 1, '='*10)
        done = False
        episode_reward = 0
        while not done:
            if render: isopen = env.render()
            else: isopen = True
            if debugging and keyboard.is_pressed('~'): done = True; isopen = False; break
            episode_step += 1
            done, reward = agent.step(training, episode + 1, episode_step)
            episode_reward += reward
        total_rewards.append(episode_reward)
        avg_rewards.append(np.mean(total_rewards[-100:]))
        print('Episode Score: {} | AVG Score: {} | Maximum Episode Score: {}'.format(episode_reward, avg_rewards[-1], np.max(total_rewards)))
        if (episode+1) % 2 == 0: 
            agent.save_rewards(total_rewards, 'total')
            agent.save_rewards(avg_rewards, 'avg')
        total_steps += episode_step
        if not isopen: env.close();  break
    env.close()