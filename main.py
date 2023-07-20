import gymnasium as gym
import numpy as np
import torch
from collections import deque
import DuelingDQN as ddqn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("Cuda unavailable")
    device = torch.device('cpu')
    pass

num_episodes = 1000
num_timesteps = 20000
batch_size = 4
base_channels = 16
env = gym.make("ALE/Pong-v5", render_mode='human')
action_size = env.action_space.n
state_size = env.observation_space.shape[0]
agent = ddqn.DuelingDQN(state_size=state_size,action_size=action_size, device=device)
all_episodes_return = deque(maxlen=100)
state_queue = deque(maxlen=8)

def pre_process_state(state):
    state = state[30:-12, 5:-4]
    state = torch.tensor(state)
    state = torch.unsqueeze(state, dim=0)
    state = torch.permute(state, (0, 3, 1, 2))
    state = state.to(device)
    state = state/255.
    state = torch.mean(state, dim=1, keepdim=True)
    return state

for episode in range(num_episodes):
    state = env.reset()[0]
    state = pre_process_state(state)
    episode_return = 0
    for time_step in range(num_timesteps):
        state_queue.append(state)
        action = agent.epsilon_greedy(state_queue)
        agent.decay_epsilon()
        next_state, reward, done, max_steps, meta_data = env.step(action)
        next_state = pre_process_state(next_state)
        if len(state_queue) == 8:
            agent.memorize(state=state_queue, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state
        episode_return += reward

        if done or max_steps:
            print("Total reward for episode {1}: {0}".format(episode_return, episode))
            all_episodes_return.append(episode_return)
            print("Running average reward = {0}".format(np.mean(np.array(all_episodes_return))))
            break

        if len(agent.replay_buffer) > batch_size:
            agent.learn_from_experience(batch_size=batch_size)
            pass
        pass


    pass





