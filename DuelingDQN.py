import random
from collections import deque
import numpy as np
import torch
import DuelingConvDQN

class DuelingDQN:
    def __init__(self, state_size, action_size, device):
        self.network = DuelingConvDQN.DuelingConvDQN(
            device=device,
            action_space_size=action_size,
            observation_space_size=state_size
        ).to(device=device)

        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.995
        self.min_epsilon = 0.3
        self.gamma = 0.99  # Discount factor
        self.replay_buffer = deque(maxlen=100000)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0004)
        self.loss_function = torch.nn.MSELoss()
        pass

    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        pass

    def epsilon_greedy(self, state_queue):
        state = torch.cat(list(state_queue), dim=1)
        # Generate random number
        if random.uniform(0,1) < self.get_epsilon():
            # Below epsilon, explore
            Q_values = np.random.randint(self.action_size)
        else:
            # Otherwise, exploit using the main network
            with torch.no_grad():
                Q_values = int(torch.argmax(self.network(state)[0]))
            pass
        return Q_values

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        pass

    def get_epsilon(self):
        return max(self.epsilon, self.min_epsilon)

    def learn_from_experience(self, batch_size):
        # Get a mini batch from the replay memory
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state_queue, action, reward, next_state, done in minibatch:
            state = torch.cat(list(state_queue), dim=1)
            state_queue.append(next_state)
            next_state = torch.cat(list(state_queue), dim=1)
            self.network.eval()
            if not done:
                with torch.no_grad():
                    target_Q = reward + self.gamma * torch.max(self.network(next_state))
            else:
                target_Q = reward
                pass
            self.network.eval()
            with torch.no_grad():
                Q_values = self.network(state)
            Q_values[0][action] = target_Q  # batch size = 1
            self.train_NN(x=state, y=Q_values)
            pass
        pass

    def train_NN(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.network(x)
        loss = self.loss_function(prediction, y)
        loss.backward()
        self.optimizer.step()
        pass






