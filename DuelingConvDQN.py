import torch.nn as nn
class DuelingConvDQN(nn.Module):
    def __init__(self, device, action_space_size, observation_space_size):
        super(DuelingConvDQN, self).__init__()
        self.action_space_size = action_space_size
        self.observation_space_size = observation_space_size

        self.hidden1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=(3,3), stride=(2,2))
        self.hidden1_activation = nn.ReLU()

        self.hidden2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2))
        self.hidden2_activation = nn.ReLU()

        self.hidden3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.hidden3_activation = nn.ReLU()

        self.hidden4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.hidden4_activation = nn.ReLU()

        self.flatten_layer = nn.Flatten()

        self.values = nn.Sequential(
            nn.Linear(in_features=18432, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        self.advantages = nn.Sequential(
            nn.Linear(in_features=18432, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.action_space_size)
        )

        pass

    def forward(self, state_tensor):
        x = self.hidden1_activation(self.hidden1(state_tensor))
        x = self.hidden2_activation(self.hidden2(x))
        x = self.hidden3_activation(self.hidden3(x))
        x = self.hidden4_activation(self.hidden4(x))
        x = self.flatten_layer(x)
        values = self.values(x)
        advantages = self.advantages(x)
        q_values = values + (advantages - advantages.mean())
        return q_values