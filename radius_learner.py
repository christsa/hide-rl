import torch
import numpy as np
from utils import layer, project_state


class RadiusModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.act_f = torch.relu

    def forward(self, X):
        return self.fc2(self.act_f(self.fc(X)))

class RadiusLearner():

    def __init__(self,
            device,
            env,
            FLAGS,
            layer_number,
            learning_rate=0.001):

        self.goal_dim = env.subgoal_goal_dim
        self.device = device

        self.n_sample = n_sample
        self.learning_rate = learning_rate
        self.layer_number = layer_number

        # Create actor network
        self.model = RadiusModel().to(self.device)
        self.train = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.path = []

    def add_paths(self, env, FLAGS, transitions):
        num_trans = len(transitions)
        indices = np.reshape(np.random.randint(low=0, high=len(num_trans), size=2*num_trans), (-1, 2))
        start_indices, end_indices = np.append(np.min(indices, axis=-1), 0), np.append(np.max(indices, axis=-1), num_trans-1)
        num_attempts = end_indices - start_indices
        distances = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            distances.append(torch.norm(transitions[end_idx][6] - project_state(env, FLAGS, self.layer_number, transitions[start_idx][0])))
        distances = torch.tensor(distances, dtype=torch.float32, device=self.device)
        num_attempts = torch.tensor(num_attempts, dtype=torch.float32, device=self.device)
        predictions = self.model(num_attempts)
        loss = torch.nn.functional.mse_loss(predictions, distances.unsqueeze(1))
        self.train.zero_grad()
        loss.backward()
        self.train.step()
        metrics["radius_learner/avg_distances"] = distances.mean().item()
        metrics["radius_learner/mse_loss"] = loss.item()

    def get_radius(self, num_attempts):
        with torch.no_grad():
            num_attempts = torch.tensor(num_attempts).view(1,1)
            return self.model(num_attempts).squeeze()

    def state_dict(self):
        return {
            'optimizer': self.train.state_dict(),
            'model': self.model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.train.load_state_dict(state_dict['optimizer'])
        self.model.load_state_dict(state_dict['model'])