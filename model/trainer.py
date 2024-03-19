import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from engine.game import GameRound
from .model import DeepQNetwork
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class TrainConfig:

    lr = 1e-5
    batch_size = 32
    buffer_size = int(1e4)
    gamma = 0.9999
    target_net_update_freq = 500
    greedy_eps = 0.9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    """
    Collects env transitions and performs optimization steps
    """
    
    def __init__(self):
        self.config = TrainConfig()
        self.policy_network = DeepQNetwork().to(self.config.device)
        self.target_network = deepcopy(self.policy_network)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        self.replay_buffer = list()
        self.step_counter = 0
        self.greedy_eps = self.config.greedy_eps
        self.losses = list()
        self.eval_mode = False

    def train(self, num_episodes, out_path):
        print('training started')
        for i in range(num_episodes):
            game = GameRound()
            agent = game.get_hybrid_agent()
            agent.config(self)
            result = game.play_game()
            print(f"episode {i+1}, loss: {np.mean(self.losses):.2f}")
            self.losses = []
        torch.save(self.policy_network.state_dict(), out_path)
        print('training end')

    def eval(self, model_path, num_episodes, vis_mode=False):
        self.policy_network.load_state_dict(torch.load(model_path))
        self.eval_mode = True
        wins = np.zeros((num_episodes,))
        print('evaluation started')
        for i in range(num_episodes):
            game = GameRound(visual_mode=vis_mode)
            agent = game.get_hybrid_agent()
            agent.config(self, eval_mode=True)
            result = game.play_game()
            if result.winner == agent.id:
                wins[i] = 1
        print(f"hybrid agent wins rate: {wins.mean():.1f}")

    def get_epsilon(self):
        """
        Returns decayed over time epsilon hyperparameter
        """
        if self.eval_mode:
            return 0
        decay_step = self.greedy_eps / 2000
        curr_eps = max(0.1, self.greedy_eps - (decay_step * self.step_counter))
        return curr_eps

    def add_transition(self, trans):
        """
        """
        if len(self.replay_buffer) == self.config.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(trans)

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def step(self):
        """
        Make policy network optimizaton step. Random choose a batch of transitions from replay buffer.
        Update target netwrok once for episodes number specified in train config.
        """

        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        self.step_counter += 1
        device = self.config.device
        
        self.optimizer.zero_grad()
        batch = np.random.choice(self.replay_buffer, self.config.batch_size)
        trans_df = pd.DataFrame.from_records(batch)

        states = np.stack(trans_df['state'].values)
        trans_df.drop('state', axis=1, inplace=True)
        states = torch.tensor(states, dtype=torch.float).to(device)
        pred_qvalues = self.policy_network(states)

        next_states = np.stack(trans_df['next_state'].values)
        trans_df.drop('next_state', axis=1, inplace=True)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        with torch.no_grad():
            next_pred_qvalues = self.target_network(next_states)

        # group action from same actions subspace to faster iterating through batch
        trans_df['ids_within_batch'] = trans_df.index
        trans_groups = trans_df.groupby('action_type').agg(list)

        part_losses = []
        for action_type, row in trans_groups.iterrows():
            ids_within_batch = row['ids_within_batch']
            action_ids = row['action_id']
            subset = pred_qvalues[action_type][ids_within_batch, action_ids]
            reward = torch.tensor(row['reward'], dtype=torch.float, requires_grad=False,).to(device)

            next_qvalues = next_pred_qvalues[action_type][ids_within_batch]
            next_selected_qvalues = next_qvalues.max()

            subset_target = reward + (self.config.gamma * next_selected_qvalues)
            loss = nn.functional.mse_loss(subset, subset_target)
            part_losses.append(loss)

        total_loss = torch.sum(torch.stack(part_losses))
        self.losses.append(total_loss.cpu().item())
        total_loss.backward()
        self.optimizer.step()

        if self.step_counter % self.config.target_net_update_freq == 0:
            self.update_target_network()