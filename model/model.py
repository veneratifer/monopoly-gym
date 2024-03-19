import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class DeepQNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(240, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        # modules dict provides better organization and easier masking
        self.output_layer = nn.ModuleDict({
            'exchange_trade': nn.Linear(512, 2268),
            'sell_trade': nn.Linear(512, 252),
            'buy_trade': nn.Linear(512, 252),
            'improve_buildings': nn.Linear(512, 44),
            'sell_buildings': nn.Linear(512, 44),
            'mortgage': nn.Linear(512, 28),
            'free_mortgage': nn.Linear(512, 28),
            'skip_turn': nn.Linear(512, 1),
            'conclude_actions': nn.Linear(512, 1),
            'use_jail_free_card': nn.Linear(512, 1),
            'pay_jail_fine': nn.Linear(512, 1)
        })
        actions = list(self.output_layer.keys())
        self.actions = actions
        self.phase_allowable_actions = {
            'pre_roll': actions,
            'out_of_turn': actions[:-2],
            'post_roll': actions[4:8]
        }

    def forward(self, x: torch.Tensor, actions_subspace: list = None) -> Dict[str, torch.Tensor]:
        """
        Returns Q-values for game state(s) represented by tensor x.
        Action space is narowed to some subset by actions_subset list.
        """
        if actions_subspace == None:
            actions_subspace = self.actions
        valid_actions = np.isin(actions_subspace, self.actions)
        if not valid_actions.all():
            invalid_actions = np.array(actions_subspace)[~valid_actions]
            message = f"actions of types: {', '.join(invalid_actions)} are not allowable for forward call"
            raise ValueError(message)
        encoded_x = self.encoder(x)
        result = dict()
        for action in actions_subspace:
            y = self.output_layer[action](encoded_x)
            result[action] = y
        return result