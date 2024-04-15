import tensorflow as tf
import keras
from keras.layers import Dense
import numpy as np
from typing import Dict


class DeepQNetwork(keras.Model):

    def __init__(self):
        super().__init__()
        self.encoder = keras.Sequential([
            Dense(1024, activation='relu'),
            Dense(512, activation='relu')
        ])
        # layers dict provides better organization and easier masking
        self.output_layer = {
            'exchange_trade': Dense(2268),
            'sell_trade': Dense(252),
            'buy_trade': Dense(252),
            'improve_buildings': Dense(44),
            'sell_buildings': Dense(44),
            'mortgage': Dense(28),
            'free_mortgage': Dense(28),
            'skip_turn': Dense(1),
            'conclude_actions': Dense(1),
            'use_jail_free_card': Dense(1),
            'pay_jail_fine': Dense(1)
        }
        actions = list(self.output_layer.keys())
        self.actions = actions
        self.phase_allowable_actions = {
            'pre_roll': actions,
            'out_of_turn': actions[:-2],
            'post_roll': actions[4:8]
        }

    def call(self, x: tf.Tensor, actions_subspace: list = None) -> Dict[str, tf.Tensor]:
        """
        Returns Q-values for game state(s) represented by tensor x.
        Action space is narowed to some subset by actions_subset list.
        """
        if x.ndim < 2:
            x = tf.expand_dims(x, axis=0)
        if actions_subspace == None:
            actions_subspace = self.actions
        valid_actions = np.isin(actions_subspace, self.actions)
        if not valid_actions.all():
            invalid_actions = np.array(actions_subspace)[~valid_actions]
            message = f"actions of types: {', '.join(invalid_actions)} are not allowable for call"
            raise ValueError(message)
        encoded_x = self.encoder(x)
        result = dict()
        for action in actions_subspace:
            y = self.output_layer[action](encoded_x)
            result[action] = y
        return result