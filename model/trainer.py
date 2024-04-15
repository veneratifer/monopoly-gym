import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.optimizers.legacy import Adam
from engine.game import GameRound
from .model import DeepQNetwork
from engine.gui import GameBoard
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


class Trainer:
    """
    Collects env transitions and performs optimization steps
    """
    
    def __init__(self):
        self.config = TrainConfig()
        self.policy_network = DeepQNetwork()
        self.target_network = deepcopy(self.policy_network)
        self.optimizer = Adam(learning_rate=self.config.lr)
        self.replay_buffer = list()
        self.step_counter = 0
        self.greedy_eps = self.config.greedy_eps
        self.losses = list()
        self.eval_mode = False
        self.loss_fn = keras.losses.MeanSquaredError()

    def train(self, num_episodes, out_path):
        print('training started')
        for i in range(num_episodes):
            game = GameRound()
            agent = game.get_hybrid_agent()
            agent.config(self)
            result = game.play_game()
            print(f"episode {i+1}, loss: {np.mean(self.losses):.2f}")
            self.losses = []

        self.policy_network.save_weights(out_path)
        print('training end')

    def eval(self, model_path, num_episodes, vis_mode=False):
        self.policy_network.load_weights(model_path)
        self.eval_mode = True
        wins = np.zeros((num_episodes,))
        print('evaluation started')
        if vis_mode:
            assert num_episodes < 2, "cannot run evaluation in visual mode for many games"
            gui = GameBoard()
        for i in range(num_episodes):
            game = GameRound()
            if vis_mode:
                game.attach_gui(gui)
            agent = game.get_hybrid_agent()
            agent.config(self, eval_mode=True)
            result = game.play_game()
            if result['winner'] == agent.player.id:
                wins[i] = 1

        print(f"hybrid agent wins rate: {wins.mean():.2f} on {num_episodes} games")

    def get_epsilon(self):
        """
        Returns decayed over time epsilon
        """
        if self.eval_mode:
            return 0
        decay_step = self.greedy_eps / 2000
        curr_eps = max(0.1, self.greedy_eps - (decay_step * self.step_counter))
        return curr_eps

    def add_transition(self, trans):
        if len(self.replay_buffer) == self.config.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(trans)

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())

    def step(self):
        """
        Make policy network optimizaton step. Random choose a batch of transitions from replay buffer.
        Update target netwrok once for episodes number specified in train config.
        """

        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        self.step_counter += 1
        
        batch = np.random.choice(self.replay_buffer, self.config.batch_size)
        trans_df = pd.DataFrame.from_records(batch)

        states = np.stack(trans_df['state'].values)
        trans_df.drop('state', axis=1, inplace=True)
        states = tf.constant(states, dtype=float)

        next_states = np.stack(trans_df['next_state'].values)
        trans_df.drop('next_state', axis=1, inplace=True)
        next_states = tf.constant(next_states, dtype=float)
        next_pred_qvalues = self.target_network(next_states)

        # group action from same actions subspace to faster iterating through batch
        trans_df['ids_within_batch'] = trans_df.index.to_list()
        trans_groups = trans_df.groupby('action_type').agg(list)

        part_losses = []
        with tf.GradientTape() as tape:
            pred_qvalues = self.policy_network(states)
            # track all model weights used in this batch
            trained_vars = list()
            for layer in self.policy_network.encoder.layers:
                trained_vars.extend(layer.variables)
            for act_type in trans_groups.index:
                layer = self.policy_network.output_layer[act_type]
                trained_vars.extend(layer.variables)

            for action_type, row in trans_groups.iterrows():
                ids_within_batch = row['ids_within_batch']
                action_ids = row['action_id']
                indicies = np.asarray(list(zip(ids_within_batch, action_ids)))
                subset = tf.gather_nd(pred_qvalues[action_type], indicies)
                reward = tf.constant(row['reward'], dtype=float)

                next_qvalues = next_pred_qvalues[action_type]
                next_qvalues = tf.gather(next_qvalues, ids_within_batch, axis=0)
                next_selected_qvalues = tf.reduce_max(next_qvalues)

                subset_target = reward + (self.config.gamma * next_selected_qvalues)
                loss = self.loss_fn(subset_target, subset)
                part_losses.append(loss)

            total_loss = tf.reduce_sum(tf.stack(part_losses))
            self.losses.append(total_loss.numpy())
            grads = tape.gradient(total_loss, trained_vars)
            self.optimizer.apply_gradients(zip(grads, trained_vars))

        if self.step_counter % self.config.target_net_update_freq == 0:
            self.update_target_network()