from engine.agent import FixedPolicyAgent
import tensorflow as tf
import numpy as np
from .masking import ActionsMasking
from typing import Dict


class HybridRLAgent(FixedPolicyAgent):
    """
    Hybrid model, makes decisions during simulation relying mainly on deep neural network
    and in few cases takes actions based on inherited methods
    """
    
    def __init__(self, player):
        self.player = player
        self.gamemaster = player.gamemaster
        self.masker = ActionsMasking(player)

    def config(self, trainer, eval_mode=False):
        self.model = trainer.policy_network
        self.trainer = trainer
        self.eval_mode = eval_mode

    def make_move(self, q_values: Dict[str, tf.Tensor]):
        """
        Find Q-value with max value and perform linked action.
        We get q_values already pre-filtered to only valid action groups during current game phase.

        Returns move code, one of three:
        - 0 if player was not idle and took some action
        - 1 means conclude actions (end of pre roll phase)
        - 2 indicate skip turn move
        """
        move_code = 0

        action_types = list(q_values.keys())
        multidim_ids, flat_ids = [], []
        max_vals = list()
        for action_type, qval in q_values.items():
            qval = tf.squeeze(qval, axis=0)
            qval_np = qval.numpy()
            qval_np = self.masker.apply(action_type, qval_np)
            action_max_val = np.max(qval_np)
            max_vals.append(action_max_val)
            multidim_ids.append(np.argwhere(qval_np == action_max_val)[0])
            if action_max_val == -np.inf:
                flat_ids.append(0)
            else:
                # flat_id using for network output indexing
                # note that for trade actions q-values masking also pad them,
                # so we have to use orginal tensor
                flat_ids.append(tf.where(qval == action_max_val).numpy().item())
        
        max_vals = np.array(max_vals)
        # greedy-epsilon algorithm
        if np.random.rand(1) < self.trainer.get_epsilon():
            valid_actions = max_vals[max_vals != -np.inf]
            randarg = np.random.choice(np.arange(len(valid_actions)))
            argwhere = np.argwhere(max_vals == valid_actions[randarg])
            argmax = argwhere[0].item()
        else:
            argmax = np.argmax(max_vals)
            
        max_action_type = action_types[argmax]
        multidim_id = multidim_ids[argmax]
        flat_id = flat_ids[argmax]
        
        transition = {
            'state': self.gamemaster.get_game_state(),
            'action_type': max_action_type,
            'action_id': flat_id
        }

        properties = self.gamemaster.board.get_properties()
        real_estates = self.gamemaster.board.get_real_estates()
        # discrete cash categories indicate fraction of property base price to request/propose
        cash_cats = [0.75, 1., 1.25]
                
        match max_action_type:
            case 'exchange_trade_offer':
                other_player_id, property_for_sale_id, property_for_buy_id = multidim_id
                offer = dict()
                offer['player_from'] = self.player.id
                offer['player_to'] = other_player_id
                prop_for_sale = properties[property_for_sale_id]
                prop_for_buy = properties[property_for_buy_id]
                offer['property_requested'] = prop_for_buy.id
                offer['property_offered'] = prop_for_sale.id
                # offer net value have to be zero
                net_val = prop_for_buy.price - prop_for_sale.price
                offer['money_offered'] = max(0, net_val)
                offer['money_requested'] = abs(min(0, net_val))
                other_player = self.gamemaster.get_player(other_player_id)
                if not other_player.pending_offer:
                    other_player.pending_offer = offer
            case 'sell_trade':
                other_player_id, property_for_sale_id, cash_category_id = multidim_id
                property_offered = properties[property_for_sale_id]
                offer = dict()
                offer['player_from'] = self.player.id
                offer['player_to'] = other_player_id
                offer['property_offered'] = property_offered.id
                offer['property_requested'] = None
                offer['money_offered'] = 0
                offer['money_requested'] = property_offered.price * cash_cats[cash_category_id]
                other_player = self.gamemaster.get_player(other_player_id)
                if not other_player.pending_offer:
                    other_player.pending_offer = offer
            case 'buy_trade':
                other_player_id, prop_for_buy_id, cash_category_id = multidim_id
                property_requested = properties[prop_for_buy_id]
                offer = dict()
                offer['player_from'] = self.player.id
                offer['player_to'] = other_player_id
                offer['property_offered'] = None
                offer['property_requested'] = property_requested.id
                offer['money_offered'] = property_requested.price * cash_cats[cash_category_id]
                offer['money_requested'] = 0
                other_player = self.gamemaster.get_player(other_player_id)
                if not other_player.pending_offer:
                    other_player.pending_offer = offer
            case 'improve_buildings':
                build_type, estate_id = multidim_id
                estate = real_estates[estate_id]
                if build_type == 0:
                    cost = estate.house_cost
                    if (self.player.money - cost) > 0:
                        estate.build_level += 1
                        self.player.money -= estate.house_cost
                elif build_type == 1:
                    cost = estate.hotel_cost
                    if (self.player.money - cost) > 0:
                        estate.build_level += 1
            case 'sell_buildings':
                sell_type, estate_id = multidim_id
                estate = real_estates[estate_id]
                if sell_type == 0:
                    estate.sell_house()
                elif sell_type == 1:
                    estate.sell_hotel()
            case 'mortgage':
                property_id = multidim_id[0]
                prop = properties[property_id]
                prop.mortgage()
            case 'free_mortgage':
                property_id = multidim_id[0]
                prop = properties[property_id]
                prop.free_mortgage()
            case 'conclude_actions':
                move_code = 1
            case 'use_jail_free_card':
                self.player.use_jail_free_card()
            case 'pay_jail_fine':
                self.player.pay_jail_fine()
            case 'skip_turn':
                move_code = 2
            case default:
                move_code = 2
        
        if not self.eval_mode:
            reward = self.get_reward()
            transition['reward'] = reward
            transition['next_state'] = self.gamemaster.get_game_state()
            self.trainer.add_transition(transition)
            self.trainer.step()
        
        return move_code
    
    def get_reward(self) -> float:
        """
        Calculates and returns reward function value for current game state
        """
        active_players = self.gamemaster.get_players_in_game()
        other_players_cumm_val = 0
        for player in active_players:
            if player.id == self.player.id:
                continue
            other_players_cumm_val += self.get_player_net_worth(player.id)
        return self.get_player_net_worth(self.player.id) / other_players_cumm_val
        
    def get_player_net_worth(self, player_id) -> float:
        """
        Comupte and return net worth for a player
        """
        player = self.gamemaster.get_player(player_id)
        cash = player.money
        cumm_properties_val = 0
        for prop_id in player.properties:
            prop = self.gamemaster.get_field(prop_id)
            color_group = self.gamemaster.board.get_property_group_ids(prop_id)
            mortgage_penalty = 1.1 * prop.price * int(prop.is_mortgage)
            if color_group != None:
                is_part_of_monopoly = set(color_group).issubset(player.properties)
                b = 2 if is_part_of_monopoly else 1.5
                prop_val = (prop.price - mortgage_penalty) * b
                if prop.build_level < 5:
                    prop_val += prop.build_level * prop.house_cost
                else:
                    prop_val += prop.house_cost * 4 + prop.hotel_cost
                cumm_properties_val += prop_val
            else:
                b = 1.5
                cumm_properties_val += (prop.price - mortgage_penalty) * b
        return cash + cumm_properties_val

    def get_qvalues(self, game_phase: str) -> Dict[str, tf.Tensor]:
        """
        Returns Q-values from policy network approved for the current game phase
        """
        state = self.gamemaster.get_game_state()
        state = tf.convert_to_tensor(state, dtype=float)
        actions = self.model.phase_allowable_actions[game_phase]
        qvalues = self.model(state, actions)
        return qvalues
    
    def make_pre_roll_moves(self):
        self.check_pending_offer()
        qvalues = self.get_qvalues('pre_roll')
        return self.make_move(qvalues)
        
    def make_out_of_turn_moves(self):
        self.check_pending_offer()
        qvalues = self.get_qvalues('out_of_turn')
        return self.make_move(qvalues)
    
    def make_post_roll_moves(self):
        if self.consider_property_buy():
            self.player.buy_property(self.player.pos)
        qvalues = self.get_qvalues('post_roll')
        return self.make_move(qvalues)
    