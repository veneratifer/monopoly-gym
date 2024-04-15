from .agent_funcs import (
    monopoly_gain, 
    identify_properties_for_buy, 
    identify_properties_for_sale, 
    identify_properties_to_improve,
    handle_negative_balance
)
import pandas as pd
import numpy as np
from copy import copy


class FixedPolicyAgent:
    """
    Baseline agent with rule-based policy. Serves for evaluation and training with RL agent.
    Also is a parent class for RL agent since hybrid model takes a few actions in same way
    """

    def __init__(self, player):
        self.player = player
        self.gamemaster = player.gamemaster

    def make_roll(self):
        """
        Roll two dices and update player position, add 200$ if go is passed
        """
        player = self.player
        roll_results = np.random.randint(1, 7, (2,))
        result_pos = player.pos + roll_results.sum()
        fields_n = len(self.gamemaster.board.fields)
        if result_pos >= fields_n:
            player.money += 200
        player.pos = result_pos % fields_n
        entered_field = self.gamemaster.get_field(player.pos)
        entered_field.react(player)
        return roll_results
    
    def consider_property_buy(self):
        """
        Decide whether to buy property where the player landed
        """
        prop = self.gamemaster.get_field(self.player.pos)
        if not hasattr(prop, 'price') or prop.owner != None:
            return False
        if (self.player.money - prop.price) >= 200:
            return True
        else:
            return False
        
    def check_pending_offer(self):
        """
        Check if player has pending offer and consider it if there is
        """
        player = self.player
        if player.pending_offer:
            offer = player.pending_offer
            if self.consider_offer(offer):
                self.gamemaster.exec_offer(offer)
            player.pending_offer = False

    def make_offers(self):
        """
        Send identified offers to other players.
        If buy/sell offer cannot succeed due to insufficient cash balance,
        try make exchange offer by incoporating other properties of interest.
        """
        player = self.player
        to_buy = identify_properties_for_buy(player)
        to_sell = identify_properties_for_sale(player)
        offer_scheme = {
            'money_offered': 0,
            'money_requested': 0,
            'property_offered': None,
            'property_requested': None,
            'player_from': player.id,
            'player_to': None
        }
        used_buy_offer_ids = []
        for i, (prop_id, player_id) in enumerate(to_buy):
            other_player = self.gamemaster.get_player(player_id)
            prop = self.gamemaster.get_field(prop_id)
            offer = copy(offer_scheme)
            if other_player.pending_offer:
                continue
            money_offered = 1.25 * prop.price
            if self.player.money <= money_offered:
                continue
            offer['money_offered'] = money_offered
            offer['player_to'] = player_id
            offer['property_requested'] = prop_id
            other_player.pending_offer = offer
            used_buy_offer_ids.append(i)
        
        used_sell_offer_ids = []
        for i, (prop_id, player_id) in enumerate(to_sell):
            other_player = self.gamemaster.get_player(player_id)
            prop = self.gamemaster.get_field(prop_id)
            offer = copy(offer_scheme)
            if other_player.pending_offer:
                continue
            money_requested = 1.5 * prop.price
            if other_player.money <= money_requested:
                continue
            offer['money_requested'] = money_requested
            offer['property_offered'] = prop_id
            offer['player_to'] = player_id
            other_player.pending_offer = offer
            used_sell_offer_ids.append(i)

        cols = ['property_id', 'player_id']
        sell_df = pd.DataFrame.from_records(to_sell, columns=cols).drop(used_sell_offer_ids, axis=0)
        buy_df = pd.DataFrame.from_records(to_buy, columns=cols).drop(used_buy_offer_ids, axis=0)

        get_field_price = lambda i, g=self.gamemaster: g.get_field(i).price
        sell_df['price'] = sell_df['property_id'].map(get_field_price)
        buy_df['price'] = buy_df['property_id'].map(get_field_price)

        get_player_cash = lambda p_id, g=self.gamemaster: g.get_player(p_id).money
        buy_df['target_player_cash'] = sell_df['player_id'].map(get_player_cash)

        trade_df = pd.merge(sell_df, buy_df, on='player_id', suffixes=['_offered', '_requested'])
        props_value_balance = trade_df['price_offered'] - trade_df['price_requested']
        # check if target player or own player has enough money to pay difference in prices
        target_mask = props_value_balance < 0
        own_mask = props_value_balance >= 0
        props_value_balance[target_mask] += trade_df['target_player_cash'][target_mask]
        props_value_balance[own_mask] += self.player.money

        trade_is_possible = props_value_balance > 0
        for id, trade_offer in trade_df[trade_is_possible].iterrows():
            target_player = self.gamemaster.get_player(trade_offer['player_id'])
            if target_player.pending_offer:
                continue
            offer = copy(offer_scheme)
            offer['player_to'] = target_player.id
            offer['property_requested'] = trade_offer['property_id_requested']
            offer['property_offered'] = trade_offer['property_id_offered']
            net_worth = self._get_offer_net_worth(offer)
            if net_worth >= 0:
                offer['money_requested'] = net_worth
            else:
                offer['money_offered'] = abs(net_worth)
            target_player.pending_offer = offer

    def make_builds(self):
        """
        Level-up available properties following agent policy
        """
        improve_poss_ids = [x[0] for x in identify_properties_to_improve(self.player)]
        for prop_id in improve_poss_ids:
            field = self.gamemaster.get_field(prop_id)
            cost = field.house_cost if field.build_level < 4 else field.hotel_cost
            if self.player.money - cost > 200:
                self.player.money -= cost
                field.build_level += 1

    def manage_mortgages(self):
        """
        Check for mortgaged properties and free mortgages if player money balance enough
        """
        for prop_id in self.player.properties:
            prop = self.gamemaster.get_field(prop_id)
            if prop.mortgage:
                cost = (prop.price / 2) * 1.1
                if self.player.money - cost > 400:
                    prop.free_mortgage()

    def _get_offer_net_worth(self, offer):
        """
        Calculate offer balance for player who would receive this offer
        """
        req_value = 0
        if offer['property_requested'] != None:
            req_value = self.gamemaster.get_field(offer['property_requested']).price
        off_value = 0
        if offer['property_offered'] != None:
            off_value = self.gamemaster.get_field(offer['property_offered']).price
        req_cash = offer['money_offered']
        off_cash = offer['money_requested']
        return (off_value + off_cash) - (req_value + req_cash)

    def _get_offer_type(self, offer: dict) -> str:
        req, off = offer['property_requested'], offer['property_offered']
        if req == None and off != None:
            return 'sell_to_player'
        elif req != None and off == None:
            return 'buy_from_player'
        elif req != None and off != None:
            return 'exchange_offer'
        else:
            raise ValueError('No property requested nor offered')

    def consider_offer(self, offer: dict) -> bool:
        """
        Accept or reject pending trade offer with simple rules
        """
        player = self.player
        offer = player.pending_offer
        offer_type = self._get_offer_type(offer)
        net_worth = self._get_offer_net_worth(offer)
        gamemaster = self.gamemaster
        if offer_type == 'sell_to_player':
            increase_monopolies = monopoly_gain(
                offer['property_offered'], 
                player.properties, 
                gamemaster
            )
            if increase_monopolies and net_worth > -200:
                return True
            elif net_worth > 0:
                return True
        elif offer_type == 'buy_from_player':
            if net_worth > 200:
                return True
        elif offer_type == 'echange_offer':
            increase_monopolies = monopoly_gain(
                offer['property_offered'], 
                list(set(player.properties) - set([offer['requested_property']])), 
                gamemaster
            )
            if increase_monopolies and net_worth > -200:
                return True
        return False
    
    def make_pre_roll_moves(self):
        """
        Make pre roll moves
        """
        self.check_pending_offer()
        player = self.player
        if player.jail_turns_remain:
            if player.community_jail_free_card or player.chance_jail_free_card:
                player.use_jail_free_card()
            elif player.money > 250:
                player.pay_jail_fine()
        self.manage_mortgages()
        self.make_builds()
        return 1
        
    def make_out_of_turn_moves(self):
        """
        Make out of turn moves
        """
        self.check_pending_offer()
        self.make_offers()
        return 2

    def make_post_roll_moves(self):
        """
        Make post roll moves
        """
        self.check_pending_offer()
        if self.consider_property_buy():
            self.player.buy_property(self.player.pos)

    def handle_negative_balance(self) -> bool:
        """
        Returns if recovering positive balance was successful
        """
        handle_negative_balance(self.player)
        return self.player.money >= 0