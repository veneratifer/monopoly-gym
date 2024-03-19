import numpy as np
from engine.agent_funcs import identify_properties_to_improve


class ActionsMasking:
    """
    Class method apply set to -np.inf Q-values of actions for hybrid model 
    that current make no sense or are forbidden by game rules, 
    like selling a property not possessed by a player or buying improved property

    Trade actions Q-values correspond to hybrid agent player itself are filled with -np.inf
    to keep masking simple and player ids mapping straight
    """

    def __init__(self, player):
        self.player_id = player.id
        self.player = player
        self.gamemaster = player.gamemaster
    
    def get_properties_trade_mask(self) -> np.ndarray:
        """
        Compute trade available mask for all 28 purchasable properties in a game
        Property is available for trade if:
        - is not mortgaged
        - is not upgrade if it's a real estate property (has no houses or hotel)
        - and of course if be possessed by some player but this is accounted by other masks

        Returns: mask with shape (28)
        """
        all_properties = self.gamemaster.board.get_properties()
        mask = np.ones_like(all_properties)
        for i, prop in enumerate(all_properties):
            if prop.is_mortgage or getattr(prop, 'build_level', 0) > 0:
                mask[i] = 0
        return mask.astype(bool)
    
    def get_properties_ownership_mask(self) -> np.ndarray:
        """
        Compute ownership status for players and properties product

        Returns: mask with shape (4, 28)
        """
        property_ids = [p.id for p in self.gamemaster.board.get_properties()]
        possessed_by_player = np.array([
            np.isin(property_ids, player.properties) 
            for player in self.gamemaster.players
        ])
        return possessed_by_player
    
    def get_players_bankruptcy_mask(self) -> np.ndarray:
        """
        Compute players bankruptcy mask

        Returns: mask with shape (4)
        """
        mask = [player.is_bankrupt for player in self.gamemaster.players]
        return np.array(mask).astype(bool)
    
    def apply_exchange_trade_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply masking for trade combinations not available currently for a player
        Authors of paper proposed to ommit one of 28 properties in the last dimension
        because same property cannot be requested and proposed simultaneously.
        This method first inserts infs into appropriate places (creates diagonal of infs),
        which restore last dimension to the orginal length (for consisent masking and ids mapping)

        Returns: masked np.ndarray with shape (4, 28, 28)
        """
        source_shape = (3, 28, 27)
        target_shape = (4, 28, 28)
        qvalues = qvalues.reshape(source_shape)
        qvalues = np.insert(qvalues, self.player_id, np.full((28, 27), -np.inf), axis=0)
        new_qvalues = np.zeros(target_shape)
        for i, matrix in enumerate(qvalues):
            for j, row in enumerate(matrix):
                new_qvalues[i, j] = np.insert(row, j, [-np.inf])

        qvalues = new_qvalues

        not_possible_mask = np.zeros(target_shape)

        ownership_mask = self.get_properties_ownership_mask()
        not_possessed_by_this_player = ~ownership_mask[self.player_id]
        not_possible_mask[:, not_possessed_by_this_player] = 1

        not_possessed_by_other_player = ~ownership_mask
        not_possible_mask = np.transpose(not_possible_mask, (0, 2, 1))
        not_possible_mask[not_possessed_by_other_player] = 1
        not_possible_mask = not_possible_mask.transpose((0, 2, 1))

        player_is_bankrupt = self.get_players_bankruptcy_mask()
        not_possible_mask[player_is_bankrupt] = 1

        property_is_not_for_trade = ~self.get_properties_trade_mask()
        not_possible_mask[:, property_is_not_for_trade] = 1
        not_possible_mask[:, :, property_is_not_for_trade] = 1
        
        not_possible_mask = not_possible_mask.astype(bool)
        qvalues[not_possible_mask] = -np.inf
        return qvalues

    def apply_buy_trade_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply masking for properties not currently available for player to buy from other players

        Returns: masked np.ndarray with shape (4, 28, 3)
        """
        source_shape = (3, 28, 3)
        qvalues = qvalues.reshape(source_shape)
        qvalues = np.insert(qvalues, self.player_id, np.full((28, 3), -np.inf), axis=0)
        for_trade_mask = self.get_properties_trade_mask()
        ownership_mask = self.get_properties_ownership_mask()
        bankrupt_mask = self.get_players_bankruptcy_mask()

        qvalues[:, ~for_trade_mask] = -np.inf
        qvalues[bankrupt_mask] = -np.inf
        qvalues[~ownership_mask] = -np.inf
        return qvalues

    def apply_sell_trade_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply masking for properties not currently available for player to sell

        Returns: masked np.ndarray with shape (3, 28, 3)
        """
        source_shape = (3, 28, 3)
        qvalues = qvalues.reshape(source_shape)
        qvalues = np.insert(qvalues, self.player_id, np.full((28, 3), -np.inf), axis=0)
        for_trade_mask = self.get_properties_trade_mask()
        ownership_mask = self.get_properties_ownership_mask()
        bankrupt_mask = self.get_players_bankruptcy_mask()

        qvalues[:, ~(for_trade_mask & ownership_mask[self.player_id])] = -np.inf
        qvalues[bankrupt_mask] = -np.inf
        return qvalues

    def apply_improve_buildings_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply masking for real estates not currently available for player to level-up
        First 22 elements correspond to the houses to build on each property,
        second 22 elements to the hotels to build

        Returns: masked np.ndarray with shape (2, 22)
        """
        qvalues = qvalues.reshape((2, 22))
        real_estates = self.gamemaster.board.get_real_estates()
        real_estates_ids = [p.id for p in real_estates]

        to_improve = identify_properties_to_improve(self.player)
        houses_improv_ids = [a[0] for a in to_improve if a[1] < 5]
        hotels_improv_ids = [a[0] for a in to_improve if a[1] == 5]
        houses_mask = np.isin(real_estates_ids, houses_improv_ids)
        hotels_mask = np.isin(real_estates_ids, hotels_improv_ids)
        mask = np.stack([houses_mask, hotels_mask])
        qvalues[~mask] = -np.inf
        return qvalues

    def apply_sell_buildings_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply masking for properties from which player can't currently sell houses or hotels
        Same mapping like in apply_improve_buildings_masking

        Returns: masked np.ndarray with shape (2, 22)
        """
        qvalues = qvalues.reshape((2, 22))
        real_estates = self.gamemaster.board.get_real_estates()
        build_levels = np.array([pro.build_level for pro in real_estates])
        houses_sell = np.zeros((22,))
        hotels_sell = np.zeros((22,))
        houses_sell[(build_levels > 0) & (build_levels < 5)] = 1
        hotels_sell[build_levels == 5] = 1
        owned_by_player = np.isin([r.id for r in real_estates], self.player.properties)
        
        can_sell_from = np.stack([
            houses_sell.astype(bool) & owned_by_player, 
            hotels_sell.astype(bool) & owned_by_player
        ])
        qvalues[~can_sell_from] = -np.inf
        return qvalues

    def apply_mortgage_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply for properties that cannot be currently mortgaged by player

        Returns: masked np.ndarray with shape (28)
        """
        properties = self.gamemaster.board.get_properties()
        owned_by_player = np.isin([p.id for p in properties], self.player.properties)
        can_mortgage = owned_by_player & np.array([not p.is_mortgage for p in properties])
        qvalues[~can_mortgage] = -np.inf
        return qvalues
        
    def apply_free_mortgage_masking(self, qvalues: np.ndarray) -> np.ndarray:
        """
        Apply masking for properties that cannot be currently unmortgaged by player

        Returns: masked np.ndarray with shape (28)
        """
        properties = self.gamemaster.board.get_properties()
        owned_by_player = np.isin([p.id for p in properties], self.player.properties)
        can_free_mortgage = owned_by_player & np.array([p.is_mortgage for p in properties])
        qvalues[~can_free_mortgage] = -np.inf
        return qvalues
    
    def apply(self, action_type, qvalues: np.ndarray) -> np.ndarray:
        """
        Call appropriate masking accoring to action_type and return masked Q-values
        """
        match action_type:
            case 'exchange_trade':
                return self.apply_exchange_trade_masking(qvalues)
            case 'sell_trade':
                return self.apply_sell_trade_masking(qvalues)
            case 'buy_trade':
                return self.apply_sell_trade_masking(qvalues)
            case 'improve_buildings':
                return self.apply_improve_buildings_masking(qvalues)
            case 'sell_buildings':
                return self.apply_sell_buildings_masking(qvalues)
            case 'mortgage':
                return self.apply_mortgage_masking(qvalues)
            case 'free_mortgage':
                return self.apply_free_mortgage_masking(qvalues)
            case default:
                return qvalues