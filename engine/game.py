import pandas as pd
import numpy as np
from . import DATA_DIR
from .fields import *
from .agent import FixedPolicyAgent
from model.hybrid_agent import HybridRLAgent


class Board:
    """
    Container for all board fields, initialize they according to the proper csv files.
    Used by GameRound instance and GUI module
    """

    def __init__(self, gamemaster):
        self.gamemaster = gamemaster
        fields_df = pd.read_csv(f"{DATA_DIR}/fields.csv")
        properties_df = pd.read_csv(f"{DATA_DIR}/properties.csv")
        self.prop_df = properties_df
        full_df = fields_df.merge(properties_df, on='id', how='left')
        self.fields = tuple([self.field_factor(row) for i, row in full_df.iterrows()])
        for f in self.fields:
            if isinstance(f, VisitJail):
                self.prison_id = f.id

    def field_factor(self, field_row):
        field_type = field_row['field_type']
        f_id, f_name = field_row[['id', 'name']]
        basic_args = (f_id, self.gamemaster, f_name)
        dev_tables = self.prop_df.set_index('id')
        match field_type:
            case 'basic':
                return Field(*basic_args)
            case 'property':
                p, c = field_row[['price', 'color']]
                dev_table = dev_tables.loc[f_id]
                return Property(*basic_args, price=p, color=c, dev_table=dev_table)
            case 'start':
                return StartField(*basic_args)
            case 'utility':
                return UtilityField(*basic_args, price=field_row['price'])
            case 'train':
                return TrainStation(*basic_args, price=field_row['price'])
            case 'income_tax':
                return IncomeTax(*basic_args)
            case 'luxury_tax':
                return LuxuryTax(*basic_args)
            case 'go_to_jail':
                return GoToJail(*basic_args)
            case 'visit_jail':
                return VisitJail(*basic_args)
            case 'chance':
                return ChanceField(*basic_args)
            case 'community_chest':
                return CommunityChest(*basic_args)
            case default:
                return Field(*basic_args)

    def get_field(self, pos_id) -> Field:
        return self.fields[pos_id]
    
    def get_prison_id(self):
        return self.prison_id
    
    def get_properties(self) -> list:
        all_properties = list()
        for field in self.fields:
            if isinstance(field, Purchasable):
                all_properties.append(field)
        return all_properties
    
    def get_real_estates(self) -> list:
        real_estates = list()
        for field in self.fields:
            if isinstance(field, Property):
                real_estates.append(field)
        return real_estates
    
    def get_property_group_ids(self, property_id):
        """
        return the list of indicies for color group that the property with passed id belongs to
        """
        check_property = self.fields[property_id]
        if not hasattr(check_property, 'color'):
            return None
        group_color = check_property.color
        belongs_to_group = lambda field, g_color=group_color: getattr(field, 'color', '') == g_color
        group = filter(belongs_to_group, self.fields)
        return [f.id for f in group]


class Player:
    """
    Class represent a player in simulation, has following attributes:
    
    - id: unique id for this player
    - gamemaster: GameRound object reference
    - agent: controller object responsible for player actions
    - pos: current player position on the game board (int)
    - properties: list of property ids that player posses 
    - jail_turns_remain: variable takes value 0 (player is not in jail) or 1-3 (remained turns in jail)
    - money: player money amount
    - is_bankrupt: boolean flag for player game status
    - pending_offer: takes value False (no offer for this player) or dict object with submitted offer
    - community_jail_free_card: does player own jail free card from community chest stack
    - chance_jail_free_card: does player own jail free card from chance stack
    """

    def __init__(self, id, gamemaster, hybrid=False):
        self.id = id
        self.gamemaster = gamemaster
        self.hybrid = hybrid
        if hybrid:
            self.agent = HybridRLAgent(self)
        else:
            self.agent = FixedPolicyAgent(self)
        self.pos = 0
        self.properties = []
        self.jail_turns_remain = 0
        self.is_bankrupt = False
        self.money = 1500
        self.community_jail_free_card = False
        self.chance_jail_free_card = False
        self.pending_offer = False

    def add_property(self, property_id):
        self.properties.append(property_id)
        field = self.gamemaster.get_field(property_id)
        field.set_owner(self.id)

    def remove_property(self, property_id):
        self.properties.remove(property_id)
        self.gamemaster.get_field(property_id).set_owner(None)

    def buy_property(self, property_id):
        self.add_property(property_id)
        field = self.gamemaster.get_field(property_id)
        self.money -= field.price

    def use_jail_free_card(self):
        """
        Search for a available jail free card and use it
        """
        from_chance = self.chance_jail_free_card
        from_community = self.community_jail_free_card
        if from_chance:
            self.chance_jail_free_card = False
            ChanceField.jail_card_out = False
            self.jail_turns_remain = 0
        elif from_community:
            self.community_jail_free_card = False
            CommunityChest.jail_card_out = False
            self.jail_turns_remain = 0

    def pay_jail_fine(self):
        self.jail_turns_remain = 0
        self.money -= 50

    def bankruptcy(self):
        """
        Terminate player participation in current game
        """
        self.is_bankrupt = True
        for prop_id in self.properties:
            field = self.gamemaster.get_field(prop_id)
            field.set_owner(None)
        self.properties = []
        self.pos = 0
        self.money = 0


class GameRound:
    """
    Main program class stores players and board, 
    intermediates in trade offers and performs main loop of simulation
    """

    def __init__(self):
        self.board = Board(self)
        self.init_players()
        self.turns_counter = 0
        self.visual_mode = False

    def attach_gui(self, gui):
        self.visual_mode = True
        gui.set_gamemaster(self)
        self.gui_ = gui

    def init_players(self):
        # todo: hybrid agent player id is fixed for now, 
        # but can be random if adjust game state representation
        self.players = tuple([Player(idx, self, hybrid=idx == 1) for idx in range(4)])

    def winner(self) -> bool:
        return np.sum([not p.is_bankrupt for p in self.players]) == 1

    def put_in_prison(self, player_id):
        player = self.get_player(player_id)
        player.pos = self.board.get_prison_id()
        player.jail_turns_remain = 3

    def get_field(self, pos) -> Field:
        return self.board.get_field(pos)
    
    def get_player(self, player_id) -> Player:
        return self.players[player_id]
    
    def get_players_in_game(self):
        return [player for player in self.players if not player.is_bankrupt]
    
    def get_hybrid_agent(self):
        hybrid_agent = [p for p in self.players if p.hybrid][0]
        return hybrid_agent.agent
    
    def exec_offer(self, offer: dict):
        """
        Execute offer described by offer argument
        """
        player_from = self.get_player(offer['player_from'])
        player_to = self.get_player(offer['player_to'])
        traded_properties = list()
        if offer['property_requested'] != None:
            player_to.remove_property(offer['property_requested'])
            player_from.add_property(offer['property_requested'])
            traded_properties.append(offer['property_requested'])
        if offer['property_offered'] != None:
            player_from.remove_property(offer['property_offered'])
            player_to.add_property(offer['property_offered'])
            traded_properties.append(offer['property_offered'])
        if offer['money_requested'] != None:
            player_from.money += offer['money_requested']
            player_to.money -= offer['money_requested']
        if offer['money_offered'] != None:
            player_from.money -= offer['money_offered']
            player_to.money += offer['money_offered']
        # remove all pending offers that regard properties traded in this offer
        for player in self.players:
            if player.pending_offer:
                off = player.pending_offer
                req, offe = off['property_requested'], off['property_offered']
                if req in traded_properties or offe in traded_properties:
                    player.pending_offer = False

    def next_turn(self):
        """
        Do full set of moves in turn for one player and simultanouse out of turn moves for other players.
        Update GUI if in interactive mode
        """
        self.turns_counter += 1
        if self.visual_mode:
            self.gui_.update_board()
            if self.gui_.stop_game.get():
                self.gui_.wait_variable(self.gui_.stop_game)

        current_player_id = self.current_player_id
        players_in_game = self.get_players_in_game()
        current_player = players_in_game[current_player_id]
        if current_player.jail_turns_remain:
            current_player.jail_turns_remain -= 1

        out_of_turn_player_id = (current_player_id + 1) % len(players_in_game)
        skip_turn_counter = 0

        current_player.agent.make_pre_roll_moves()

        out_of_turn_count = 0
        # Out of turn phase for all players
        oot_limit = 5 * len(players_in_game)
        while skip_turn_counter < len(players_in_game) and out_of_turn_count < oot_limit:
            if self.visual_mode:
                self.gui_.update_board()
            out_of_turn_count += 1
            out_of_turn_player = players_in_game[out_of_turn_player_id]
            move_code = out_of_turn_player.agent.make_out_of_turn_moves()
            if move_code == 2:
                # 2 is a code for skip turn
                skip_turn_counter += 1
            else:
                skip_turn_counter = 0

            out_of_turn_player_id = (out_of_turn_player_id + 1) % len(players_in_game)

        if not current_player.jail_turns_remain:
            results = current_player.agent.make_roll()
            current_player.agent.make_post_roll_moves()
            if self.visual_mode:
                self.gui_.update_dices(results)
                self.gui_.update_board()

        if current_player.money < 0:
            if current_player.agent.handle_negative_balance() == False:
                current_player.bankruptcy()
                # don't update current_player_id because after bakruptcy is already points to the next player
                if current_player_id == len(players_in_game) - 1:
                    self.current_player_id = 0
        else:
            self.current_player_id = (current_player_id + 1) % len(players_in_game)

        if self.visual_mode:
            if self.winner():
                self.gui_.quit()
                return
            self.gui_.after(500, self.next_turn)

    def play_game(self):
        """
        simulation loop executed until winner appears, 
        game turn is divided into three phases:
        - pre-roll: current player decide about actions before making dices roll
        - out-of-turn: player in this phase can take some actions 
        - post-roll: after roll dices
        """
        self.current_player_id = 0
        if self.visual_mode:
            self.gui_.init_elems()
            self.gui_.after(500, self.next_turn)
            self.gui_.mainloop()
        else:
            while self.winner() == False:
                self.next_turn()

        result = {
            'winner': self.get_players_in_game()[0].id,
            'turns_num': self.turns_counter
        }
        return result

    def get_board_state(self) -> np.ndarray:
        """
        Get representation of board purchasable fields for RL agent.
        Each such field has attributed flat vector with following components:
        - one-hot encoded ownership (4 numbers)
        - flag for being part of color group monopoly (always False for trains and utilities)
        - fraction of builded houses on max 4 available (0 for trains and utilities)
        - fraction of builded hotels on max 1 available (0 for trains and utilities)
        - flag means whether property is current mortgaged

        Returns: np.ndarray with shape (224)
        """
        vectors = list()
        for field in self.board.fields:
            if isinstance(field, Purchasable):
                owner_vec = np.zeros((4,))
                if field.owner != None:
                    owner_vec[field.owner] = 1
                is_part_of_monopoly = False
                houses_builds_frac = 0.
                hotels_builds_frac = 0.
                if isinstance(field, Property):
                    houses_builds_frac = min(1, field.build_level / 4)
                    hotels_builds_frac = field.build_level == 5
                    color_group = self.board.get_property_group_ids(field.id)
                    owners = [self.get_field(f_id).owner for f_id in color_group]
                    if not None in owners:
                        # same owner for all fields indicate monopoly
                        is_part_of_monopoly = all([own == owners[0] for own in owners])
                mortgage_flag = field.is_mortgage
                rest_vec = np.array([
                    is_part_of_monopoly, 
                    houses_builds_frac, 
                    hotels_builds_frac,
                    mortgage_flag
                ], dtype=float)
                
                field_vec = np.concatenate([owner_vec, rest_vec])
                vectors.append(field_vec)

        # shape (28 x 8)
        return np.stack(vectors).flatten()
    
    def get_player_state(self, player_id) -> np.ndarray:
        """
        Get representation of player state for RL agent.
        Player vector consists of these components:
        - current location on board (int)
        - amount of cash player has
        - flag denoting if player is current in jail
        - flag indicating if player has get out of jail free card

        Returns: np.ndarray with shape (4)
        """
        player = self.get_player(player_id)
        has_free_jail_card = player.community_jail_free_card or player.chance_jail_free_card
        return np.array([
            player.pos,
            player.money, 
            player.jail_turns_remain > 0,
            has_free_jail_card
            ], dtype=float
        )

    def get_game_state(self) -> np.ndarray:
        """
        Returns concatenated representations of board and players with shape (240)
        """
        players_repr = np.concatenate([self.get_player_state(idx) for idx in range(4)])
        full_repr = np.concatenate([players_repr, self.get_board_state()])
        return full_repr