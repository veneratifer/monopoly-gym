import numpy as np
from .cards import (
    get_chance_cards, 
    get_community_chest_cards, 
    GetOutOfJailFree
)


class Field:
    """
    Base board field class, may be subclassed
    """

    def __init__(self, id, gamemaster, name):
        self.id = id
        self.name = name
        self.gamemaster = gamemaster

    def react(self, player):
        """called always when player enters field"""
        pass


class Purchasable(Field):
    """
    Represent each board field with purchase option (real estates and utilities).
    Attributes:
    - owner: None or int indicate ownership
    - price: just property price
    - is_mortgage: bool flag determines if property is mortgaged now
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.owner = None
        self.price = float(kwargs['price'])
        self.is_mortgage = False

    def fee_(self, player, *args, **kwargs):
        """calculate and apply fee, to implement in derived classes"""
        pass

    def set_owner(self, val):
        self.owner = val
        if val == None:
            self.is_mortgage = False

    def react(self, player, *args, **kwargs):
        if player.id != self.owner and self.owner != None:
            if player.jail_turns_remain:
                return
            self.fee_(player, *args, **kwargs)

    def mortgage(self):
        player = self.gamemaster.get_player(self.owner)
        player.money += self.price / 2
        self.is_mortgage = True

    def free_mortgage(self):
        cost = int((self.price / 2) * 1.1)
        owner = self.gamemaster.get_player(self.owner)
        owner.money -= cost
        self.is_mortgage = False


class Property(Purchasable):
    """
    Real estate field
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = None
        self.color = kwargs['color']
        self.build_level = 0
        dev_table = kwargs['dev_table']
        self.house_cost = float(dev_table['house_price'])
        self.hotel_cost = 4 * self.house_cost + self.house_cost
        rent_cols = ['rent_0', 'rent_1', 'rent_2', 'rent_3', 'rent_4', 'rent_hotel']
        self.fees = dev_table[rent_cols].values

    def fee_(self, player, *args, **kwargs):
        if self.is_mortgage:
            return
        fee = self.fees[self.build_level]
        owner = self.gamemaster.get_player(self.owner)
        player.money -= fee
        owner.money += fee

    def sell_house(self):
        self.build_level -= 1
        self.gamemaster.get_player(self.owner).money += self.house_cost / 2

    def sell_hotel(self):
        self.build_level -= 1
        self.gamemaster.get_player(self.onwer).money += self.hotel_cost / 2


class UtilityField(Purchasable):

    def utils_common_owner(self):
        utils = list(filter(lambda x: isinstance(x, UtilityField), self.gamemaster.board.fields))
        return utils[0].owner == utils[1].owner

    def fee_(self, player, *args, **kwargs):
        owner = self.gamemaster.get_player(self.owner)
        multipier = 10 if self.utils_common_owner() else 4
        if 'force_roll_sum' in kwargs.keys():
            roll_sum = kwargs['force_roll_sum']
        else:
            roll_sum = np.random.randint(1, 7, (2,)).sum()
        fee = roll_sum * multipier
        player.money -= fee
        owner.money += fee
        

class StartField(Field):
    pass


class TrainStation(Purchasable):

    fees = {1: 25, 2: 50, 3: 100, 4: 200}

    def calc_fee(self):
        owned_train_num = 0
        for loc in self.gamemaster.board.fields:
            if isinstance(loc, TrainStation):
                if loc.owner == self.owner:
                    owned_train_num += 1
        return TrainStation.fees[owned_train_num]

    def fee_(self, player, *args, **kwargs):
        owner = self.gamemaster.get_player(self.owner)
        fee = self.calc_fee()
        player.money -= fee
        owner.money += fee
    

class IncomeTax(Field):

    def react(self, player):
        player.money -= 200


class LuxuryTax(Field):

    def react(self, player):
        player.money -= 100


class GoToJail(Field):

    def react(self, player):
        self.gamemaster.put_in_prison(player.id)


class VisitJail(Field):
    pass


class CardsField(Field):
    """
    Parent class for handling functionality of both Chance cards and Community Chest cards.
    Shuffling is done by permutation of indicies and player draw card according to the current stack_pos
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        if not hasattr(self.__class__, 'cards_stack'):
            self.__class__.cards_stack = self.get_cards_func()

    @classmethod
    def react(cls, player):
        if cls.stack_pos == len(cls.cards_stack):
            cls.stack_pos = 0
            np.random.shuffle(cls.cards_stack)
        if cls.jail_card_out:
            # skip this stack position since jail card is already owned by some player
            x = cls.cards_stack[cls.stack_pos]
            if isinstance(x, GetOutOfJailFree):
                cls.stack_pos += 1
                cls.react(player)
                return
        card = cls.cards_stack[cls.stack_pos]
        if isinstance(card, GetOutOfJailFree):
            cls.jail_card_out = True
        card(player)
        cls.stack_pos += 1


class CommunityChest(CardsField):

    stack_pos = 0
    jail_card_out = False
    get_cards_func = get_community_chest_cards


class ChanceField(CardsField):

    stack_pos = 0
    jail_card_out = False
    get_cards_func = get_chance_cards