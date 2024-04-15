from typing import List
import numpy as np


class Card:
    """
    Each chance or community card must inherite from this class
    """
    
    def __init__(self, name, gamemaster):
        self.name = name
        self.gamemaster = gamemaster

    def __call__(self, player):
        """
        Action performed when player draw this card
        """
        pass


# Chance cards
    
class AdvanceToBoardWalk(Card):

    def __call__(self, player):
        player.pos = 39
        field = self.gamemaster.get_field(39)
        field.react(player)


class AdvanceToGo(Card):

    def __call__(self, player):
        player.pos = 0
        player.money += 200


class AdvanceToIllinoisAvenue(Card):

    def __call__(self, player):
        field = self.gamemaster.get_field(24)
        player.pos = 24
        field.react(player)


class AdvanceToCharlesPlace(Card):

    def __call__(self, player):
        field = self.gamemaster.get_field(11)
        player.pos = 11
        field.react(player)


def get_nearest_field(current_pos, class_name: str, gamemaster):
    """
    Get field of defined class (e.g utility or train) that is closest (moving forward) to the current player position
    """
    fields = gamemaster.board.fields
    match_class = lambda f, c_name=class_name: type(f).__name__ == c_name
    fit_fields = list(filter(match_class, fields))
    assert len(fit_fields) > 0, f"no fields of class {class_name} found on board"
    fit_fields_positions = [field.id for field in fit_fields]
    if current_pos > max(fit_fields_positions):
        return fit_fields[0]
    else:
        nearest_id = np.argmax(np.array(fit_fields_positions) > current_pos)
        return fit_fields[nearest_id]


class AdvanceToNeareastRailroad(Card):

    def __call__(self, player):
        nearest_railroad = get_nearest_field(player.pos, 'TrainStation', self.gamemaster)
        player.pos = nearest_railroad.id
        # two calls for two times fee
        nearest_railroad.react(player)
        nearest_railroad.react(player)


class AdvanceToNearestUtility(Card):

    def __call__(self, player):
        nearest_utility = get_nearest_field(player.pos, 'UtilityField', self.gamemaster)
        player.pos = nearest_utility.id
        roll_sum = 0
        if nearest_utility.owner != None:
            roll_sum = np.random.randint(1, 7, (2,)).sum()
        nearest_utility.react(player, force_roll_sum=roll_sum * 10)


class BankDividend(Card):

    def __call__(self, player):
        player.money += 50


class GetOutOfJailFree(Card):
    pass


class CommunityJailFree(GetOutOfJailFree):

    def __call__(self, player):
        player.community_jail_free_card = True


class ChanceJailFreeCard(GetOutOfJailFree):

    def __call__(self, player):
        player.chance_jail_free_card = True


class GoBack3Spaces(Card):

    def __call__(self, player):
        curr_pos = player.pos
        next_pos = curr_pos - 3 if (curr_pos - 3) >= 0 else 40 - abs(curr_pos - 3)
        player.pos = next_pos
        field = self.gamemaster.get_field(next_pos)
        field.react(player)


class GoToJail(Card):

    def __call__(self, player):
        self.gamemaster.put_in_prison(player.id)


class RepairsOfProperties(Card):

    def __call__(self, player):
        cumm_fee = 0
        for prop_id in player.properties:
            prop = self.gamemaster.get_field(prop_id)
            if not hasattr(prop, 'build_level'):
                continue
            lvl = prop.build_level
            fee = lvl * 25
            if lvl == 5:
                fee += 75
            cumm_fee += fee
        player.money -= cumm_fee


class SpeedingFines(Card):

    def __call__(self, player):
        player.money -= 15


class GoToReadingRailroad(Card):

    def __call__(self, player):
        curr_pos = player.pos
        destin = self.gamemaster.get_field(5)
        if curr_pos > destin.id:
            player.money += 200
        player.pos = destin.id


class ChairmanElection(Card):

    def __call__(self, player):
        for pl in self.gamemaster.get_players_in_game():
            if pl.id != player.id:
                player.money += 50
                pl.money -= 50
        

class LoanMatures(Card):

    def __call__(self, player):
        player.money += 150


def get_chance_cards(self) -> List[Card]:
    gamemaster = self.gamemaster
    return [
        AdvanceToBoardWalk('Advance to Boardwalk', gamemaster),
        AdvanceToGo('Advance to Go', gamemaster),
        AdvanceToIllinoisAvenue('Advance to Illinois Avenue', gamemaster),
        AdvanceToCharlesPlace('Advance to St. Charles Place', gamemaster),
        AdvanceToNeareastRailroad('Advance to nearest railroad', gamemaster),
        AdvanceToNearestUtility('Advance to nearest utility', gamemaster),
        BankDividend('Bank dividened', gamemaster),
        GetOutOfJailFree('Get out of jail free card', gamemaster),
        GoBack3Spaces('Go back 3 spaces', gamemaster),
        GoToJail('Go dirctly to jail', gamemaster),
        RepairsOfProperties('Obligatory repairs of properties', gamemaster),
        SpeedingFines('Speeding Fines', gamemaster),
        GoToReadingRailroad('Go to Reading Railroad', gamemaster),
        ChairmanElection('Election of Chairman of the Board', gamemaster),
        LoanMatures('Load matures collect', gamemaster)
    ]


# Community Chest cards

class BankError(Card):

    def __call__(self, player):
        player.money += 200


class DocktorFee(Card):

    def __call__(self, player):
        player.money -= 50


class StockSale(Card):

    def __call__(self, player):
        player.money += 50


class HolidayFundMatures(Card):

    def __call__(self, player):
        player.money += 100


class Birthday(Card):

    def __call__(self, player):
        for pl in self.gamemaster.get_players_in_game():
            if pl.id != player.id:
                player.money += 10
                pl.money -= 10


class IncomeTaxRefund(Card):

    def __call__(self, player):
        player.money += 20
      

class LifeInsuranceMatures(Card):

    def __call__(self, player):
        player.money += 100


class HospitalFees(Card):

    def __call__(self, player):
        player.money -= 100


class SchoolFees(Card):

    def __call__(self, player):
        player.money -= 50


class ConsultancyFee(Card):

    def __call__(self, player):
        player.money += 25


class StreetRepair(Card):

    def __call__(self, player):
        cumm_fee = 0
        for prop_id in player.properties:
            prop = self.gamemaster.get_field(prop_id)
            if not hasattr(prop, 'build_level'):
                continue
            lvl = prop.build_level
            fee = lvl * 40
            if lvl == 5:
                fee += 75
            cumm_fee += fee
        player.money -= cumm_fee


class BeautyPrize(Card):

    def __call__(self, player):
        player.money += 10


class InheritanceCard(Card):

    def __call__(self, player):
        player.money += 100


def get_community_chest_cards(self) -> List[Card]:
    gamemaster = self.gamemaster
    return [
        AdvanceToGo('Advance to Go', gamemaster),
        BankError('Bank error in your favor', gamemaster),
        DocktorFee('Doctor fee', gamemaster),
        StockSale('From sale of stock you get 50$', gamemaster),
        GetOutOfJailFree('Get out of jail free card', gamemaster),
        GoToJail('Go to jail directly', gamemaster),
        HolidayFundMatures('Holiday fund matures', gamemaster),
        IncomeTaxRefund('Income tax refund', gamemaster),
        Birthday('It is your birthday', gamemaster),
        LifeInsuranceMatures('Life insurance matures', gamemaster),
        HospitalFees('Pay hospital fees', gamemaster),
        SchoolFees('Pay school fees', gamemaster),
        ConsultancyFee('Receive consultancy fee', gamemaster),
        StreetRepair('Ypu are assessed for street repair', gamemaster),
        BeautyPrize('You won second prize in beauty consent', gamemaster),
        InheritanceCard('You inherit 100$', gamemaster)
    ]