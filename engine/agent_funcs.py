import pandas as pd
import numpy as np
from .fields import *
from typing import Tuple, List


def monopoly_gain(property_id, player_properties, gamemaster) -> bool:
    """
    Determine if property indicated by the property_id 
    added to the current player properties cause a new monopoly
    """
    color_group = gamemaster.board.get_property_group_ids(property_id)
    if color_group == None:
        return False
    new_properties = set(player_properties + [property_id])
    return set(color_group).issubset(new_properties)


def identify_properties_for_sale(player) -> List[Tuple[int, int]]:
    """
    Find player properties that other player would want
    since they lack they to gain new monopoly

    Returns: list of tuples, each tuple in format (property_id, buyer_player_id)
    """
    gamemaster = player.gamemaster
    players_in_game = list(filter(lambda p: not p.is_bankrupt, gamemaster.players))
    sale_poss = list()
    for prop_id in player.properties:
        for other_player in players_in_game:
            if other_player.id == player.id:
                continue
            if monopoly_gain(prop_id, other_player.properties, gamemaster):
                sale_poss.append((prop_id, other_player.id))
    return sale_poss


def identify_properties_for_buy(player) -> List[Tuple[int, int]]:
    """
    Find properties that player could buy to gain new monopoly

    Returns: list of tuples, each tuple in format (property_id, property_owner_id)
    """
    gamemaster = player.gamemaster
    players_in_game = list(filter(lambda p: not p.is_bankrupt, gamemaster.players))
    df = gamemaster.board.prop_df
    df = df[df['color'] != 'neutral']
    player_df = pd.Series(
        data=None,
        index=player.properties,
        name='owned'
    )
    q = df.merge(player_df, left_on='id', right_index=True, how='left')
    lacks = q.groupby('color')['owned'].agg(lambda c: c.isna().sum()).rename('lack_num')
    q = q.merge(lacks, left_on='color', right_index=True)
    buy_poss = list()
    one_lack_properties_ids = q[(q['lack_num'] == 1) & (q['owned'].isna())]['id']
    for searched_property in one_lack_properties_ids:
        for other_player in players_in_game:
            if other_player.id == player.id:
                continue
            if searched_property in other_player.properties:
                buy_poss.append((searched_property, other_player.id))
    return buy_poss


def identify_mortgages(player, money_needed) -> Tuple[List[int], float]:
    """
    Identify which player's properties could be mortgage 
    in order to raise cash specified by money_needed argument

    Returns: tuple containing:
    - list of property ids to mortgage with minimum length to satisfy money_needed demand,
    sorted by ascending mortgage value
    - total cash raised from mortgage of listed properties
    """
    gamemaster = player.gamemaster
    to_mortage = []
    for property_id in player.properties:
        property = gamemaster.board.fields[property_id]
        if property.is_mortgage:
            continue
        if hasattr(property, 'color'):
            if property.build_level > 0:
                continue
        cash_from_bank = property.price // 2
        to_mortage.append((property.id, cash_from_bank))

    to_mortage = sorted(to_mortage, key=lambda t: t[1])
    mortages_cash = np.array(list(map(lambda p: p[1], to_mortage)))
    cash_against_need = (mortages_cash.cumsum() >= money_needed)
    if cash_against_need.any():
        subset_slice = slice(cash_against_need.argmax() + 1)
    else:
        subset_slice = slice(None)
    mortgage_ids = [x[0] for x in to_mortage[subset_slice]]
    return mortgage_ids, mortages_cash[subset_slice].sum()


def identify_sales_to_bank(player, money_needed) -> Tuple[List[int], float]:
    """
    Identify houses and hotels that player can sell to bank 
    to raise money specified by money_needed argument

    Returns: tuple containing:
    - list of property ids from which sell house/hotel (each occurence of property decrease its build level by 1),
    with minimum length to satisfy money_needed demand,
    sorted chronological as they must be performed to align with game rules
    - total amount of money raised by selling all listed buildings
    """
    gamemaster = player.gamemaster
    to_sell = []
    for property_id in player.properties:
        property = gamemaster.get_field(property_id)
        if not hasattr(property, 'color'):
            continue
        if property.build_level == 0:
            continue
        for lvl in range(1, property.build_level + 1):
            cash = property.house_cost / 2 if lvl != 5 else property.hotel_cost / 2
            elem = (property.id, lvl, property.color, cash)
            to_sell.append(elem)
    # fulfill "evenly" selling constraint with sorting secondary key (inverse of level)
    # and "strategic" rule to sell buildings from less valuable assets first with sorting primary key (colors order)
    df = pd.DataFrame.from_records(to_sell, columns=['id', 'level', 'color', 'cash'])
    colors_order = df.groupby('color')['id'].agg('min').rename('colors_order')
    df = df.merge(colors_order, how='left', left_on='color', right_index=True)
    df['inv_level'] = 1 / df['level']
    df.sort_values(['colors_order', 'inv_level'], inplace=True)
    cash_against_need = df['cash'].cumsum() >= money_needed
    if cash_against_need.any():
        subset_slice = slice(cash_against_need.argmax() + 1)
    else:
        subset_slice = slice(None)
    ids = list(df.iloc[subset_slice]['id'].values)
    return ids, df['cash'][subset_slice].sum()
    

def identify_properties_to_improve(player) -> List[Tuple[int, int]]:
    """
    Find players properties that can be level-up by build house or hotel

    Returns: list of property ids to level-up
    """
    gamemaster = player.gamemaster
    may_improve = []
    for property_id in player.properties:
        property = gamemaster.get_field(property_id)
        if not hasattr(property, 'build_level'):
            continue
        color_group = gamemaster.board.get_property_group_ids(property_id)
        if color_group == None:
            continue
        # cannot build when no monopoly upon color group
        if not set(color_group).issubset(set(player.properties)):
            continue
        group_levels = [gamemaster.get_field(p_id).build_level for p_id in color_group]
        # houses could be build only "evenly" and up to 5 build level (hotel)
        if property.build_level == min(group_levels) and property.build_level < 5:
            next_level = property.build_level + 1
            may_improve.append((property_id, next_level))
    return may_improve


def handle_negative_balance(player):
    """
    Try get a money for player to recover from negative cash balance

    Notice that after first identify_mortgages call new possibilities may arise 
    from properties emptied of houses during first call
    """
    gamemaster = player.gamemaster
    needed_cash = abs(player.money)
    mortgages, cash_raised_mortgages = identify_mortgages(player, needed_cash)
    if cash_raised_mortgages < needed_cash:
        buildings_sale, cash_raised_houses = identify_sales_to_bank(
            player, 
            needed_cash - cash_raised_mortgages
        )
        full_cash_raised = cash_raised_mortgages + cash_raised_houses
        for pro_id in buildings_sale:
            pro = gamemaster.get_field(pro_id)
            pro.sell_house()
        if full_cash_raised < needed_cash:
            # further mortgages
            new_mortgages, cahs_raised_sec = identify_mortgages(
                player, 
                needed_cash - full_cash_raised,
            )
            mortgages = mortgages + new_mortgages

    for mortgage_id in mortgages:
        prop = gamemaster.get_field(mortgage_id)
        prop.mortgage()
