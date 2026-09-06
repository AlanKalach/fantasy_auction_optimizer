# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:06:47 2023

@author: alank
"""
#Fantasy auction optimizer
import pandas as pd
import numpy as np
import streamlit as st

 

#Import and prepare data--------------------------------------------------------------------------

DATA_FILE = "fantasy_app.xlsx"

st.title('Fantasy Football Optimizer 2026 Season by Alan Kalach')

# Each sheet in the workbook that carries a player table is a selectable source
# (e.g. "Sleeper", "ESPN"). Salaries and stats can be pulled from different ones.
def load_sources(path):
    book = pd.read_excel(path, sheet_name=None)
    return {name: df for name, df in book.items()
            if {'Player', 'Pos', 'Avg. Salary (AVG)', 'Proj 23'}.issubset(df.columns)}

sources = load_sources(DATA_FILE)
SOURCE_NAMES = list(sources.keys())
_default = 'Sleeper' if 'Sleeper' in SOURCE_NAMES else SOURCE_NAMES[0]

st.markdown("### Data Sources")
salary_source = st.selectbox(
    "Salary source (auction $)", SOURCE_NAMES, index=SOURCE_NAMES.index(_default),
    help="Whose auction values to price players with. Pick the platform you draft on.")
stats_source = st.selectbox(
    "Stats / projection source", SOURCE_NAMES, index=SOURCE_NAMES.index(_default),
    help="Whose projected stat lines / points to score players with.")

def _name_key(name):
    # Match players across sources despite "Jr."/"Sr."/suffix and punctuation differences
    key = str(name).lower()
    for junk in [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv", " v", ".", "'", "-"]:
        key = key.replace(junk, "")
    return " ".join(key.split())

def build_players_df(stats_df, salary_df):
    df = stats_df.drop(columns=['Avg. Salary (AVG)'], errors='ignore').copy()
    prices = (salary_df.assign(_k=salary_df['Player'].map(_name_key))
                       .drop_duplicates('_k').set_index('_k')['Avg. Salary (AVG)'])
    df['Avg. Salary (AVG)'] = df['Player'].map(_name_key).map(prices)
    missing = sorted(df.loc[df['Avg. Salary (AVG)'].isna(), 'Player'])
    df['Avg. Salary (AVG)'] = df['Avg. Salary (AVG)'].fillna(1).clip(lower=1)
    return df, missing

players_df, _missing_prices = build_players_df(sources[stats_source], sources[salary_source])
if _missing_prices:
    st.caption(
        f"{len(_missing_prices)} player(s) from the {stats_source} stat list have no "
        f"{salary_source} auction value and were priced at $1: "
        + ", ".join(_missing_prices))

# Manual projection adjustments — injuries, suspensions, legal trouble, or just
# players you don't trust. Negative % cuts the projection (-20 => 80% of points),
# positive % raises it. The workbook's "Adjustments" sheet seeds the defaults;
# the table below is editable per session (add / edit / delete rows).
ADJ_COLS = ['Player', 'Adjust %', 'Comment']

def load_adjustment_seed(path):
    try:
        adj = pd.read_excel(path, sheet_name='Adjustments')
    except (ValueError, KeyError):
        adj = pd.DataFrame(columns=ADJ_COLS)
    adj = adj.rename(columns={'Reason': 'Comment'})
    for col in ADJ_COLS:
        if col not in adj.columns:
            adj[col] = None
    return adj[ADJ_COLS]

with st.expander("Manual player adjustments  —  injuries, suspensions, players you don't trust",
                 expanded=False):
    st.caption("Negative % lowers a player's projected points, positive raises it "
               "(-20 = value them at 80%). Injury / legal defaults load from the workbook; "
               "add, edit or delete rows for this session.")
    _adj_edited = st.data_editor(
        load_adjustment_seed(DATA_FILE), num_rows="dynamic",
        use_container_width=True, hide_index=True, key="adj_editor",
        column_config={
            "Player": st.column_config.TextColumn("Player", required=True),
            "Adjust %": st.column_config.NumberColumn("Adjust %", min_value=-100,
                                                     max_value=100, step=5),
            "Comment": st.column_config.TextColumn("Comment", width="large"),
        })

ADJUSTMENTS = {}
for _p, _v in zip(_adj_edited['Player'], _adj_edited['Adjust %']):
    if isinstance(_p, str) and _p.strip() and pd.notna(_v) and float(_v) != 0:
        ADJUSTMENTS[_name_key(_p)] = 1 + float(_v) / 100.0

if ADJUSTMENTS:
    _hit = set(players_df.loc[players_df['Player'].map(_name_key).isin(ADJUSTMENTS), 'Player']
               .map(_name_key))
    _absent = sorted({p for p in _adj_edited['Player']
                      if isinstance(p, str) and p.strip() and _name_key(p) not in _hit})
    if _absent:
        st.caption("Adjustment set but not matched to a player in the current list: "
                   + ", ".join(_absent))
#Define roster size

st.markdown("### Total Roster Size")
roster_size = st.number_input("Total Roster Size", min_value=14, max_value=20, step=1)
st.markdown("### Starting Roster Size")
starting_QBs = st.number_input("QBs", min_value=1, max_value=2, step=1)
starting_RBs = st.number_input("RBs", min_value=2, max_value=4, step=1)
starting_WRs = st.number_input("WRs", min_value=2, max_value=4, step=1)
starting_TEs = st.number_input("TEs", min_value=1, max_value=2, step=1)
starting_FLEX = st.number_input("FLEX (RB/WR)", min_value=0, max_value=3, step=1)

st.markdown("### Scoring Settings")
pass_td = st.number_input("Pass TD Points", min_value=4, max_value=6, step=1)
rec = st.number_input("Points per Rec", min_value=0.0, max_value=1.0, step=0.1, format="%0.1f")

roster_data = pd.DataFrame({
    'Number': [starting_QBs, starting_RBs, starting_WRs, starting_TEs, starting_FLEX, 1, 1],
    'Slot': ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DEF', 'K']
})

#Positions a FLEX slot can be filled with
FLEX_POSITIONS = ['RB', 'WR']

starting_total = roster_data['Number'].sum()
if starting_total > roster_size:
    st.warning(
        f"The starting lineup needs {starting_total} players but the total roster size is "
        f"{roster_size}. Raise the roster size or drop a starting slot."
    )

scoring = {
    "pass_touchdown": pass_td,
    "passing_yard": .04,
    "rushing_yard": .1,
    "receiving_yard": .1,
    "reception": rec,
    "interception": -2,
    "fumble_lost": -2
}



def run_optimizer(roster_data, scoring, players_df, adjustments=None):

    def calculate_points(row, scoring):
        points = 0
        points += row.get('Pass TD', 0) * scoring['pass_touchdown']
        points += row.get('Pass Yds', 0) * scoring['passing_yard']
        points += row.get('Ru Yds', 0) * scoring['rushing_yard']
        points += row.get('Rec Yds', 0) * scoring['receiving_yard']
        points += row.get('Rec', 0) * scoring['reception']
        points += row.get('Pass Int', 0) * scoring['interception']
        points += row.get('Fum', 0) * scoring['fumble_lost']
        points += (row.get('Ru TD',0) + row.get('Rec TD',0) + row.get('FumTD',0)+row.get('Ret TD',0))*6 + row.get('2PT',0)*2
        return points
    
    def fill_missing_points(df, scoring):
        # Add a new column for points
        # Update only where the column value is NaN
        df['Proj 23'] = df.apply(
            lambda row: calculate_points(row, scoring) if pd.isna(row['Proj 23']) else row['Proj 23'],
            axis=1
            )   
        return df
    
    players_df = fill_missing_points(players_df, scoring)

    # Apply manual projection adjustments after points are settled
    if adjustments:
        players_df = players_df.copy()
        factor = players_df['Player'].map(_name_key).map(adjustments).fillna(1.0)
        players_df['Proj 23'] = players_df['Proj 23'] * factor

    # Define budget. Every bench spot is assumed to cost the $1 minimum.
    bench_slots = max(roster_size - roster_data['Number'].sum(), 0)
    available_budget = 200 - bench_slots
    
    #create field with points per $ spent
    players_df['points/$'] = players_df['Proj 23'] / players_df['Avg. Salary (AVG)']
    
    #create sturcture to store iteration information
    history_rows = []
    change_history = []
    roster_history =[]
    goal_seeks = []   # marginal points-per-$ rate of each accepted swap
    
    #Below all functions------------------------------------------------------------------
    # Which player positions are allowed to fill a given roster slot
    def slot_positions(slot):
        return FLEX_POSITIONS if slot == 'FLEX' else [slot]
    
    #Function to do sensitivity analysis for non selected players 
    def sensitivity(matrix_df, players_df_hardcopy, max_marginal_improvement_row, count):
        matrix_df=matrix_df.apply(pd.to_numeric)
        matrix_df['Maximum'] = matrix_df.apply(max, axis=1)
        rep_player = matrix_df.idxmax(axis=1)
        new_player = pd.Series(matrix_df.index, index=matrix_df.index)
        matrix_df['rep_salary'] = rep_player.map(players_df_hardcopy.set_index('Player')['Avg. Salary (AVG)'])
        matrix_df['rep_points'] = rep_player.map(players_df_hardcopy.set_index('Player')['Proj 23'])
        matrix_df['new_points'] = new_player.map(players_df_hardcopy.set_index('Player')['Proj 23'])
        matrix_df['Goal Seek'] = max_marginal_improvement_row['Marginal Improvement']
        matrix_df['Sensitivity'] = ((matrix_df['new_points']-matrix_df['rep_points'])/matrix_df['Goal Seek'])+matrix_df['rep_salary']
        for player, row in matrix_df.iterrows():
            new_sensitivity = row['Sensitivity']
            # Find the corresponding row in players_df_updated
            player_row = players_df_hardcopy[players_df_hardcopy['Player'] == player]
            # Update sensitivity if the new value is larger
            if not player_row.empty and new_sensitivity > player_row['Sensitivity'].values[0]:
                players_df_hardcopy.loc[player_row.index, 'Sensitivity'] = new_sensitivity
                players_df_hardcopy.loc[player_row.index, 'Iteration'] = count
        return players_df_hardcopy
    
    #Iteration 0--------------------------------------------------------------------------
    
    #Create 2 hardcopys of the players_df
    players_df_hardcopy = players_df.copy()
    players_df_hardcopy_2 = players_df.copy()
    players_df_hardcopy_2['Sensitivity'] =    0.0
    players_df_hardcopy_2['Iteration'] =    0
    
    # Fill the dedicated slots first, each from its own position
    roster_parts = []
    for slot, number in zip(roster_data['Slot'], roster_data['Number']):
        if slot == 'FLEX' or number <= 0:
            continue
        picked = players_df[players_df['Pos'] == slot].nlargest(int(number), 'points/$').copy()
        picked['Slot'] = slot
        roster_parts.append(picked)
    
    # Flex slots then take the best of whoever is left in the eligible positions
    flex_number = int(roster_data.loc[roster_data['Slot'] == 'FLEX', 'Number'].sum())
    if flex_number > 0:
        taken = pd.concat(roster_parts)['Player'] if roster_parts else pd.Series(dtype=object)
        flex_pool = players_df[players_df['Pos'].isin(FLEX_POSITIONS) & ~players_df['Player'].isin(taken)]
        picked = flex_pool.nlargest(flex_number, 'points/$').copy()
        picked['Slot'] = 'FLEX'
        roster_parts.append(picked)
    
    roster = pd.concat(roster_parts)
    #drop selected players
    #players_df = players_df[~players_df['Player'].isin(roster['Player'])]
    
    #calculate points and spent budget
    points_game = roster["Proj 23"].sum()/17
    spent_budget = roster['Avg. Salary (AVG)'].sum() + bench_slots
    available_budget = 200 - spent_budget
    
    #store iteration information
    # Update history DataFrame
    history_rows.append({'Budget spent': spent_budget, 'Points per Game': points_game})
    
    # Update roster_history list
    roster_history += [roster.copy()] 

    #Iteration 1-n -------------------------------------------------------------------------
    count = 1
    iterate = True 
    while count <= 50 and iterate==True:
        #Drop roster players
        players_df = players_df_hardcopy[~players_df_hardcopy['Player'].isin(roster['Player'])]
        result_rows = []
        #for every slot create a matrix of available players vs existing players
        matrix_storage={}
        for slot in roster_data['Slot']:
            pos_players = players_df[players_df['Pos'].isin(slot_positions(slot))]
            pos_roster = roster[roster['Slot']==slot]
            #nothing to swap when the slot is unused or the position pool is exhausted
            if pos_players.empty or pos_roster.empty:
                continue
            # Points and salary deltas for every candidate/incumbent pair at once.
            # Row i, column j is candidate i replacing the incumbent in slot j.
            delta_points = (pos_players['Proj 23'].values[:, None]
                            - pos_roster['Proj 23'].values[None, :])
            delta_salary = (pos_players['Avg. Salary (AVG)'].values[:, None]
                            - pos_roster['Avg. Salary (AVG)'].values[None, :])
            # A swap is off the table if it loses points or busts the budget
            blocked = (delta_points <= 0) | (delta_salary > available_budget)
            with np.errstate(divide='ignore', invalid='ignore'):
                values = delta_points / delta_salary
            values[blocked] = -10*np.random.uniform(size=int(blocked.sum()))
            matrix_df = pd.DataFrame(values, index=pos_players['Player'], columns=pos_roster['Player'])
            #find maximum improvement player
            max_value = float(matrix_df.values.max())
            max_position = matrix_df.values.argmax()
            max_row_index, max_col_index = divmod(max_position, matrix_df.shape[1])
            # Get the names of the row and column for the maximum value
            max_row_name = matrix_df.index[max_row_index]
            max_col_name = matrix_df.columns[max_col_index]
            # Collect the row; result_df is built once, after the position loop
            result_rows.append({
                'Marginal Improvement': max_value,
                'New Player': max_row_name,
                'Old Player': max_col_name,
                'Slot': slot
            })
            #sotre matrix_df for sensitivity purposes
            matrix_name=f'matrix_df_{slot}'
            matrix_storage[matrix_name]=matrix_df
        #no slot had a candidate to evaluate, so there is nothing left to improve
        if not result_rows:
            break
        #find best marginal improvement
        result_df = pd.DataFrame(result_rows)
        max_marginal_improvement_row = result_df.loc[result_df['Marginal Improvement'].idxmax()]
        if max_marginal_improvement_row['Marginal Improvement'] > 0:
            old_player = max_marginal_improvement_row['Old Player']
            new_player = max_marginal_improvement_row['New Player']
            roster_row_index = roster[roster['Player'] == old_player].index[0]
            #replace player
            roster.at[roster_row_index, 'Player'] = new_player
            #a flex swap can bring in a different position, so Pos travels with the player
            roster.at[roster_row_index, 'Pos'] = players_df[players_df['Player']==new_player]['Pos'].values[0]
            roster.at[roster_row_index, 'Avg. Salary (AVG)'] = players_df[players_df['Player']==new_player]['Avg. Salary (AVG)'].values[0]
            roster.at[roster_row_index, 'Proj 23'] = players_df[players_df['Player']==new_player]['Proj 23'].values[0]
            roster.at[roster_row_index, 'points/$'] = players_df[players_df['Player']==new_player]['points/$'].values[0]
            #calculate points and spent budget
            points_game = roster["Proj 23"].sum()/17
            spent_budget = roster['Avg. Salary (AVG)'].sum() + bench_slots
            available_budget = 200 - spent_budget
            #store iteration information
            history_rows.append({'Budget spent': spent_budget, 'Points per Game': points_game})
            goal_seeks.append(float(max_marginal_improvement_row['Marginal Improvement']))
            new_change = pd.DataFrame([max_marginal_improvement_row])
      
            #new_roster = pd.DataFrame([roster])
            #roster_history = pd.concat([roster_history, new_roster], ignore_index=True)
            #apply sensitivity analysis
            for key, dataframe in matrix_storage.items():
                players_df_hardcopy_2= sensitivity(dataframe, players_df_hardcopy_2, max_marginal_improvement_row, count)
        else:
            iterate = False
        count += 1
    
    #Iteration display------------------------------------------------------------------
    history = pd.DataFrame(history_rows)
    changes = pd.DataFrame(change_history)
    
    column_dict = {col: [] for col in ['Player', 'Avg. Salary (AVG)', 'Proj 23']}
    for df in roster_history:
        for col in ['Player', 'Avg. Salary (AVG)', 'Proj 23']:
            column_dict[col].extend(df[col])
    roster_evolution = pd.DataFrame(column_dict)
    # marginal points-per-$ at the optimum: average the last few accepted swaps
    # (the tail of the run, where it has nearly converged) to damp the algorithm's
    # built-in randomness
    lam = float(np.mean(goal_seeks[-3:])) if goal_seeks else None
    return roster, points_game, spent_budget, players_df_hardcopy_2, lam, available_budget, bench_slots


def bid_analysis(roster, pool, lam, one_player_cap):
    """Turn the last optimization into auction guidance.

    lam = marginal points gained per extra $ at the optimum. 1/lam = $ per point.
    For each rostered player: the ceiling price you could pay before the best
    still-available replacement becomes the better buy.
    For everyone else: the price at or below which they'd bump your weakest
    starter at their position and belong in the roster instead.
    """
    if not lam or lam <= 0:
        return None, None
    dollars_per_point = 1.0 / lam

    def slot_positions(slot):
        return FLEX_POSITIONS if slot == 'FLEX' else [slot]

    rostered = set(roster['Player'])
    avail = pool[~pool['Player'].isin(rostered)].copy()
    slots = list(roster['Slot'].unique())

    weakest_starter = {}   # current starter most at risk in each slot
    for s in slots:
        held = roster[roster['Slot'] == s]
        if not held.empty:
            weakest_starter[s] = held.loc[held['Proj 23'].idxmin()]

    def fallback_for(r):
        # who you'd actually slot in if you lost this player: the best still-available
        # option at his slot costing the same or less; if everything left costs more,
        # the cheapest one available
        cand = avail[avail['Pos'].isin(slot_positions(r['Slot']))]
        if cand.empty:
            return None
        cheaper = cand[cand['Avg. Salary (AVG)'] <= r['Avg. Salary (AVG)']]
        if not cheaper.empty:
            return cheaper.loc[cheaper['Proj 23'].idxmax()]
        return cand.loc[cand['Avg. Salary (AVG)'].idxmin()]

    ceil_rows = []
    for _, r in roster.iterrows():
        alt = fallback_for(r)
        if alt is None:
            ceiling, altname = one_player_cap, '-'
        else:
            ceiling = alt['Avg. Salary (AVG)'] + (r['Proj 23'] - alt['Proj 23']) * dollars_per_point
            altname = alt['Player']
        ceiling = max(1.0, min(float(ceiling), float(one_player_cap)))
        ceil_rows.append({
            'Player': r['Player'], 'Slot': r['Slot'],
            'Proj': round(float(r['Proj 23']), 1),
            'Est. $': int(round(r['Avg. Salary (AVG)'])),
            'Max bid $': int(round(ceiling)),
            'vs Est.': f"{int(round(ceiling - r['Avg. Salary (AVG)'])):+d}",
            'Fallback': altname,
        })

    tgt_rows = []
    for _, p in avail.iterrows():
        opts = [s for s in weakest_starter if p['Pos'] in slot_positions(s)]
        if not opts:
            continue
        s = max(opts, key=lambda s: p['Proj 23'] - weakest_starter[s]['Proj 23'])
        w = weakest_starter[s]
        target = w['Avg. Salary (AVG)'] + (p['Proj 23'] - w['Proj 23']) * dollars_per_point
        tgt_rows.append({
            'Player': p['Player'], 'Pos': p['Pos'],
            'Proj': round(float(p['Proj 23']), 1),
            'Est. $': int(round(p['Avg. Salary (AVG)'])),
            'Buy at/below $': int(np.floor(target)),
            'Bumps': w['Player'],
        })
    ceilings = pd.DataFrame(ceil_rows)
    targets = pd.DataFrame(tgt_rows)
    if not targets.empty:
        targets = (targets[targets['Buy at/below $'] >= 1]
                   .sort_values('Proj', ascending=False).reset_index(drop=True))
    return ceilings, targets


# Button to run the program
show_bids = st.checkbox(
    "Advanced: show max bid per rostered player and target price for the rest",
    value=False)

if st.button('Run Program'):
    # Process data based on inputs
    (result_df, points_game, spent_budget,
     _pool, _lam, _avail_budget, _bench) = run_optimizer(
        roster_data, scoring, players_df, adjustments=ADJUSTMENTS)

    roster_for_bids = result_df.copy()   # keep original column names for bid_analysis

    # Add the sum row to the DataFrame
    result_df = result_df.rename(columns={"Proj 23": "Projected Points"})
    result_df = result_df.rename(columns={"Avg. Salary (AVG)": "Avg. Salary"})
    columns_to_display = ["Player", "Slot", "Pos", "Projected Points", "Avg. Salary"]
    filtered_results=result_df[columns_to_display].copy()
    # Define the custom order for the 'Slot' field
    slot_order = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'FLEX': 5, 'K': 6, 'DEF': 7}
    # Add a new column to sort by custom slot order
    filtered_results['Slot Order'] = filtered_results['Slot'].map(slot_order)
    # Sort the DataFrame by 'Slot Order' and 'Projected Points'
    results_sorted = filtered_results.sort_values(by=['Slot Order', 'Projected Points'], ascending=[True, False])
    # Drop the temporary 'Slot Order' column
    results_sorted = results_sorted.drop(columns=['Slot Order'])
    st.write('Optimal Roster:')
    st.dataframe(results_sorted, width=700, height=400)
    st.write(f'Points per Game: {points_game:.2f}')
    st.write(f'Budget Spent: {spent_budget}')

    if show_bids:
        st.markdown("### Advanced: auction bid guidance")
        one_player_cap = 200 - _bench - (int(roster_data['Number'].sum()) - 1)
        ceilings, targets = bid_analysis(roster_for_bids, _pool, _lam, one_player_cap)
        if ceilings is None:
            st.info("Not enough optimizer movement to estimate bid values this run.")
        else:
            st.caption(
                f"Marginal value at this roster: ~${1/_lam:,.1f} per projected point "
                f"(1 extra $ buys ~{_lam:,.2f} pts). Numbers are guidance, not hard limits."
            )
            st.markdown("**Players you drafted — how high you can go**")
            st.caption("Max bid = the price at which the best still-available "
                       "replacement at that slot becomes the smarter buy.")
            st.dataframe(ceilings, hide_index=True, width=700)
            st.markdown("**Players you didn't draft — when they become worth it**")
            st.caption("Buy at/below = the price at which this player would bump "
                       "your weakest starter at his position.")
            st.dataframe(targets, hide_index=True, width=700, height=430)
 



