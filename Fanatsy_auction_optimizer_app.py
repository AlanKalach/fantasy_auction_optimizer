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



def run_optimizer(roster_data, scoring, players_df):

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
    
    # Define budget. Every bench spot is assumed to cost the $1 minimum.
    bench_slots = max(roster_size - roster_data['Number'].sum(), 0)
    available_budget = 200 - bench_slots
    
    #create field with points per $ spent
    players_df['points/$'] = players_df['Proj 23'] / players_df['Avg. Salary (AVG)']
    
    #create sturcture to store iteration information
    history_rows = []
    change_history = []
    roster_history =[]
    
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
    return roster, points_game, spent_budget
    

# Button to run the program
if st.button('Run Program'):
    # Process data based on inputs
    result_df, points_game, spent_budget = run_optimizer(roster_data, scoring, players_df)

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
 



