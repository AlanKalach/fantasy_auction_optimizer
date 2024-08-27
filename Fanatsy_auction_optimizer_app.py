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

# Read Excel data into a data frame
players_df = pd.read_excel("fantasy_app.xlsx")
#Define roster size

st.title('Fantasy Football Optimizer')

st.markdown("### Total Roster Size")
roster_size = st.number_input("Total Roster Size", min_value=14, max_value=20, step=1)
st.markdown("### Starting Roster Size")
starting_QBs = st.number_input("QBs", min_value=1, max_value=2, step=1)
starting_RBs = st.number_input("RBs", min_value=2, max_value=4, step=1)
starting_WRs = st.number_input("WRs", min_value=2, max_value=4, step=1)
starting_TEs = st.number_input("TEs", min_value=1, max_value=2, step=1)

st.markdown("### Scoring Settings")
pass_td = st.number_input("Pass TD Points", min_value=4, max_value=6, step=1)
rec = st.number_input("Points per Rec", min_value=0, max_value=1, step=1)


roster_data = pd.DataFrame({
    'Number': [starting_QBs, starting_RBs, starting_WRs, starting_TEs, 1, 1],
    'Pos': ['QB', 'RB', 'WR', 'TE', 'DEF', 'K']
})

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
    
    # Define budget
    available_budget = 200 - (roster_size-roster_data['Number'].sum())
    
    #create field with points per $ spent
    players_df['points/$'] = players_df['Proj 23'] / players_df['Avg. Salary (AVG)']
    
    #create sturcture to store iteration information
    history = pd.DataFrame(columns=['Budget spent', 'Points per Game'])
    change_history = []
    roster_history =[]
    
    #Below all functions------------------------------------------------------------------
    # Function to get top 'Number' players for each position, for iteration 0
    def get_top_players(group):
        return group.nlargest(group['Number'].iloc[0], 'points/$')
    
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
    players_df_hardcopy_2['Sensitivity'] =    0
    players_df_hardcopy_2['Iteration'] =    0
    
    # Merge data frames based on 'Position'
    players_df = players_df.merge(roster_data, on='Pos')
    
    # Apply the function to get the top 'Number' players for each position
    roster = players_df.groupby('Pos', group_keys=False).apply(get_top_players)
    #drop selected players
    #players_df = players_df[~players_df['Player'].isin(roster['Player'])]
    
    #calculate points and spent budget
    points_game = roster["Proj 23"].sum()/17
    spent_budget = roster['Avg. Salary (AVG)'].sum() + (15-roster_data['Number'].sum())
    available_budget = 200 - spent_budget
    
    #store iteration information
    # Update history DataFrame
    new_row = pd.DataFrame([[spent_budget, points_game]], columns=['Budget spent', 'Points per Game'])
    history = pd.concat([history, new_row], ignore_index=True)
    
    # Update roster_history list
    roster_history += [roster.copy()] 
    
    #Iteration 1-n -------------------------------------------------------------------------
    count = 1
    iterate = True 
    while count <= 50 and iterate==True:
        #Drop roster players
        players_df = players_df_hardcopy[~players_df_hardcopy['Player'].isin(roster['Player'])]
        result_df= pd.DataFrame(columns=['Marginal Improvement', 'New Player','Old Player', 'Pos' ] )
        #for every position create a matrix of available players vs existing players
        matrix_storage={}
        for pos in roster_data['Pos']:
            pos_players = players_df[players_df['Pos']==pos]
            pos_roster = roster[roster['Pos']==pos]
            matrix_df = pd.DataFrame(index=pos_players['Player'], columns=pos_roster['Player'])
            for pos_player in matrix_df.index:
                for roster_player in matrix_df.columns:
                    if ((players_df[players_df['Player'] == pos_player]['Proj 23'].values[0])-(roster[roster['Player'] == roster_player]['Proj 23'].values[0])) <= 0 or ((players_df[players_df['Player'] == pos_player]['Avg. Salary (AVG)'].values[0])-(roster[roster['Player'] == roster_player]['Avg. Salary (AVG)'].values[0])) > available_budget:
                        matrix_df.loc[pos_player, roster_player] = -10*np.random.uniform()
                    else:
                        matrix_df.loc[pos_player, roster_player] = ((players_df[players_df['Player'] == pos_player]['Proj 23'].values[0])-(roster[roster['Player'] == roster_player]['Proj 23'].values[0])) / ((players_df[players_df['Player'] == pos_player]['Avg. Salary (AVG)'].values[0])-(roster[roster['Player'] == roster_player]['Avg. Salary (AVG)'].values[0]))
            #find maximum improvement player
            max_value = matrix_df.values.max()
            max_position = matrix_df.values.argmax()
            max_row_index, max_col_index = divmod(max_position, matrix_df.shape[1])
            # Get the names of the row and column for the maximum value
            max_row_name = matrix_df.index[max_row_index]
            max_col_name = matrix_df.columns[max_col_index]
            # Convert the new row into a DataFrame and concatenate it with the existing result_df DataFrame
            new_row = pd.DataFrame([{
                'Marginal Improvement': max_value,
                'New Player': max_row_name,
                'Old Player': max_col_name,
                'Pos': pos
            }])
            result_df = pd.concat([result_df, new_row], ignore_index=True)
            #find best marginal improvement
            max_marginal_improvement_row = result_df.loc[result_df['Marginal Improvement'].idxmax()]
            #sotre matrix_df for sensitivity purposes
            matrix_name=f'matrix_df_{pos}'
            matrix_storage[matrix_name]=matrix_df
        if max_marginal_improvement_row['Marginal Improvement'] > 0:
            old_player = max_marginal_improvement_row['Old Player']
            new_player = max_marginal_improvement_row['New Player']
            roster_row_index = roster[roster['Player'] == old_player].index[0]
            #replace player
            roster.at[roster_row_index, 'Player'] = new_player
            roster.at[roster_row_index, 'Avg. Salary (AVG)'] = players_df[players_df['Player']==new_player]['Avg. Salary (AVG)']
            roster.at[roster_row_index, 'Proj 23'] = players_df[players_df['Player']==new_player]['Proj 23']
            roster.at[roster_row_index, 'points/$'] = players_df[players_df['Player']==new_player]['points/$']
            #calculate points and spent budget
            points_game = roster["Proj 23"].sum()/17
            spent_budget = roster['Avg. Salary (AVG)'].sum() + (15-roster_data['Number'].sum())
            available_budget = 200 - spent_budget
            #store iteration information
            new_entry = pd.DataFrame([{'Budget spent': spent_budget, 'Points per Game': points_game}])
            history = pd.concat([history, new_entry], ignore_index=True)
            new_change = pd.DataFrame([max_marginal_improvement_row])
            # Ensure max_marginal_improvement_row is a DataFrame
            new_change = pd.DataFrame([max_marginal_improvement_row])

            # Concatenate with change_history
            change_history = pd.concat([change_history, new_change], ignore_index=True)
   
            new_roster = pd.DataFrame([roster])
            roster_history = pd.concat([roster_history, new_roster], ignore_index=True)
            #apply sensitivity analysis
            for key, dataframe in matrix_storage.items():
                players_df_hardcopy_2= sensitivity(dataframe, players_df_hardcopy_2, max_marginal_improvement_row, count)
        else:
            iterate = False
        count += 1
    
    #Iteration display------------------------------------------------------------------
    changes = pd.DataFrame(change_history)
    
    column_dict = {col: [] for col in ['Player', 'Avg. Salary (AVG)', 'Proj 23']}
    for df in roster_history:
        for col in ['Player', 'Avg. Salary (AVG)', 'Proj 23']:
            column_dict[col].extend(df[col])
    roster_evolution = pd.DataFrame(column_dict)
    return roster

# Button to run the program
if st.button('Run Program'):
    # Process data based on inputs
    result_df = run_optimizer(roster_data, scoring, players_df)
    st.write('Results:')
    st.dataframe(result_df, width=700, height=300)

# Option to reset or change inputs
if st.button('Reset'):
    st.experimental_rerun()

