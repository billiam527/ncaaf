# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:20:33 2023

@author: wfish
"""

import pandas as pd
import numpy as np


def edit_sbr_tbl(file_path,
                 new_file_path):
    
    spreads_and_scores = pd.read_csv(file_path)

    spreads_and_scores = spreads_and_scores.replace('pk', '0')
    spreads_and_scores = spreads_and_scores.replace('PK', '0')
    spreads_and_scores = spreads_and_scores.replace('7-105', '7')
    spreads_and_scores = spreads_and_scores.replace('3-115', '3')
    spreads_and_scores = spreads_and_scores.replace('3-105', '3')
    spreads_and_scores = spreads_and_scores.replace('1..5', '1.5')
    spreads_and_scores = spreads_and_scores.replace('24,5', '24.5')
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Home_Final'] != '&nbsp;']
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Home_Final'] != 'NL']
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Away_Final'] != '&nbsp;']
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Away_Final'] != 'NL']
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Home_Close'] != '&nbsp;']
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Home_Close'] != 'NL']
    spreads_and_scores = spreads_and_scores[spreads_and_scores['Away_Close'] != 'NL']
    
    spreads_and_scores["Home_Close"] = pd.to_numeric(spreads_and_scores["Home_Close"])
    spreads_and_scores["Away_Close"] = pd.to_numeric(spreads_and_scores["Away_Close"])
    spreads_and_scores["Home_Final"] = pd.to_numeric(spreads_and_scores["Home_Final"])
    spreads_and_scores["Away_Final"] = pd.to_numeric(spreads_and_scores["Away_Final"])
    
    
    spreads_and_scores['Spread'] = np.where((spreads_and_scores['Home_Close'] < spreads_and_scores['Away_Close']), 
                                            spreads_and_scores['Home_Close']*-1, spreads_and_scores['Away_Close'])
    
    spreads_and_scores['Home_Score_Dif'] = spreads_and_scores['Home_Final'] - spreads_and_scores['Away_Final']
    spreads_and_scores['Dif_Check_Col'] = -spreads_and_scores['Home_Score_Dif']
    spreads_and_scores['Spread_minus_Check_Col'] = spreads_and_scores['Spread'] - spreads_and_scores['Dif_Check_Col']
    
    spreads_and_scores.to_csv(new_file_path)
    

if __name__ == "__main__":

    edit_sbr_tbl('sbr_tbl.csv', 'editted_sbr_tbl.csv')