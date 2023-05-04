# -*- coding: utf-8 -*-
"""
Operators functions for Data Analyst
@author: asngar
"""
# ******
    ## -- Packages --

import pandas as pd 
import numpy as np
import seaborn as sns

    ## -- function which delete all variables with more than 90% of missing value --
    
def remov_mv90(df): 
    print("New data with only less than 90% of missing values ")
    df_new = df[df.columns[df.isna().sum()/df.shape[0] < 0.9]]
    return df_new

    ## -- function which delete all ouliers values --

def outliers(df,target):
    Q1 = df["target"].quantile(0.25)
    Q3 = df["target"].quantile(0.75)
    IQR = Q3 - Q1
    No_outliers = df[~((df["target"] < (Q1-1.5*IQR)) | (df["target"] > (Q3 + 1.5*IQR)))]
    return No_outliers

    ## -- function which split dataset according the target categori --
def sub_target_df(df,target):
    target_yes = df[df[target] == "positive"]  
    target_no = df[df[target] == "negative"]
    return   target_yes, target_n